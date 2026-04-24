#!/usr/bin/env python3
"""Unified ablation comparison for retrieval and triplet classification.

This script evaluates six model settings under the same runtime optical
window, shared gallery construction, and shared evaluation data:

1. optical-only
2. w/o contrastive learning
3. w/o cross-attention
4. w/o fusion branch (contrastive-only scoring)
5. full multimodal (w/o hard mining)
6. full multimodal + hard mining

Outputs a single JSON artifact containing paper-table-ready retrieval rows,
per-model retrieval metrics, and multimodal triplet-classification metrics.
"""

import argparse
import gc
import h5py
import json
import os
import random
import re
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import (  # noqa: E402
    RelationalHDF5Dataset,
    BalancedGWBatchedSampler,
    _build_dataloader,
    _read_root_time_window_attrs,
    build_effective_input_window_metadata,
    build_gw_to_lc_mapping,
)
from model import OpticalKNClassifier, migrate_time_embed_state_dict  # noqa: E402
from retrieval_gallery import (  # noqa: E402
    aggregate_gallery_outcomes,
    build_comparison_model_specs,
    build_curve_rows,
    build_prefixed_gallery_specs,
    build_time_sky_candidate_sequences,
    plot_retrieval_coverage,
    plot_retrieval_curves,
    score_all_galleries_skymap,
)
from test_evaluate import (  # noqa: E402
    _autocast_context,
    _build_gallery_query_cache,
    _compute_credible_level_single_gw,
    _is_dual_fusion_model,
    _lookup_source_type,
    _model_requires_cred_level,
    _resolve_eval_amp,
    apply_time_offsets,
    compute_time_delta_days,
    EvalNegativeTimeOffsetPolicy,
    build_ref_time,
    evaluate_classification_triplet,
    evaluate_classification_triplet_by_source,
    extract_triplet_logits,
    load_gw_event_time_mjd_table,
    load_gw_source_types,
    load_model,
    load_negative_gw_indices,
    load_negative_optical_samples,
    parse_day_windows,
    parse_dt_bin_edges,
)


DEFAULT_COMPARISON_WINDOW = (-0.1, 0.2)
DEFAULT_DT_BIN_EDGES = "0,3,5,10,30,100,300,inf"
TABLE_METRIC_LABELS = ["R@1", "R@5", "R@10", "MRR"]
TABLE_METRIC_KEYS = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
WORKSPACE_DIR = MODEL_DIR.parent.parent
DEFAULT_TUTORIAL_NEG_DATA_PATH = str(
    (WORKSPACE_DIR / "data" / "Optical_Only_dataset" / "Tutorial_negative_dataset.h5").resolve()
)
DEFAULT_TUTORIAL_NEG_GROUP = "Tutorial/optical_data"


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(base_dir: Path, value: Optional[str]) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _parse_gallery_sizes(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    text = str(value or "100,500")
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or [100, 500]


def _parse_n_neg_samples(value: Any, default: int = -1) -> int:
    if value is None:
        return int(default)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "all", "full"}:
            return -1
    out = int(value)
    return -1 if out <= 0 else out


def _resolve_target_samples(n_neg_samples: int) -> Optional[int]:
    value = int(n_neg_samples)
    return value if value > 0 else None


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _resolve_checkpoint_path(checkpoint_path: str, model_type: str) -> str:
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        return str(path.resolve())
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Unsupported checkpoint path: {checkpoint_path}")

    preferred_names = ["optical_only_best.pth", "best.pth"] if model_type == "optical" else ["albef_best.pth", "best.pth"]
    for filename in preferred_names:
        matches = sorted(path.rglob(filename))
        if matches:
            return str(matches[0].resolve())

    epoch_candidates: List[Tuple[int, Path]] = []
    epoch_patterns = [
        re.compile(r"albef_epoch_(\d+)\.pth$"),
        re.compile(r"optical_only_epoch_(\d+)\.pth$"),
    ]
    for candidate in path.rglob("*.pth"):
        for pattern in epoch_patterns:
            match = pattern.fullmatch(candidate.name)
            if match:
                epoch_candidates.append((int(match.group(1)), candidate))
                break
    if epoch_candidates:
        return str(max(epoch_candidates, key=lambda item: item[0])[1].resolve())

    fallback_names = ["optical_only_last.pth", "albef_last.pth", "last.pth"]
    for filename in fallback_names:
        matches = sorted(path.rglob(filename))
        if matches:
            return str(matches[0].resolve())

    any_pth = sorted(path.rglob("*.pth"))
    if any_pth:
        return str(any_pth[0].resolve())

    raise FileNotFoundError(f"No checkpoint file found under directory: {checkpoint_path}")


def _build_model_specs(cfg: Dict[str, Any], cfg_dir: Path) -> List[Dict[str, Any]]:
    return build_comparison_model_specs(cfg, cfg_dir)


def load_optical_model(checkpoint_path: str, device: torch.device) -> Tuple[OpticalKNClassifier, Dict[str, Any]]:
    """Load optical-only model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    if isinstance(ckpt_args, argparse.Namespace):
        ckpt_args = vars(ckpt_args)

    model = OpticalKNClassifier(
        optical_input_dim=int(ckpt_args.get("optical_input_dim", 6)),
        ref_time_dim=int(ckpt_args.get("n_ref", 64)),
        enc_dim=int(ckpt_args.get("enc_dim", 128)),
        num_heads=int(ckpt_args.get("num_heads", 4)),
        k_dim=int(ckpt_args.get("k_dim", 64)),
        opt_dropout=0.0,
        feature_dropout=0.0,
        head_hidden_dim=ckpt_args.get("head_hidden_dim"),
        head_dropout=0.0,
        universal_aux_enable=bool(ckpt_args.get("universal_aux_enable", False)),
        proj_dim=int(ckpt_args.get("proj_dim", 64)),
        grl_lambda=float(ckpt_args.get("grl_lambda", 1.0)),
        mtan_period_range_days=tuple(ckpt_args.get("mtan_period_range_days", (0.5, 100.0))),
        mtan_time_scale_divisor=float(ckpt_args.get("mtan_time_scale_divisor", 100.0)),
    )

    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict = migrate_time_embed_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"  Loaded optical-only model from {checkpoint_path}")
    return model, ckpt_args


def load_multimodal_bundle(
    checkpoint_path: str,
    config_path: str,
    device: torch.device,
    *,
    test_data_path: str,
    neg_data_path: Optional[str],
    comparison_window: Tuple[float, float],
    nonkn_cls_base_field: str,
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """Load multimodal model and build runtime eval args for comparison."""
    args_ns = SimpleNamespace(
        checkpoint=checkpoint_path,
        config=config_path,
        test_data_path=test_data_path,
        nonkn_cls_base_field=nonkn_cls_base_field,
    )
    model, model_args, saved_args = load_model(args_ns, device)
    if isinstance(saved_args, argparse.Namespace):
        saved_args = vars(saved_args)
    saved_args = dict(saved_args or {})

    original_ref_start = float(model_args.get("ref_start", comparison_window[0]))
    original_ref_end = float(model_args.get("ref_end", comparison_window[1]))
    dataset_window_start, dataset_window_end = _read_root_time_window_attrs(test_data_path)

    runtime_model_args = dict(model_args)
    runtime_model_args["original_ref_start"] = original_ref_start
    runtime_model_args["original_ref_end"] = original_ref_end
    runtime_model_args["ref_start"] = float(comparison_window[0])
    runtime_model_args["ref_end"] = float(comparison_window[1])
    runtime_model_args["nonkn_cls_base_field"] = str(nonkn_cls_base_field)
    runtime_model_args["dataset_window_metadata"] = {
        "positive": _read_root_time_window_attrs(test_data_path),
        "negative": _read_root_time_window_attrs(neg_data_path),
    }
    runtime_model_args["effective_input_window_metadata"] = build_effective_input_window_metadata(
        original_ref_start,
        original_ref_end,
        runtime_input_window_start=float(comparison_window[0]),
        runtime_input_window_end=float(comparison_window[1]),
        dataset_window_start=dataset_window_start,
        dataset_window_end=dataset_window_end,
    )
    return model, runtime_model_args, saved_args



def _multimodal_has_cls_head(saved_args: Dict[str, Any]) -> bool:
    return float(saved_args.get("cls_weight", 0.0) or 0.0) > 0.0


def _resolve_multimodal_scoring(model_spec: Dict[str, Any], saved_args: Dict[str, Any]) -> str:
    requested = str(model_spec.get("scoring", "auto")).strip().lower()
    has_cls_head = _multimodal_has_cls_head(saved_args)
    if not has_cls_head:
        return "contrastive"
    if requested in {"", "auto"}:
        return "logits"
    return requested


def _build_test_loader(
    test_data_path: str,
    neg_data_path: Optional[str],
    neg_group: str,
    ref_start: float,
    ref_end: float,
    *,
    batch_size: int,
    test_steps: Optional[int],
    num_workers: int,
    target_samples: Optional[int] = None,
    nonkn_cls_base_field: str = "zero_time_mjd_base",
    return_zero_time_mjd: bool = False,
):
    """Build comparison dataloader using shared runtime crop."""
    dataset = RelationalHDF5Dataset(
        test_data_path,
        negative_h5_path=neg_data_path,
        negative_group=neg_group,
        return_zero_time_mjd=bool(return_zero_time_mjd),
        nonkn_cls_base_field=str(nonkn_cls_base_field),
        opt_input_window_start=ref_start,
        opt_input_window_end=ref_end,
    )
    gw_to_lc = build_gw_to_lc_mapping(test_data_path)
    n_gw = len(gw_to_lc)
    bs = min(int(batch_size), n_gw)
    if bs <= 0:
        raise ValueError("No GW events found in test dataset")

    if test_steps is None:
        if target_samples is None:
            steps = 5
        else:
            steps = max(1, (int(target_samples) + bs - 1) // bs)
    else:
        steps = int(test_steps)

    sampler = BalancedGWBatchedSampler(gw_to_lc, batch_size=bs, steps_per_epoch=steps)
    loader = _build_dataloader(
        dataset,
        sampler,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
        prefetch_factor=2 if int(num_workers) > 0 else None,
    )
    print(f"  Test sampling: {steps} steps x {bs} batch_size = ~{steps * bs} samples")
    return loader


def cache_test_batches(loader):
    """Iterate loader once, cache all batch raw tensors on CPU."""
    cached = []
    for batch_data in tqdm(loader, desc="  Caching test batches"):
        cached.append(tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in batch_data))
    return cached


def build_cached_batches(
    *,
    test_data_path: str,
    neg_data_path: Optional[str],
    neg_group: str,
    comparison_window: Tuple[float, float],
    batch_size: int,
    test_steps: Optional[int],
    num_workers: int,
    target_samples: Optional[int],
    nonkn_cls_base_field: str,
) -> Tuple[List[Tuple[Any, ...]], int]:
    """Build and cache retrieval batches, retrying with single-worker if needed."""
    try_num_workers = int(num_workers)
    while True:
        try:
            _seed_all(42)
            loader = _build_test_loader(
                test_data_path,
                neg_data_path,
                neg_group,
                comparison_window[0],
                comparison_window[1],
                batch_size=batch_size,
                test_steps=test_steps,
                num_workers=try_num_workers,
                target_samples=target_samples,
                nonkn_cls_base_field=nonkn_cls_base_field,
            )
            return cache_test_batches(loader), try_num_workers
        except PermissionError:
            if try_num_workers <= 0:
                raise
            print("WARNING: DataLoader multiprocessing failed. Retrying retrieval cache with num_workers=0.")
            try_num_workers = 0


def load_test_positive_index_map(test_data_path: str) -> Tuple[Dict[int, np.ndarray], List[int]]:
    with h5py.File(test_data_path, "r") as f:
        if "events/optical_data/parent_gw_idx" not in f:
            raise KeyError(f"Missing required field 'events/optical_data/parent_gw_idx' in {test_data_path}")
        parent_gw_idx = np.asarray(f["events/optical_data/parent_gw_idx"][:], dtype=np.int64)

    gw_positive_indices: Dict[int, List[int]] = {}
    for opt_idx, gw_id in enumerate(parent_gw_idx.tolist()):
        gw_positive_indices.setdefault(int(gw_id), []).append(int(opt_idx))

    final_map = {
        int(gw_id): np.asarray(indices, dtype=np.int64)
        for gw_id, indices in gw_positive_indices.items()
    }
    return final_map, sorted(final_map.keys())


def load_selected_positive_bank(
    test_data_path: str,
    selected_optical_indices: np.ndarray,
) -> Tuple[Dict[str, torch.Tensor], Dict[int, int]]:
    selected = np.unique(np.asarray(selected_optical_indices, dtype=np.int64).reshape(-1))
    if selected.size == 0:
        raise ValueError("selected_optical_indices must be non-empty to build a positive query bank.")

    with h5py.File(test_data_path, "r") as f:
        if "events/optical_data" not in f:
            raise KeyError(f"Missing required group 'events/optical_data' in {test_data_path}")
        opt = f["events/optical_data"]
        required_fields = ["times", "values", "masks", "errors", "coordinates", "parent_gw_idx"]
        for field in required_fields:
            if field not in opt:
                raise KeyError(f"Missing required field 'events/optical_data/{field}' in {test_data_path}")

        opt_t_raw = torch.from_numpy(np.asarray(opt["times"][selected], dtype=np.float32))
        opt_v_raw = torch.from_numpy(np.asarray(opt["values"][selected], dtype=np.float32))
        opt_mask_raw = torch.from_numpy(np.asarray(opt["masks"][selected], dtype=np.float32))
        opt_err_raw = torch.from_numpy(np.asarray(opt["errors"][selected], dtype=np.float32))
        opt_coords = torch.from_numpy(np.asarray(opt["coordinates"][selected], dtype=np.float32))
        gw_indices = torch.from_numpy(np.asarray(opt["parent_gw_idx"][selected], dtype=np.int64))

    bank: Dict[str, torch.Tensor] = {
        "times": opt_t_raw,
        "values": opt_v_raw,
        "masks": opt_mask_raw,
        "errors": opt_err_raw,
        "coordinates": opt_coords,
        "gw_indices": gw_indices,
        "source_optical_indices": torch.from_numpy(selected.astype(np.int64, copy=False)),
        "opt_t_raw": opt_t_raw,
        "opt_v_raw": opt_v_raw,
        "opt_mask_raw": opt_mask_raw,
        "opt_err_raw": opt_err_raw,
        "opt_coords": opt_coords,
    }
    remap = {int(source_idx): int(compact_idx) for compact_idx, source_idx in enumerate(selected.tolist())}
    return bank, remap


def remap_gallery_positive_indices(
    galleries: Mapping[Tuple[int, int, int], Mapping[str, Any]],
    positive_index_remap: Mapping[int, int],
) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    remapped: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for key, gallery_spec in galleries.items():
        source_positive_index = int(gallery_spec["positive_index"])
        if source_positive_index not in positive_index_remap:
            raise KeyError(f"Gallery positive index {source_positive_index} missing from compact positive-bank remap.")
        updated_spec = dict(gallery_spec)
        updated_spec["source_positive_index"] = source_positive_index
        updated_spec["positive_index"] = int(positive_index_remap[source_positive_index])
        remapped[key] = updated_spec
    return remapped


@torch.no_grad()
def extract_gallery_embeddings(
    model,
    cached_batches,
    device,
    *,
    n_ref: int,
    ref_start: float,
    ref_end: float,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
):
    """Extract optical-side embeddings and raw tensors for gallery scoring."""
    all_h_l = []
    all_z_l = []
    all_opt_coords = []
    all_gw_indices = []
    all_opt_t = []
    all_opt_v = []
    all_opt_mask = []
    all_opt_err = []
    ref_cache = None

    for batch_data in tqdm(cached_batches, desc="  Extracting embeddings"):
        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data[:8]

        gw_s = gw_s.to(device)
        gw_m = gw_m.to(device)
        opt_t = opt_t.to(device)
        opt_v = opt_v.to(device)
        opt_mask = opt_mask.to(device)
        opt_err = opt_err.to(device)
        opt_coords = opt_coords.to(device)

        bs = gw_s.size(0)
        if ref_cache is None or ref_cache.shape[0] != bs or ref_cache.dtype != opt_t.dtype:
            ref_cache = build_ref_time(bs, n_ref, ref_start, ref_end, device, opt_t.dtype)

        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            _g, z_l, h_l, _H_gw = model.encode(
                gw_s, gw_m, opt_coords, opt_t, opt_v, ref_cache, opt_mask, opt_err
            )

        all_h_l.append(h_l.float().cpu())
        all_z_l.append(z_l.float().cpu())
        all_opt_coords.append(opt_coords.float().cpu())
        all_gw_indices.append(gw_indices.long().cpu())
        all_opt_t.append(opt_t.float().cpu())
        all_opt_v.append(opt_v.float().cpu())
        all_opt_mask.append(opt_mask.float().cpu())
        all_opt_err.append(opt_err.float().cpu())

    return {
        "h_l_cls": torch.cat(all_h_l),
        "z_l_cls": torch.cat(all_z_l),
        "opt_coords": torch.cat(all_opt_coords),
        "gw_indices": torch.cat(all_gw_indices),
        "opt_t_raw": torch.cat(all_opt_t),
        "opt_v_raw": torch.cat(all_opt_v),
        "opt_mask_raw": torch.cat(all_opt_mask),
        "opt_err_raw": torch.cat(all_opt_err),
        "dual_fusion": _is_dual_fusion_model(model),
    }


@torch.no_grad()
def extract_optical_candidate_embeddings(
    model,
    optical_data,
    device,
    *,
    n_ref: int,
    ref_start: float,
    ref_end: float,
    chunk_size: int = 1024,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
    desc: str = "  Extracting optical candidate embeddings",
):
    """Extract optical embeddings for a preloaded optical candidate bank."""
    core_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    all_h_l = []
    all_z_l = []
    all_opt_coords = []
    all_opt_t = []
    all_opt_v = []
    all_opt_mask = []
    all_opt_err = []

    total = int(optical_data["times"].shape[0])
    for start in tqdm(range(0, total, int(chunk_size)), desc=desc):
        end = min(start + int(chunk_size), total)
        chunk_idx = torch.arange(start, end, dtype=torch.long)
        opt_coords = optical_data["coordinates"].index_select(0, chunk_idx).to(device)
        opt_t = optical_data["times"].index_select(0, chunk_idx).to(device)
        opt_v = optical_data["values"].index_select(0, chunk_idx).to(device)
        opt_mask = optical_data["masks"].index_select(0, chunk_idx).to(device)
        opt_err = optical_data["errors"].index_select(0, chunk_idx).to(device)
        ref_time = build_ref_time(opt_t.size(0), n_ref, ref_start, ref_end, device, opt_t.dtype)
        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            z_l, h_l = core_model.encode_optical(
                opt_coords, opt_t, opt_v, ref_time, opt_mask, opt_err
            )

        all_h_l.append(h_l.float().cpu())
        all_z_l.append(z_l.float().cpu())
        all_opt_coords.append(opt_coords.float().cpu())
        all_opt_t.append(opt_t.float().cpu())
        all_opt_v.append(opt_v.float().cpu())
        all_opt_mask.append(opt_mask.float().cpu())
        all_opt_err.append(opt_err.float().cpu())

    result = {
        "h_l_cls": torch.cat(all_h_l),
        "z_l_cls": torch.cat(all_z_l),
        "opt_coords": torch.cat(all_opt_coords),
        "opt_t_raw": torch.cat(all_opt_t),
        "opt_v_raw": torch.cat(all_opt_v),
        "opt_mask_raw": torch.cat(all_opt_mask),
        "opt_err_raw": torch.cat(all_opt_err),
    }
    if "gw_indices" in optical_data:
        gw_indices = optical_data["gw_indices"]
        result["gw_indices"] = (
            gw_indices.detach().cpu().to(torch.long)
            if isinstance(gw_indices, torch.Tensor)
            else torch.as_tensor(gw_indices, dtype=torch.long)
        )
    if "source_optical_indices" in optical_data:
        source_indices = optical_data["source_optical_indices"]
        result["source_optical_indices"] = (
            source_indices.detach().cpu().to(torch.long)
            if isinstance(source_indices, torch.Tensor)
            else torch.as_tensor(source_indices, dtype=torch.long)
        )
    return result


def precompute_tutorial_galleries(
    gw_positive_indices: Mapping[int, Any],
    n_tutorial_negatives: int,
    gallery_sizes,
    n_trials,
    seed,
):
    """Build galleries with one matched KN and tutorial non-KN distractors."""
    rng = np.random.default_rng(int(seed))
    unique_gw = sorted(int(gw_id) for gw_id in gw_positive_indices.keys())
    galleries: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    neg_pool = np.arange(int(n_tutorial_negatives), dtype=np.int64)

    for gallery_size in gallery_sizes:
        for trial in range(int(n_trials)):
            for gw_id in unique_gw:
                gw_idxs = np.asarray(gw_positive_indices.get(int(gw_id), []), dtype=np.int64).reshape(-1)
                if gw_idxs.size == 0:
                    raise ValueError(f"GW event {gw_id} has no positive optical samples available for gallery construction.")
                correct_idx = int(gw_idxs[rng.integers(gw_idxs.size)])
                if int(gallery_size) <= 1:
                    galleries[(int(gallery_size), int(trial), int(gw_id))] = {
                        "positive_index": correct_idx,
                        "negative_indices": np.empty((0,), dtype=np.int64),
                        "negative_credible_levels": np.empty((0,), dtype=np.float32),
                        "negative_abs_dt_days": np.empty((0,), dtype=np.float32),
                        "requested_gallery_size": int(gallery_size),
                        "actual_gallery_size": 1,
                        "coverage_met": True,
                        "is_undersized": False,
                    }
                    continue

                n_distract = min(int(gallery_size) - 1, len(neg_pool))
                if n_distract <= 0:
                    continue
                distract_idxs = rng.choice(neg_pool, size=n_distract, replace=False)
                galleries[(int(gallery_size), int(trial), int(gw_id))] = {
                    "positive_index": correct_idx,
                    "negative_indices": np.asarray(distract_idxs, dtype=np.int64),
                    "negative_credible_levels": np.full((n_distract,), np.nan, dtype=np.float32),
                    "negative_abs_dt_days": np.full((n_distract,), np.nan, dtype=np.float32),
                    "requested_gallery_size": int(gallery_size),
                    "actual_gallery_size": 1 + int(n_distract),
                    "coverage_met": True,
                    "is_undersized": False,
                }

    return galleries, unique_gw


def enrich_gallery_outcomes(ranks: Mapping[Tuple[int, int, int], int], galleries: Mapping[Tuple[int, int, int], Mapping[str, Any]]) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    outcomes: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for key, rank in ranks.items():
        gallery_spec = galleries[key]
        outcomes[key] = {
            "rank": int(rank),
            "requested_gallery_size": int(gallery_spec.get("requested_gallery_size", key[0])),
            "actual_gallery_size": int(gallery_spec.get("actual_gallery_size", key[0])),
            "coverage_met": bool(gallery_spec.get("coverage_met", True)),
            "is_undersized": bool(gallery_spec.get("is_undersized", False)),
        }
    return outcomes


@torch.no_grad()
def score_gallery_optical_only(
    opt_model: OpticalKNClassifier,
    candidate_indices: np.ndarray,
    opt_t_all: torch.Tensor,
    opt_v_all: torch.Tensor,
    opt_mask_all: torch.Tensor,
    opt_err_all: torch.Tensor,
    device: torch.device,
    *,
    n_ref: int,
    ref_start: float,
    ref_end: float,
    chunk_size: int = 1024,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
) -> np.ndarray:
    """Score gallery candidates using optical-only model p(KN|optical)."""
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.float32)

    scores = []
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    for start in range(0, len(candidate_indices), int(chunk_size)):
        chunk_np = candidate_indices[start:start + int(chunk_size)]
        chunk_idx = torch.from_numpy(chunk_np).long()
        batch_size = len(chunk_np)

        opt_t = opt_t_all.index_select(0, chunk_idx).to(device)
        opt_v = opt_v_all.index_select(0, chunk_idx).to(device)
        opt_mask = opt_mask_all.index_select(0, chunk_idx).to(device)
        opt_err = opt_err_all.index_select(0, chunk_idx).to(device)
        ref_time = build_ref_time(batch_size, n_ref, ref_start, ref_end, device, opt_t.dtype)

        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            logits = opt_model(opt_t, opt_v, ref_time, opt_mask, opt_err).squeeze(-1)

        scores.append(torch.sigmoid(logits.float()).cpu())

    return torch.cat(scores).numpy()



@torch.no_grad()
def extract_negative_gallery_embeddings(
    model,
    neg_optical_data,
    device,
    *,
    n_ref: int,
    ref_start: float,
    ref_end: float,
    chunk_size: int = 1024,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
):
    """Extract optical embeddings for tutorial negative distractors."""
    if neg_optical_data is None:
        raise ValueError("Tutorial negative optical data is required for gallery distractors.")

    core_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    all_h_l = []
    all_z_l = []
    all_opt_coords = []
    all_opt_t = []
    all_opt_v = []
    all_opt_mask = []
    all_opt_err = []

    total = int(neg_optical_data["times"].shape[0])
    for start in tqdm(range(0, total, int(chunk_size)), desc="  Extracting tutorial distractor embeddings"):
        end = min(start + int(chunk_size), total)
        chunk_idx = torch.arange(start, end, dtype=torch.long)
        opt_coords = neg_optical_data["coordinates"].index_select(0, chunk_idx).to(device)
        opt_t = neg_optical_data["times"].index_select(0, chunk_idx).to(device)
        opt_v = neg_optical_data["values"].index_select(0, chunk_idx).to(device)
        opt_mask = neg_optical_data["masks"].index_select(0, chunk_idx).to(device)
        opt_err = neg_optical_data["errors"].index_select(0, chunk_idx).to(device)
        ref_time = build_ref_time(opt_t.size(0), n_ref, ref_start, ref_end, device, opt_t.dtype)
        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            z_l, h_l = core_model.encode_optical(
                opt_coords, opt_t, opt_v, ref_time, opt_mask, opt_err
            )

        all_h_l.append(h_l.float().cpu())
        all_z_l.append(z_l.float().cpu())
        all_opt_coords.append(opt_coords.float().cpu())
        all_opt_t.append(opt_t.float().cpu())
        all_opt_v.append(opt_v.float().cpu())
        all_opt_mask.append(opt_mask.float().cpu())
        all_opt_err.append(opt_err.float().cpu())

    return {
        "h_l_cls": torch.cat(all_h_l),
        "z_l_cls": torch.cat(all_z_l),
        "opt_coords": torch.cat(all_opt_coords),
        "opt_t_raw": torch.cat(all_opt_t),
        "opt_v_raw": torch.cat(all_opt_v),
        "opt_mask_raw": torch.cat(all_opt_mask),
        "opt_err_raw": torch.cat(all_opt_err),
    }


@torch.no_grad()
def _score_candidate_bank_with_logits(
    model,
    query_cache,
    candidate_indices: np.ndarray,
    candidate_bank: Dict[str, torch.Tensor],
    device: torch.device,
    dual: bool,
    *,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
) -> np.ndarray:
    """Score a query GW against an optical candidate bank with fusion logits."""
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.float32)

    need_cred = _model_requires_cred_level(model)
    g_query = query_cache["g"].to(device)
    gw_s_query = query_cache["gw_s"].to(device)
    scores = []
    if dual:
        H_query = query_cache["H_gw"].to(device)
        gw_m_query = query_cache["gw_m"].to(device)

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    for start in range(0, len(candidate_indices), 1024):
        chunk_np = candidate_indices[start:start + 1024]
        chunk_idx = torch.from_numpy(chunk_np).long()
        h_chunk = candidate_bank["h_l_cls"].index_select(0, chunk_idx).to(device)
        opt_coords_chunk = candidate_bank["opt_coords"].index_select(0, chunk_idx).to(device)
        z_chunk = candidate_bank["z_l_cls"].index_select(0, chunk_idx).to(device) if dual else None

        batch_size = h_chunk.size(0)
        g_chunk = g_query.unsqueeze(0).expand(batch_size, -1)
        if dual:
            H_chunk = H_query.unsqueeze(0).expand(batch_size, -1, -1)
            gw_s_chunk = gw_s_query.unsqueeze(0).expand(batch_size, -1)
            gw_m_chunk = gw_m_query.unsqueeze(0).expand(batch_size, -1, -1)
            cred_chunk = _compute_credible_level_single_gw(gw_m_query, opt_coords_chunk) if need_cred else None
        else:
            H_chunk = None
            gw_s_chunk = None
            gw_m_chunk = None
            cred_chunk = None

        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            logits = model.fusion_logits(
                g_chunk,
                h_chunk,
                z_l=z_chunk,
                H_gw=H_chunk,
                cred_level=cred_chunk,
                gw_s=gw_s_chunk,
                gw_m=gw_m_chunk,
                opt_coords=opt_coords_chunk,
            )
            probs = torch.softmax(logits, dim=1)[:, 1]
        scores.append(probs.float().cpu())

    return torch.cat(scores).numpy()


@torch.no_grad()
def _score_candidate_bank_contrastive(
    model,
    query_cache,
    candidate_indices: np.ndarray,
    candidate_bank: Dict[str, torch.Tensor],
    device: torch.device,
    *,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
) -> np.ndarray:
    """Score a query GW against an optical candidate bank with contrastive similarity."""
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.float32)

    core_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    g_query = query_cache["g"].to(device).unsqueeze(0)
    with _autocast_context(device, amp_dtype, enabled=amp_enabled):
        feat_g = core_model.project_gw_features(g_query)

    scores = []
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    for start in range(0, len(candidate_indices), 1024):
        chunk_np = candidate_indices[start:start + 1024]
        chunk_idx = torch.from_numpy(chunk_np).long()
        z_chunk = candidate_bank["z_l_cls"].index_select(0, chunk_idx).to(device)
        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            feat_o = core_model.project_optical_features(z_chunk)
        sims = torch.matmul(feat_g, feat_o.T).squeeze(0)
        scores.append(sims.float().cpu())

    return torch.cat(scores).numpy()


@torch.no_grad()
def score_all_galleries_multimodal(
    model,
    positive_bank,
    negative_bank,
    galleries,
    unique_gw,
    *,
    test_data_path: str,
    device: torch.device,
    model_args: Dict[str, Any],
    gw_event_time_mjd_table=None,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
):
    """Score galleries with one matched KN and tutorial distractors using fusion logits."""
    dual = bool(positive_bank.get("dual_fusion", _is_dual_fusion_model(model)))
    print("  Building GW query cache...")
    query_cache = _build_gallery_query_cache(
        model,
        unique_gw,
        test_data_path,
        device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )

    ranks: Dict[Tuple[int, int, int], int] = {}
    for key, gallery_spec in tqdm(galleries.items(), desc="  Scoring galleries"):
        _gallery_size, _trial, gw_id = key
        pos_index = int(gallery_spec["positive_index"])
        neg_indices = np.asarray(gallery_spec["negative_indices"], dtype=np.int64)
        pos_probs = _score_candidate_bank_with_logits(
            model,
            query_cache[int(gw_id)],
            np.asarray([pos_index], dtype=np.int64),
            positive_bank,
            device,
            dual,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        neg_probs = _score_candidate_bank_with_logits(
            model,
            query_cache[int(gw_id)],
            neg_indices,
            negative_bank,
            device,
            dual,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        probs = np.concatenate([pos_probs, neg_probs], axis=0)
        ranked = np.argsort(-probs)
        ranks[key] = int(np.where(ranked == 0)[0][0])

    return ranks


@torch.no_grad()
def score_all_galleries_contrastive(
    model,
    positive_bank,
    negative_bank,
    galleries,
    unique_gw,
    *,
    test_data_path: str,
    device: torch.device,
    model_args: Dict[str, Any],
    gw_event_time_mjd_table=None,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
):
    """Score galleries with one matched KN and tutorial distractors using contrastive similarity."""
    print("  Building GW query cache...")
    query_cache = _build_gallery_query_cache(
        model,
        unique_gw,
        test_data_path,
        device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )

    ranks: Dict[Tuple[int, int, int], int] = {}
    for key, gallery_spec in tqdm(galleries.items(), desc="  Scoring galleries (contrastive)"):
        _gallery_size, _trial, gw_id = key
        pos_index = int(gallery_spec["positive_index"])
        neg_indices = np.asarray(gallery_spec["negative_indices"], dtype=np.int64)
        pos_sims = _score_candidate_bank_contrastive(
            model,
            query_cache[int(gw_id)],
            np.asarray([pos_index], dtype=np.int64),
            positive_bank,
            device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        neg_sims = _score_candidate_bank_contrastive(
            model,
            query_cache[int(gw_id)],
            neg_indices,
            negative_bank,
            device,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        sims = np.concatenate([pos_sims, neg_sims], axis=0)
        ranked = np.argsort(-sims)
        ranks[key] = int(np.where(ranked == 0)[0][0])

    return ranks


@torch.no_grad()
def score_all_galleries_optical(
    opt_model,
    positive_raw_opt,
    negative_raw_opt,
    galleries,
    *,
    device: torch.device,
    n_ref: int,
    ref_start: float,
    ref_end: float,
    amp_dtype: torch.dtype = torch.float32,
    amp_enabled: bool = False,
):
    """Score galleries with one matched KN and tutorial distractors using optical-only logits."""
    ranks: Dict[Tuple[int, int, int], int] = {}
    for key, gallery_spec in tqdm(galleries.items(), desc="  Scoring galleries"):
        pos_index = int(gallery_spec["positive_index"])
        neg_indices = np.asarray(gallery_spec["negative_indices"], dtype=np.int64)
        pos_probs = score_gallery_optical_only(
            opt_model,
            np.asarray([pos_index], dtype=np.int64),
            positive_raw_opt["opt_t_raw"],
            positive_raw_opt["opt_v_raw"],
            positive_raw_opt["opt_mask_raw"],
            positive_raw_opt["opt_err_raw"],
            device,
            n_ref=n_ref,
            ref_start=ref_start,
            ref_end=ref_end,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        neg_probs = score_gallery_optical_only(
            opt_model,
            neg_indices,
            negative_raw_opt["times"],
            negative_raw_opt["values"],
            negative_raw_opt["masks"],
            negative_raw_opt["errors"],
            device,
            n_ref=n_ref,
            ref_start=ref_start,
            ref_end=ref_end,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        probs = np.concatenate([pos_probs, neg_probs], axis=0)
        ranked = np.argsort(-probs)
        ranks[key] = int(np.where(ranked == 0)[0][0])
    return ranks


def aggregate_metrics(ranks, gallery_sizes, n_trials, unique_gw, *, gw_source_map=None):
    """Convert per-gallery ranks to Recall@K and MRR metrics."""
    metrics: Dict[str, float] = {}
    by_source: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for gallery_size in gallery_sizes:
        recalls = {1: [], 5: [], 10: []}
        mrrs: List[float] = []
        for trial in range(int(n_trials)):
            for gw_id in unique_gw:
                key = (int(gallery_size), int(trial), int(gw_id))
                if key not in ranks:
                    continue
                rank = int(ranks[key])
                for k in recalls:
                    recalls[k].append(1.0 if rank < k else 0.0)
                mrrs.append(1.0 / float(rank + 1))

                if gw_source_map:
                    source_label = gw_source_map.get(int(gw_id), "unknown")
                    if source_label not in by_source:
                        by_source[source_label] = {}
                    if int(gallery_size) not in by_source[source_label]:
                        by_source[source_label][int(gallery_size)] = {
                            "recalls": {1: [], 5: [], 10: []},
                            "mrrs": [],
                        }
                    source_bucket = by_source[source_label][int(gallery_size)]
                    for k in source_bucket["recalls"]:
                        source_bucket["recalls"][k].append(1.0 if rank < k else 0.0)
                    source_bucket["mrrs"].append(1.0 / float(rank + 1))

        for k, values in recalls.items():
            metrics[f"gallery_{gallery_size}_recall_at_{k}"] = float(np.mean(values)) if values else 0.0
        metrics[f"gallery_{gallery_size}_mrr"] = float(np.mean(mrrs)) if mrrs else 0.0

    source_metrics: Dict[str, Dict[str, float]] = {}
    for source_label, size_acc in by_source.items():
        src_out: Dict[str, float] = {}
        for gallery_size, acc in size_acc.items():
            for k, values in acc["recalls"].items():
                src_out[f"gallery_{gallery_size}_recall_at_{k}"] = float(np.mean(values)) if values else 0.0
            src_out[f"gallery_{gallery_size}_mrr"] = float(np.mean(acc["mrrs"])) if acc["mrrs"] else 0.0
        source_metrics[source_label] = src_out

    return metrics, source_metrics


def print_unified_table(all_results, gallery_sizes, model_names):
    """Print comparison table matching the paper format."""
    col_w = 7
    name_w = max(max(len(name) for name in model_names), len("Method")) + 2
    group_w = col_w * len(TABLE_METRIC_LABELS) + (len(TABLE_METRIC_LABELS) - 1)

    header_1 = f"{'Method':<{name_w}}"
    header_2 = f"{'':<{name_w}}"
    for gallery_size in gallery_sizes:
        header_1 += f" | {'gallery=' + str(gallery_size):^{group_w}}"
        for metric_label in TABLE_METRIC_LABELS:
            header_2 += f" | {metric_label:>{col_w - 2}}"

    sep = "-" * len(header_2)
    print(f"\n{sep}")
    print(header_1)
    print(header_2)
    print(sep)

    for name in model_names:
        row_metrics = all_results.get(name, {})
        row = f"{name:<{name_w}}"
        for gallery_size in gallery_sizes:
            for metric_key in TABLE_METRIC_KEYS:
                full_key = f"gallery_{gallery_size}_{metric_key}"
                value = row_metrics.get(full_key)
                if value is None:
                    row += f" | {'--':>{col_w - 2}}"
                else:
                    row += f" | {value:>{col_w - 2}.4f}"
        print(row)

    print(sep)


def print_latex_table(all_results, gallery_sizes, model_names):
    """Print LaTeX-formatted table for the paper."""
    n_metrics = len(TABLE_METRIC_LABELS)
    print("\n% --- LaTeX table ---")
    print("\\begin{table}[htbp]")
    print("\\centering")
    col_spec = "l" + "|".join(["" + "c" * n_metrics for _ in gallery_sizes])
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    header_parts = ["\\multirow{2}{*}{Method}"]
    for gallery_size in gallery_sizes:
        header_parts.append(f"\\multicolumn{{{n_metrics}}}{{c}}{{gallery={gallery_size}}}")
    print(" & ".join(header_parts) + " \\")

    sub_parts = [""]
    for _gallery_size in gallery_sizes:
        sub_parts.extend(TABLE_METRIC_LABELS)
    print(" & ".join(sub_parts) + " \\")
    print("\\midrule")

    best_vals: Dict[str, float] = {}
    for gallery_size in gallery_sizes:
        for metric_key in TABLE_METRIC_KEYS:
            full_key = f"gallery_{gallery_size}_{metric_key}"
            values = [all_results[name].get(full_key, -1.0) for name in model_names if all_results.get(name)]
            if values:
                best_vals[full_key] = max(values)

    for name in model_names:
        row_metrics = all_results.get(name, {})
        parts = [name.replace("_", "\\_")]
        for gallery_size in gallery_sizes:
            for metric_key in TABLE_METRIC_KEYS:
                full_key = f"gallery_{gallery_size}_{metric_key}"
                if not row_metrics:
                    parts.append("--")
                    continue
                value = row_metrics.get(full_key)
                if value is None:
                    parts.append("--")
                    continue
                cell = f"{value:.4f}"
                if abs(value - best_vals.get(full_key, -1.0)) < 1e-6:
                    cell = f"\\textbf{{{cell}}}"
                parts.append(cell)
        print(" & ".join(parts) + " \\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Retrieval comparison across model variants.}")
    print("\\label{tab:retrieval_comparison}")
    print("\\end{table}")
    print("% --- end LaTeX table ---")


def build_table_rows(all_results, gallery_sizes, model_names):
    rows = []
    for name in model_names:
        row = {"method": name}
        row_metrics = all_results.get(name, {})
        for gallery_size in gallery_sizes:
            gallery_block = {}
            for metric_label, metric_key in zip(TABLE_METRIC_LABELS, TABLE_METRIC_KEYS):
                gallery_block[metric_label] = row_metrics.get(f"gallery_{gallery_size}_{metric_key}")
            row[f"gallery_{gallery_size}"] = gallery_block
        rows.append(row)
    return rows


def build_gw_source_map(gw_source_types, gw_ids):
    if gw_source_types is None:
        return {}
    if isinstance(gw_ids, torch.Tensor):
        unique_gw = sorted(torch.unique(gw_ids.detach().cpu().to(torch.long)).tolist())
    else:
        unique_gw = sorted({int(gw_id) for gw_id in gw_ids})
    return {int(gw_id): _lookup_source_type(int(gw_id), gw_source_types) for gw_id in unique_gw}


def build_neg_offset_policy(cfg: Dict[str, Any], seed: int) -> EvalNegativeTimeOffsetPolicy:
    return EvalNegativeTimeOffsetPolicy(
        enabled=_as_bool(cfg.get("neg_time_offset_enable", False)),
        dist_npz=cfg.get("neg_offset_dist_npz"),
        dist_key=str(cfg.get("neg_offset_dist_key", "delta_days_combined")),
        eval_mode=str(cfg.get("neg_offset_eval_mode", "quantile_ensemble")),
        eval_quantiles=str(cfg.get("neg_offset_eval_quantiles", "0.1,0.3,0.5,0.7,0.9")),
        scale_divisor=float(cfg.get("neg_offset_scale_days_divisor", 100.0)),
        seed=int(seed),
        bank_size=int(cfg.get("neg_offset_bank_size", 1000000)),
    )


def build_simple_inbatch_negative_indices(gw_indices: torch.Tensor) -> torch.Tensor:
    """Deterministic roll-and-skip negative pairing for ablation comparison."""
    gw_indices = gw_indices.to(torch.long)
    batch_size = int(gw_indices.numel())
    device = gw_indices.device
    base = torch.arange(batch_size, device=device, dtype=torch.long)
    if batch_size <= 1:
        return base

    result = torch.roll(base, shifts=1, dims=0)
    unresolved = gw_indices[result] == gw_indices
    shift = 2
    while unresolved.any() and shift < batch_size:
        candidate = torch.roll(base, shifts=shift, dims=0)
        valid = gw_indices[candidate] != gw_indices
        update_mask = unresolved & valid
        result = torch.where(update_mask, candidate, result)
        unresolved = unresolved & (~valid)
        shift += 1

    if unresolved.any():
        unresolved_rows = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
        for row in unresolved_rows.tolist():
            candidates = torch.nonzero(gw_indices != gw_indices[row], as_tuple=False).squeeze(-1)
            if candidates.numel() > 0:
                result[row] = candidates[0]
            else:
                result[row] = row
    return result


def sample_simple_inbatch_hard_negatives_with_time(
    sim_g2o,
    gw_indices,
    batch_event_time_mjd,
    window_days,
    min_candidates,
    semi_hard,
    semi_hard_margin,
    fallback_mode,
    active_rows=None,
):
    del sim_g2o, batch_event_time_mjd, window_days, min_candidates, semi_hard, semi_hard_margin, fallback_mode
    full_idx = build_simple_inbatch_negative_indices(gw_indices)
    if active_rows is not None:
        active_rows = active_rows.to(device=gw_indices.device, dtype=torch.long)
        return full_idx[active_rows], [0], 0
    return full_idx, [0], 0


@contextmanager
def temporary_simple_hard_negative_sampling(model):
    import ALBEF_train

    patch_targets = [model]
    core_model = getattr(model, "_orig_mod", None)
    if core_model is not None:
        patch_targets.append(core_model)

    original_time_sampler = ALBEF_train.sample_inbatch_hard_negatives_with_time
    original_methods = []

    def _simple_sampler(sim_g2o, gw_indices=None, margin=0.2):
        del sim_g2o, margin
        if gw_indices is None:
            raise ValueError("simple hard-negative sampler requires gw_indices")
        return build_simple_inbatch_negative_indices(gw_indices)

    ALBEF_train.sample_inbatch_hard_negatives_with_time = sample_simple_inbatch_hard_negatives_with_time
    for target in patch_targets:
        original_methods.append((target, getattr(target, "sample_semi_hard_negatives", None), getattr(target, "sample_hard_negatives", None)))
        if hasattr(target, "sample_semi_hard_negatives"):
            target.sample_semi_hard_negatives = _simple_sampler
        if hasattr(target, "sample_hard_negatives"):
            target.sample_hard_negatives = _simple_sampler
    try:
        yield
    finally:
        ALBEF_train.sample_inbatch_hard_negatives_with_time = original_time_sampler
        for target, original_semi, original_hard in original_methods:
            if original_semi is not None:
                target.sample_semi_hard_negatives = original_semi
            if original_hard is not None:
                target.sample_hard_negatives = original_hard


def evaluate_multimodal_classification(
    model,
    runtime_model_args: Dict[str, Any],
    saved_args: Dict[str, Any],
    *,
    seed: int,
    test_data_path: str,
    neg_data_path: Optional[str],
    neg_group: str,
    batch_size: int,
    test_steps: Optional[int],
    num_workers: int,
    n_neg_samples: int,
    comparison_window: Tuple[float, float],
    nonkn_cls_base_field: str,
    report_dt_bins: bool,
    dt_bin_edges: Optional[List[float]],
    report_dt_macro: bool,
    neg_optical_data,
    neg_gw_indices,
    gw_source_types,
    gw_event_time_mjd_table,
    shared_cfg: Dict[str, Any],
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    device: torch.device,
):
    """Run triplet classification for a multimodal model."""
    hardneg_windows_days = parse_day_windows(str(saved_args.get("hardneg_time_window_days", "30,60,120")))
    hardneg_min_candidates = int(saved_args.get("hardneg_min_candidates", 4))
    hardneg_semi_hard = _as_bool(saved_args.get("semi_hard", True), default=True)
    hardneg_semi_hard_margin = float(saved_args.get("semi_hard_margin", 0.2))
    hardneg_fallback_mode = str(saved_args.get("hardneg_fallback_mode", "inbatch_semihard"))
    hard_negative_strategy = str(shared_cfg.get("hard_negative_strategy", "simple")).strip().lower()

    try_num_workers = int(num_workers)
    while True:
        try:
            _seed_all(seed)
            loader = _build_test_loader(
                test_data_path,
                neg_data_path,
                neg_group,
                comparison_window[0],
                comparison_window[1],
                batch_size=batch_size,
                test_steps=test_steps,
                num_workers=try_num_workers,
                target_samples=n_neg_samples,
                nonkn_cls_base_field=nonkn_cls_base_field,
                return_zero_time_mjd=False,
            )
            neg_offset_policy = build_neg_offset_policy(shared_cfg, seed)
            if hard_negative_strategy == "simple":
                with temporary_simple_hard_negative_sampling(model):
                    triplet_logits = extract_triplet_logits(
                        model,
                        loader,
                        device,
                        runtime_model_args,
                        neg_optical_data,
                        neg_gw_indices,
                        gw_source_types=gw_source_types,
                        shuffle_gw=False,
                        shuffle_seed=int(seed),
                        amp_dtype=amp_dtype,
                        amp_enabled=amp_enabled,
                        neg_offset_policy=neg_offset_policy,
                        gw_event_time_mjd_table=gw_event_time_mjd_table,
                        hardneg_windows_days=hardneg_windows_days,
                        hardneg_min_candidates=hardneg_min_candidates,
                        hardneg_semi_hard=hardneg_semi_hard,
                        hardneg_semi_hard_margin=hardneg_semi_hard_margin,
                        hardneg_fallback_mode=hardneg_fallback_mode,
                    )
            else:
                triplet_logits = extract_triplet_logits(
                    model,
                    loader,
                    device,
                    runtime_model_args,
                    neg_optical_data,
                    neg_gw_indices,
                    gw_source_types=gw_source_types,
                    shuffle_gw=False,
                    shuffle_seed=int(seed),
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                    neg_offset_policy=neg_offset_policy,
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    hardneg_windows_days=hardneg_windows_days,
                    hardneg_min_candidates=hardneg_min_candidates,
                    hardneg_semi_hard=hardneg_semi_hard,
                    hardneg_semi_hard_margin=hardneg_semi_hard_margin,
                    hardneg_fallback_mode=hardneg_fallback_mode,
                )
            cls_metrics = evaluate_classification_triplet(
                triplet_logits,
                report_dt_bins=bool(report_dt_bins),
                dt_bin_edges=dt_bin_edges,
                report_dt_macro=bool(report_dt_macro),
            )
            cls_by_source = evaluate_classification_triplet_by_source(triplet_logits)
            return cls_metrics, cls_by_source
        except PermissionError:
            if try_num_workers <= 0:
                raise
            print("  WARNING: DataLoader multiprocessing failed. Retrying triplet classification with num_workers=0.")
            try_num_workers = 0


def normalize_shared_config(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    cfg_dir = cfg_path.parent
    comparison_window = (
        float(cfg.get("comparison_window_start", DEFAULT_COMPARISON_WINDOW[0])),
        float(cfg.get("comparison_window_end", DEFAULT_COMPARISON_WINDOW[1])),
    )
    requested_neg_data_path = _resolve_path(cfg_dir, cfg.get("neg_data_path"))
    requested_neg_group = str(cfg.get("neg_group", DEFAULT_TUTORIAL_NEG_GROUP))
    tutorial_neg_data_path = _resolve_path(cfg_dir, cfg.get("tutorial_neg_data_path")) or DEFAULT_TUTORIAL_NEG_DATA_PATH
    tutorial_neg_group = str(cfg.get("tutorial_neg_group", DEFAULT_TUTORIAL_NEG_GROUP))
    force_tutorial_distractors = _as_bool(cfg.get("force_tutorial_distractors", True), default=True)
    normalized = {
        "input_config": str(cfg_path),
        "test_data_path": _resolve_path(cfg_dir, cfg.get("test_data_path")),
        "neg_data_path": tutorial_neg_data_path if force_tutorial_distractors else requested_neg_data_path,
        "neg_group": tutorial_neg_group if force_tutorial_distractors else requested_neg_group,
        "requested_neg_data_path": requested_neg_data_path,
        "requested_neg_group": requested_neg_group,
        "tutorial_neg_data_path": tutorial_neg_data_path,
        "tutorial_neg_group": tutorial_neg_group,
        "force_tutorial_distractors": bool(force_tutorial_distractors),
        "output_dir": _resolve_path(cfg_dir, cfg.get("output_dir")) or str((MODEL_DIR / "eval_results" / "ablation_comparison").resolve()),
        "device": str(cfg.get("device", "cuda")),
        "seed": int(cfg.get("seed", 42)),
        "batch_size": int(cfg.get("batch_size", 512)),
        "test_steps": None if cfg.get("test_steps") is None else int(cfg.get("test_steps")),
        "num_workers": int(cfg.get("num_workers", 2)),
        "gallery_sizes": _parse_gallery_sizes(cfg.get("gallery_sizes", "10,100,500,1000,2000,5000")),
        "gallery_trials": int(cfg.get("gallery_trials", 1)),
        "gallery_candidate_mode": str(cfg.get("gallery_candidate_mode", "time_sky_hard")).strip().lower(),
        "gallery_candidate_time_window_days": float(cfg.get("gallery_candidate_time_window_days", 50.0)),
        "gallery_candidate_credible_level_max": float(cfg.get("gallery_candidate_credible_level_max", 0.9)),
        "gallery_include_undersized": _as_bool(cfg.get("gallery_include_undersized", True), default=True),
        "n_neg_samples": _parse_n_neg_samples(cfg.get("n_neg_samples", -1), default=-1),
        "amp_dtype": str(cfg.get("amp_dtype", "auto")),
        "no_latex": _as_bool(cfg.get("no_latex", False)),
        "comparison_window": [float(comparison_window[0]), float(comparison_window[1])],
        "comparison_window_start": float(comparison_window[0]),
        "comparison_window_end": float(comparison_window[1]),
        "neg_time_offset_enable": _as_bool(cfg.get("neg_time_offset_enable", False)),
        "neg_offset_dist_npz": _resolve_path(cfg_dir, cfg.get("neg_offset_dist_npz")),
        "neg_offset_dist_key": str(cfg.get("neg_offset_dist_key", "delta_days_combined")),
        "neg_offset_eval_mode": str(cfg.get("neg_offset_eval_mode", "quantile_ensemble")),
        "neg_offset_eval_quantiles": str(cfg.get("neg_offset_eval_quantiles", "0.1,0.3,0.5,0.7,0.9")),
        "neg_offset_scale_days_divisor": float(cfg.get("neg_offset_scale_days_divisor", 100.0)),
        "neg_offset_bank_size": int(cfg.get("neg_offset_bank_size", 1000000)),
        "nonkn_cls_base_field": str(cfg.get("nonkn_cls_base_field", "zero_time_mjd_cls_base")),
        "report_dt_bins": _as_bool(cfg.get("report_dt_bins", False)),
        "dt_bin_edges": str(cfg.get("dt_bin_edges", DEFAULT_DT_BIN_EDGES)),
        "report_dt_macro": _as_bool(cfg.get("report_dt_macro", False)),
        "hard_negative_strategy": str(cfg.get("hard_negative_strategy", "simple")),
    }
    if normalized["test_data_path"] is None:
        raise ValueError("Comparison config is missing test_data_path")
    return normalized


def main():
    parser = argparse.ArgumentParser(description="Unified retrieval and triplet ablation comparison.")
    parser.add_argument("--config", type=str, required=True, help="Path to comparison JSON config")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    raw_cfg = _load_json(config_path)
    cfg = normalize_shared_config(raw_cfg, config_path)
    model_specs = _build_model_specs(raw_cfg, config_path.parent)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    amp_dtype, amp_enabled = _resolve_eval_amp(cfg["amp_dtype"], device)
    comparison_window = (float(cfg["comparison_window_start"]), float(cfg["comparison_window_end"]))
    gallery_sizes = list(cfg["gallery_sizes"])
    n_trials = int(cfg["gallery_trials"])
    seed = int(cfg["seed"])

    print("=" * 60)
    print("Unified Ablation Comparison")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Comparison window: {comparison_window}")
    print(f"Gallery sizes: {gallery_sizes}")
    print(f"Models: {[spec['name'] for spec in model_specs]}")
    print(f"Gallery distractor pool: {cfg['neg_data_path']} [{cfg['neg_group']}]")
    if cfg["force_tutorial_distractors"]:
        print("Tutorial distractors forced on for retrieval/classification negatives.")

    _seed_all(seed)
    gw_source_types = load_gw_source_types(cfg["test_data_path"])
    gw_event_time_mjd_table = load_gw_event_time_mjd_table(cfg["test_data_path"], device, required=False)
    neg_gw_indices = load_negative_gw_indices(cfg["test_data_path"])

    neg_target_samples = _resolve_target_samples(cfg["n_neg_samples"])

    neg_optical_data = None
    if cfg["neg_data_path"] and os.path.exists(cfg["neg_data_path"]):
        neg_optical_data = load_negative_optical_samples(
            cfg["neg_data_path"],
            cfg["neg_group"],
            n_samples=neg_target_samples,
            seed=seed,
            require_zero_time_mjd_base=False,
            require_zero_time_mjd_cls_base=False,
            nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
            runtime_input_window_start=comparison_window[0],
            runtime_input_window_end=comparison_window[1],
        )
    if neg_optical_data is None:
        raise FileNotFoundError(
            f"Tutorial distractor dataset is required but unavailable: {cfg['neg_data_path']}"
        )

    print(f"\n{'=' * 60}")
    print("Phase 1: Load and cache shared retrieval batches")
    print("=" * 60)
    cached_batches, used_num_workers = build_cached_batches(
        test_data_path=cfg["test_data_path"],
        neg_data_path=None,
        neg_group=cfg["neg_group"],
        comparison_window=comparison_window,
        batch_size=cfg["batch_size"],
        test_steps=cfg["test_steps"],
        num_workers=cfg["num_workers"],
        target_samples=neg_target_samples,
        nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
    )
    cfg["used_num_workers"] = int(used_num_workers)

    gw_indices_parts = []
    for batch_data in cached_batches:
        gw_indices_parts.append(batch_data[7].long().cpu())
    shared_gw_indices = torch.cat(gw_indices_parts)
    unique_sampled_gw = sorted(torch.unique(shared_gw_indices).cpu().tolist())
    sampled_positive_counts = np.asarray(
        [int((shared_gw_indices == int(gw_id)).sum().item()) for gw_id in unique_sampled_gw],
        dtype=np.int64,
    )
    sampled_positive_summary = {
        "n_unique_gw": int(len(unique_sampled_gw)),
        "n_sampled_positives": int(shared_gw_indices.numel()),
        "sampled_positives_per_gw": {
            "min": int(sampled_positive_counts.min()) if sampled_positive_counts.size else 0,
            "median": float(np.median(sampled_positive_counts)) if sampled_positive_counts.size else 0.0,
            "mean": float(np.mean(sampled_positive_counts)) if sampled_positive_counts.size else 0.0,
            "max": int(sampled_positive_counts.max()) if sampled_positive_counts.size else 0,
        },
    }
    gw_positive_indices, all_test_gw_ids = load_test_positive_index_map(cfg["test_data_path"])
    full_test_positive_counts = np.asarray(
        [int(np.asarray(gw_positive_indices[int(gw_id)]).size) for gw_id in all_test_gw_ids],
        dtype=np.int64,
    )
    gallery_positive_summary = {
        "n_unique_gw": int(len(all_test_gw_ids)),
        "n_total_positives": int(sum(int(count) for count in full_test_positive_counts.tolist())),
        "positives_per_gw": {
            "min": int(full_test_positive_counts.min()) if full_test_positive_counts.size else 0,
            "median": float(np.median(full_test_positive_counts)) if full_test_positive_counts.size else 0.0,
            "mean": float(np.mean(full_test_positive_counts)) if full_test_positive_counts.size else 0.0,
            "max": int(full_test_positive_counts.max()) if full_test_positive_counts.size else 0,
        },
    }
    gw_source_map = build_gw_source_map(gw_source_types, all_test_gw_ids)

    print(f"\n{'=' * 60}")
    print("Phase 2: Pre-generate shared galleries")
    print("=" * 60)
    gallery_candidate_mode = str(cfg["gallery_candidate_mode"]).strip().lower()
    gallery_gw_skymaps: Dict[int, torch.Tensor] = {}
    if gallery_candidate_mode == "time_sky_hard":
        candidate_sequences, gallery_gw_skymaps, _gallery_gw_times = build_time_sky_candidate_sequences(
            test_data_path=cfg["test_data_path"],
            unique_gw_ids=all_test_gw_ids,
            neg_optical_data=neg_optical_data,
            n_trials=n_trials,
            seed=seed,
            time_window_days=cfg["gallery_candidate_time_window_days"],
            credible_level_max=cfg["gallery_candidate_credible_level_max"],
            zero_time_field=cfg["nonkn_cls_base_field"],
        )
        galleries, unique_gw = build_prefixed_gallery_specs(
            gw_positive_indices=gw_positive_indices,
            candidate_sequences=candidate_sequences,
            gallery_sizes=gallery_sizes,
            n_trials=n_trials,
            seed=seed,
            include_undersized=bool(cfg["gallery_include_undersized"]),
        )
        print(
            "  Built hard-filtered galleries with "
            f"time_window=±{cfg['gallery_candidate_time_window_days']:.1f} days, "
            f"credible<= {cfg['gallery_candidate_credible_level_max']:.3f}, "
            f"include_undersized={bool(cfg['gallery_include_undersized'])}"
        )
    elif gallery_candidate_mode == "tutorial_random":
        galleries, unique_gw = precompute_tutorial_galleries(
            gw_positive_indices,
            int(neg_optical_data["times"].shape[0]),
            gallery_sizes,
            n_trials,
            seed,
        )
    else:
        raise ValueError(
            f"Unsupported gallery_candidate_mode='{gallery_candidate_mode}'. "
            "Expected one of {'time_sky_hard', 'tutorial_random'}."
        )
    if not galleries:
        raise ValueError("Gallery construction produced no instances.")
    selected_positive_indices = np.unique(
        np.asarray([int(spec["positive_index"]) for spec in galleries.values()], dtype=np.int64)
    )
    positive_query_bank, positive_index_remap = load_selected_positive_bank(
        cfg["test_data_path"],
        selected_positive_indices,
    )
    galleries = remap_gallery_positive_indices(galleries, positive_index_remap)
    print(
        "  Selected "
        f"{int(positive_query_bank['opt_t_raw'].shape[0])} positive optical queries "
        "from the full test set"
    )
    print(f"  Built {len(galleries)} gallery instances for {len(unique_gw)} GW events")

    dt_bin_edges = parse_dt_bin_edges(cfg["dt_bin_edges"]) if cfg["report_dt_bins"] else None
    model_results: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    retrieval_rows: OrderedDict[str, Dict[str, float]] = OrderedDict()
    curve_rows: List[Dict[str, Any]] = []
    model_names: List[str] = []

    for idx, model_spec in enumerate(model_specs, start=1):
        name = model_spec["name"]
        model_type = model_spec["type"]
        model_names.append(name)
        print(f"\n{'=' * 60}")
        print(f"Phase 3.{idx}: {name} ({model_type})")
        print("=" * 60)
        print(f"  Resolved checkpoint: {model_spec['resolved_checkpoint'] or 'N/A'}")
        if model_spec.get("resolved_config"):
            print(f"  Config: {model_spec['resolved_config']}")

        if model_type == "optical":
            model, ckpt_args = load_optical_model(model_spec["resolved_checkpoint"], device)
            optical_n_ref = int(ckpt_args.get("n_ref", 64))
            original_window = [
                float(ckpt_args.get("ref_start", comparison_window[0])),
                float(ckpt_args.get("ref_end", comparison_window[1])),
            ]
            ranks = score_all_galleries_optical(
                model,
                positive_query_bank,
                neg_optical_data,
                galleries,
                device=device,
                n_ref=optical_n_ref,
                ref_start=comparison_window[0],
                ref_end=comparison_window[1],
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            outcomes = enrich_gallery_outcomes(ranks, galleries)
            retrieval_metrics, retrieval_by_source, coverage_stats = aggregate_gallery_outcomes(
                outcomes=outcomes,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                unique_gw=unique_gw,
                gw_source_map=gw_source_map if gw_source_map else None,
            )
            model_results[name] = {
                "type": model_type,
                "scoring": "optical",
                "resolved_checkpoint": model_spec["resolved_checkpoint"],
                "resolved_config": model_spec.get("resolved_config"),
                "original_model_window": original_window,
                "comparison_window": list(comparison_window),
                "retrieval": retrieval_metrics,
                "retrieval_by_source": retrieval_by_source,
                "coverage": {key: float(val.get("coverage", 0.0)) for key, val in coverage_stats.items()},
                "effective_gallery_size_stats": coverage_stats,
                "classification_supported": False,
                "classification": None,
                "classification_by_source": {},
            }
            del model
        elif model_type == "multimodal":
            model, runtime_model_args, saved_args = load_multimodal_bundle(
                model_spec["resolved_checkpoint"],
                model_spec["resolved_config"],
                device,
                test_data_path=cfg["test_data_path"],
                neg_data_path=cfg["neg_data_path"],
                comparison_window=comparison_window,
                nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
            )
            embeddings = extract_optical_candidate_embeddings(
                model,
                positive_query_bank,
                device,
                n_ref=int(runtime_model_args.get("n_ref", 64)),
                ref_start=float(runtime_model_args["ref_start"]),
                ref_end=float(runtime_model_args["ref_end"]),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
                desc="  Extracting positive query embeddings",
            )
            negative_embeddings = extract_negative_gallery_embeddings(
                model,
                neg_optical_data,
                device,
                n_ref=int(runtime_model_args.get("n_ref", 64)),
                ref_start=float(runtime_model_args["ref_start"]),
                ref_end=float(runtime_model_args["ref_end"]),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            scoring_mode = _resolve_multimodal_scoring(model_spec, saved_args)

            if scoring_mode == "contrastive":
                ranks = score_all_galleries_contrastive(
                    model,
                    embeddings,
                    negative_embeddings,
                    galleries,
                    unique_gw,
                    test_data_path=cfg["test_data_path"],
                    device=device,
                    model_args=runtime_model_args,
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                )
            else:
                ranks = score_all_galleries_multimodal(
                    model,
                    embeddings,
                    negative_embeddings,
                    galleries,
                    unique_gw,
                    test_data_path=cfg["test_data_path"],
                    device=device,
                    model_args=runtime_model_args,
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                )
            outcomes = enrich_gallery_outcomes(ranks, galleries)
            retrieval_metrics, retrieval_by_source, coverage_stats = aggregate_gallery_outcomes(
                outcomes=outcomes,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                unique_gw=unique_gw,
                gw_source_map=gw_source_map if gw_source_map else None,
            )

            if scoring_mode == "contrastive":
                print("  Skipping triplet classification (fusion classifier untrained for contrastive-only model)")
                classification_metrics = None
                classification_by_source = {}
                classification_supported = False
            else:
                print("  Computing triplet classification metrics...")
                classification_metrics, classification_by_source = evaluate_multimodal_classification(
                    model,
                    runtime_model_args,
                    saved_args,
                    seed=seed,
                    test_data_path=cfg["test_data_path"],
                    neg_data_path=cfg["neg_data_path"],
                    neg_group=cfg["neg_group"],
                    batch_size=cfg["batch_size"],
                    test_steps=cfg["test_steps"],
                    num_workers=cfg["num_workers"],
                    n_neg_samples=neg_target_samples,
                    comparison_window=comparison_window,
                    nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
                    report_dt_bins=cfg["report_dt_bins"],
                    dt_bin_edges=dt_bin_edges,
                    report_dt_macro=cfg["report_dt_macro"],
                    neg_optical_data=neg_optical_data,
                    neg_gw_indices=neg_gw_indices,
                    gw_source_types=gw_source_types,
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    shared_cfg=cfg,
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                    device=device,
                )
                classification_supported = True
            model_results[name] = {
                "type": model_type,
                "scoring": scoring_mode,
                "resolved_checkpoint": model_spec["resolved_checkpoint"],
                "resolved_config": model_spec.get("resolved_config"),
                "original_model_window": [
                    float(runtime_model_args.get("original_ref_start", comparison_window[0])),
                    float(runtime_model_args.get("original_ref_end", comparison_window[1])),
                ],
                "comparison_window": list(comparison_window),
                "retrieval": retrieval_metrics,
                "retrieval_by_source": retrieval_by_source,
                "coverage": {key: float(val.get("coverage", 0.0)) for key, val in coverage_stats.items()},
                "effective_gallery_size_stats": coverage_stats,
                "classification_supported": classification_supported,
                "classification": classification_metrics,
                "classification_by_source": classification_by_source,
                "effective_input_window_metadata": runtime_model_args.get("effective_input_window_metadata"),
            }
            del negative_embeddings, embeddings, model
        elif model_type == "skymap":
            if gallery_candidate_mode != "time_sky_hard":
                raise ValueError("skymap-only model requires gallery_candidate_mode='time_sky_hard'")
            outcomes = score_all_galleries_skymap(
                positive_bank={"opt_coords": positive_query_bank["opt_coords"]},
                galleries=galleries,
                gw_skymaps=gallery_gw_skymaps,
            )
            retrieval_metrics, retrieval_by_source, coverage_stats = aggregate_gallery_outcomes(
                outcomes=outcomes,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                unique_gw=unique_gw,
                gw_source_map=gw_source_map if gw_source_map else None,
            )
            model_results[name] = {
                "type": model_type,
                "scoring": "skymap",
                "resolved_checkpoint": None,
                "resolved_config": None,
                "original_model_window": None,
                "comparison_window": list(comparison_window),
                "retrieval": retrieval_metrics,
                "retrieval_by_source": retrieval_by_source,
                "coverage": {key: float(val.get("coverage", 0.0)) for key, val in coverage_stats.items()},
                "effective_gallery_size_stats": coverage_stats,
                "classification_supported": False,
                "classification": None,
                "classification_by_source": {},
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        retrieval_rows[name] = model_results[name]["retrieval"]
        model_curve_rows = build_curve_rows(
            method_name=name,
            gallery_sizes=gallery_sizes,
            retrieval_metrics=model_results[name]["retrieval"],
            coverage_stats=model_results[name]["effective_gallery_size_stats"],
        )
        model_results[name]["curve_rows"] = model_curve_rows
        curve_rows.extend(model_curve_rows)

        for gallery_size in gallery_sizes:
            coverage_info = model_results[name]["effective_gallery_size_stats"].get(f"gallery_{gallery_size}", {})
            r1 = model_results[name]["retrieval"].get(f"gallery_{gallery_size}_recall_at_1", 0.0)
            r5 = model_results[name]["retrieval"].get(f"gallery_{gallery_size}_recall_at_5", 0.0)
            r10 = model_results[name]["retrieval"].get(f"gallery_{gallery_size}_recall_at_10", 0.0)
            mrr = model_results[name]["retrieval"].get(f"gallery_{gallery_size}_mrr", 0.0)
            print(
                f"  gallery={gallery_size}  R@1={r1:.4f}  R@5={r5:.4f}  "
                f"R@10={r10:.4f}  MRR={mrr:.4f}  "
                f"coverage={coverage_info.get('coverage', 0.0):.3f}  "
                f"effN={coverage_info.get('effective_gallery_size_mean', 0.0):.1f}"
            )
        if model_results[name]["classification_supported"] and model_results[name]["classification"]:
            cls = model_results[name]["classification"]
            print(
                "  classification "
                f"AUROC={cls.get('auroc', 0.0):.4f} "
                f"AUPRC={cls.get('auprc', 0.0):.4f} "
                f"F1={cls.get('f1_optimal', 0.0):.4f}"
            )
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'=' * 60}")
    print("Results: Overall Retrieval")
    print("=" * 60)
    print_unified_table(retrieval_rows, gallery_sizes, model_names)
    if not cfg["no_latex"]:
        print_latex_table(retrieval_rows, gallery_sizes, model_names)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_retrieval_curves(curve_rows, output_dir)
    plot_retrieval_coverage(curve_rows, output_dir)
    output = {
        "table": {
            "gallery_sizes": gallery_sizes,
            "metric_labels": TABLE_METRIC_LABELS,
            "rows": build_table_rows(retrieval_rows, gallery_sizes, model_names),
        },
        "curve_rows": curve_rows,
        "models": dict(model_results),
        "sampled_positive_summary": sampled_positive_summary,
        "gallery_positive_summary": gallery_positive_summary,
        "config": {
            "input_config": cfg["input_config"],
            "test_data_path": cfg["test_data_path"],
            "neg_data_path": cfg["neg_data_path"],
            "neg_group": cfg["neg_group"],
            "requested_neg_data_path": cfg["requested_neg_data_path"],
            "requested_neg_group": cfg["requested_neg_group"],
            "tutorial_neg_data_path": cfg["tutorial_neg_data_path"],
            "tutorial_neg_group": cfg["tutorial_neg_group"],
            "force_tutorial_distractors": cfg["force_tutorial_distractors"],
            "output_dir": str(output_dir),
            "device": str(device),
            "amp_dtype": cfg["amp_dtype"],
            "seed": seed,
            "batch_size": cfg["batch_size"],
            "test_steps": cfg["test_steps"],
            "num_workers": cfg["num_workers"],
            "used_num_workers": cfg.get("used_num_workers", cfg["num_workers"]),
            "gallery_sizes": gallery_sizes,
            "gallery_trials": n_trials,
            "gallery_candidate_mode": cfg["gallery_candidate_mode"],
            "gallery_candidate_time_window_days": cfg["gallery_candidate_time_window_days"],
            "gallery_candidate_credible_level_max": cfg["gallery_candidate_credible_level_max"],
            "gallery_include_undersized": cfg["gallery_include_undersized"],
            "comparison_window": list(comparison_window),
            "n_neg_samples": cfg["n_neg_samples"],
            "neg_time_offset_enable": cfg["neg_time_offset_enable"],
            "neg_offset_dist_npz": cfg["neg_offset_dist_npz"],
            "neg_offset_dist_key": cfg["neg_offset_dist_key"],
            "neg_offset_eval_mode": cfg["neg_offset_eval_mode"],
            "neg_offset_eval_quantiles": cfg["neg_offset_eval_quantiles"],
            "neg_offset_scale_days_divisor": cfg["neg_offset_scale_days_divisor"],
            "nonkn_cls_base_field": cfg["nonkn_cls_base_field"],
            "report_dt_bins": cfg["report_dt_bins"],
            "dt_bin_edges": dt_bin_edges if dt_bin_edges is not None else None,
            "report_dt_macro": cfg["report_dt_macro"],
            "hard_negative_strategy": cfg["hard_negative_strategy"],
            "n_unique_gw": gallery_positive_summary["n_unique_gw"],
            "n_gallery_positives": gallery_positive_summary["n_total_positives"],
            "gallery_positives_per_gw": gallery_positive_summary["positives_per_gw"],
            "n_sampled_positives": sampled_positive_summary["n_sampled_positives"],
            "n_sampled_positive_gw": sampled_positive_summary["n_unique_gw"],
            "sampled_positives_per_gw": sampled_positive_summary["sampled_positives_per_gw"],
            "models": [
                {
                    "name": spec["name"],
                    "type": spec["type"],
                    "scoring": spec.get("scoring"),
                    "checkpoint": spec["checkpoint"],
                    "config": spec.get("config"),
                    "resolved_checkpoint": spec["resolved_checkpoint"],
                    "resolved_config": spec.get("resolved_config"),
                }
                for spec in model_specs
            ],
        },
    }
    output_path = output_dir / "ablation_comparison.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(output), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
