from pathlib import Path
import sys

MODEL_DIR = Path(__file__).resolve().parents[2]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import (
    create_training_dataloader,
    create_train_val_dataloaders,
    create_supcon_dataloaders,
    build_effective_input_window_metadata,
)
from model import MAGIKSModel, normalize_fusion_mode, migrate_time_embed_state_dict
from validation_gallery import (
    ValidationGalleryContext,
    parse_validation_gallery_sizes,
)
from mixed_retrieval import allocate_mixed_negative_counts
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import h5py
import numpy as np
import os
import datetime
import argparse
import copy
import math
import gc
import json
import time
import warnings
import random
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")


BASH_ONLY_CONFIG_KEYS = {"stage_to_jobfs"}


def _load_json_config_file(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"JSON config must contain an object: {path}")
    return {
        k: v for k, v in cfg.items() if k not in BASH_ONLY_CONFIG_KEYS and v is not None
    }


def load_merged_json_config(
    *,
    default_json_config: Optional[str] = None,
    json_config: Optional[str] = None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged.update(_load_json_config_file(default_json_config))
    merged.update(_load_json_config_file(json_config))
    return merged


def apply_json_config_defaults(
    parser: argparse.ArgumentParser,
    *,
    default_json_config: Optional[str] = None,
    json_config: Optional[str] = None,
) -> Dict[str, Any]:
    merged = load_merged_json_config(
        default_json_config=default_json_config,
        json_config=json_config,
    )
    if merged:
        parser.set_defaults(**merged)
    return merged


def _close_loader_dataset_handles(loader, *, cache_in_memory: bool, label: str) -> None:
    if loader is None or cache_in_memory:
        return
    dataset = getattr(loader, "dataset", None)
    close_fn = getattr(dataset, "close", None)
    if not callable(close_fn):
        return
    try:
        closed = int(close_fn())
    except Exception as exc:
        print(f"[WARN] {label}: failed to close lazy dataset handles: {exc}")
        return
    if closed > 0:
        print(f"{label}: closed {closed} lazy HDF5 handle(s) after evaluation.")


DEFAULT_MTAN_LUPT_M5 = np.asarray(
    [23.9, 25.0, 24.7, 24.0, 23.3, 22.1], dtype=np.float64
)


def build_ref_time(batch_size, n_ref, ref_start, ref_end, device, dtype):
    ref = torch.linspace(ref_start, ref_end, n_ref, dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)


def parse_quantiles(text: str) -> List[float]:
    vals: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        q = float(part)
        if q < 0.0 or q > 1.0:
            raise ValueError(f"Invalid quantile {q}; expected in [0, 1].")
        vals.append(q)
    if not vals:
        raise ValueError("quantile string produced empty list.")
    return vals


def parse_day_windows(text: str) -> List[float]:
    vals: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if v <= 0:
            raise ValueError(
                f"Invalid hard-negative time window value: {v}. Must be > 0."
            )
        vals.append(v)
    if not vals:
        raise ValueError("hardneg_time_window_days produced empty list.")
    vals = sorted(set(vals))
    return vals


def parse_lupt_m5_mag_text(text: str) -> np.ndarray:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError(
            "mtan_lupt_m5_mag must provide exactly 6 comma-separated values in order u,g,r,i,z,Y."
        )
    try:
        vals = np.asarray([float(p) for p in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("mtan_lupt_m5_mag contains non-numeric values.") from exc
    if not np.all(np.isfinite(vals)):
        raise ValueError("mtan_lupt_m5_mag must contain finite values.")
    return vals


def _jsonify_metadata_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _read_optical_h5_window_metadata(path: Optional[str]) -> Dict[str, object]:
    out: Dict[str, object] = {"path": None if path is None else str(path)}
    if path is None:
        return out
    if not os.path.exists(path):
        out["exists"] = False
        return out
    out["exists"] = True
    with h5py.File(path, "r") as f:
        for key in (
            "observation_window_mode",
            "pre_first_detection_points_kept",
            "post_last_detection_points_kept",
            "enforce_time_window",
            "time_window_start",
            "time_window_end",
            "fixed_offset_days",
        ):
            if key in f.attrs:
                out[key] = _jsonify_metadata_value(f.attrs[key])
    return out


def load_gw_event_time_mjd_table(h5_path, device, required=False):
    ds_path = "events/gw_data/event_time_mjd"
    if h5_path is None or not os.path.exists(h5_path):
        if required:
            raise FileNotFoundError(
                f"data_path not found for event_time_mjd: {h5_path}"
            )
        print(f"WARNING: data_path not found for event_time_mjd: {h5_path}")
        return None

    with h5py.File(h5_path, "r") as f:
        n_gw = int(f["events/gw_data/scalars"].shape[0])
        if ds_path not in f:
            if required:
                raise KeyError(f"Required field missing: {ds_path} in {h5_path}")
            print(f"WARNING: '{ds_path}' missing in {h5_path}.")
            return None
        arr = np.asarray(f[ds_path][:], dtype=np.float32).reshape(-1)

    if arr.shape[0] != n_gw:
        if required:
            raise ValueError(
                f"event_time_mjd length mismatch in {h5_path}: got {arr.shape[0]}, expected {n_gw}"
            )
        print(
            f"WARNING: event_time_mjd length mismatch in {h5_path}: got {arr.shape[0]}, expected {n_gw}"
        )
        return None
    return torch.from_numpy(arr).to(device=device)


def load_gw_source_type_table(h5_path, device, required=False):
    """Load GW source labels as integer codes for equality-based sampling."""
    ds_path = "events/gw_data/source_type"
    if h5_path is None or not os.path.exists(h5_path):
        if required:
            raise FileNotFoundError(f"data_path not found for source_type: {h5_path}")
        print(f"WARNING: data_path not found for source_type: {h5_path}")
        return None

    with h5py.File(h5_path, "r") as f:
        n_gw = int(f["events/gw_data/scalars"].shape[0])
        if ds_path not in f:
            if required:
                raise KeyError(f"Required field missing: {ds_path} in {h5_path}")
            print(f"WARNING: '{ds_path}' missing in {h5_path}.")
            return None
        raw_source_types = np.asarray(f[ds_path][:]).reshape(-1)

    if raw_source_types.shape[0] != n_gw:
        if required:
            raise ValueError(
                f"source_type length mismatch in {h5_path}: "
                f"got {raw_source_types.shape[0]}, expected {n_gw}"
            )
        print(
            f"WARNING: source_type length mismatch in {h5_path}: "
            f"got {raw_source_types.shape[0]}, expected {n_gw}"
        )
        return None

    source_names = []
    for value in raw_source_types:
        if isinstance(value, (bytes, np.bytes_)):
            value = value.decode("utf-8", errors="strict")
        source_name = str(value).strip().lower()
        if not source_name:
            raise ValueError(f"Empty source_type found in {h5_path}.")
        source_names.append(source_name)

    labels, codes = np.unique(np.asarray(source_names, dtype=str), return_inverse=True)
    counts = np.bincount(codes, minlength=len(labels))
    summary = ", ".join(
        f"{label}={int(count)}"
        for label, count in zip(labels.tolist(), counts.tolist())
    )
    print(f"Loaded GW source types for mismatched negatives: {summary}")
    return torch.from_numpy(codes.astype(np.int64, copy=False)).to(device=device)


def load_gw_neg_type_table(h5_path, device, required=False):
    """Load per-event negative-GW type codes."""
    ds_path = "events/gw_data/neg_type"
    with h5py.File(h5_path, "r") as f:
        n_gw = int(f["events/gw_data/scalars"].shape[0])
        if ds_path not in f:
            if required:
                raise KeyError(f"Required field missing: {ds_path} in {h5_path}")
            return None
        values = np.asarray(f[ds_path][:], dtype=np.int64).reshape(-1)
    if values.shape[0] != n_gw:
        raise ValueError(
            f"neg_type length mismatch: got {values.shape[0]}, expected {n_gw}"
        )
    return torch.from_numpy(values).to(device=device)


def _collect_dataset_window_metadata(args) -> Dict[str, object]:
    return {
        "train_positive": _read_optical_h5_window_metadata(
            getattr(args, "data_path", None)
        ),
        "train_negative": _read_optical_h5_window_metadata(
            getattr(args, "neg_data_path", None)
        ),
    }


def _build_effective_input_window_metadata(args) -> Dict[str, object]:
    pos_meta = getattr(args, "_dataset_window_metadata", {}).get("train_positive", {})
    dataset_window_start = (
        pos_meta.get("time_window_start") if isinstance(pos_meta, dict) else None
    )
    dataset_window_end = (
        pos_meta.get("time_window_end") if isinstance(pos_meta, dict) else None
    )
    return build_effective_input_window_metadata(
        float(args.ref_start),
        float(args.ref_end),
        runtime_input_window_start=float(args.ref_start),
        runtime_input_window_end=float(args.ref_end),
        dataset_window_start=(
            None if dataset_window_start is None else float(dataset_window_start)
        ),
        dataset_window_end=(
            None if dataset_window_end is None else float(dataset_window_end)
        ),
    )


def resolve_mtan_runtime_config(args) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "mtan_snr_s0": float(getattr(args, "mtan_snr_s0", 3.0)),
        "mtan_snr_beta": float(getattr(args, "mtan_snr_beta", 1.0)),
        "mtan_snr_clip_min": float(getattr(args, "mtan_snr_clip_min", -8.0)),
        "mtan_snr_clip_max": float(getattr(args, "mtan_snr_clip_max", 20.0)),
        "mtan_snr_eps": float(getattr(args, "mtan_snr_eps", 1e-9)),
        "mtan_lupt_psfflux_zp": 31.4,
        "mtan_lupt_k": 1.0,
        "mtan_lupt_m5_mag": DEFAULT_MTAN_LUPT_M5.copy(),
    }

    data_path = str(getattr(args, "data_path", "") or "")
    if data_path and os.path.exists(data_path):
        try:
            with h5py.File(data_path, "r") as f:
                if "mtan_snr_s0" in f.attrs:
                    cfg["mtan_snr_s0"] = float(f.attrs["mtan_snr_s0"])
                if "mtan_snr_beta" in f.attrs:
                    cfg["mtan_snr_beta"] = float(f.attrs["mtan_snr_beta"])
                if "mtan_snr_clip_min" in f.attrs:
                    cfg["mtan_snr_clip_min"] = float(f.attrs["mtan_snr_clip_min"])
                if "mtan_snr_clip_max" in f.attrs:
                    cfg["mtan_snr_clip_max"] = float(f.attrs["mtan_snr_clip_max"])
                if "mtan_snr_eps" in f.attrs:
                    cfg["mtan_snr_eps"] = float(f.attrs["mtan_snr_eps"])
                if "psfflux_zp" in f.attrs:
                    cfg["mtan_lupt_psfflux_zp"] = float(f.attrs["psfflux_zp"])
                if "lupt_k" in f.attrs:
                    cfg["mtan_lupt_k"] = float(f.attrs["lupt_k"])
                if "lupt_m5_mag" in f.attrs:
                    m5 = np.asarray(f.attrs["lupt_m5_mag"], dtype=np.float64).reshape(
                        -1
                    )
                    if m5.shape == (6,) and np.all(np.isfinite(m5)):
                        cfg["mtan_lupt_m5_mag"] = m5
        except Exception as exc:
            print(f"[WARN] Failed to read mTAN luptitude attrs from {data_path}: {exc}")

    if getattr(args, "mtan_lupt_psfflux_zp", None) is not None:
        cfg["mtan_lupt_psfflux_zp"] = float(args.mtan_lupt_psfflux_zp)
    if getattr(args, "mtan_lupt_k", None) is not None:
        cfg["mtan_lupt_k"] = float(args.mtan_lupt_k)
    if getattr(args, "mtan_lupt_m5_mag", None):
        cfg["mtan_lupt_m5_mag"] = parse_lupt_m5_mag_text(args.mtan_lupt_m5_mag)

    if float(cfg["mtan_lupt_k"]) <= 0:
        raise ValueError("mtan_lupt_k must be > 0.")
    if float(cfg["mtan_lupt_psfflux_zp"]) <= 0:
        raise ValueError("mtan_lupt_psfflux_zp must be > 0.")
    m5_arr = np.asarray(cfg["mtan_lupt_m5_mag"], dtype=np.float64).reshape(-1)
    if m5_arr.shape != (6,) or (not np.all(np.isfinite(m5_arr))):
        raise ValueError("mtan_lupt_m5_mag must be 6 finite values.")
    cfg["mtan_lupt_m5_mag"] = tuple(float(x) for x in m5_arr.tolist())
    return cfg


def _start_cuda_timer(enabled: bool):
    if not enabled or not torch.cuda.is_available():
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return start, end


def _stop_cuda_timer(timer) -> float:
    if timer is None:
        return 0.0
    start, end = timer
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def apply_time_offsets(opt_t, opt_mask, delta_days, scale_divisor):
    valid = (opt_mask.sum(dim=-1) > 0).to(dtype=opt_t.dtype)
    shift = (
        delta_days.to(device=opt_t.device, dtype=opt_t.dtype) / float(scale_divisor)
    ).unsqueeze(1)
    return opt_t + shift * valid


def apply_hard_negative_time_shift(opt_t, opt_mask, delta_days, scale_divisor):
    """
    Shift optical timeline for hard negatives with absolute mismatch preserved.

    t_shift = t + (delta_days / scale_divisor), where:
      delta_days = t_candidate_gw - t_anchor_gw
    """
    valid = (opt_mask.sum(dim=-1) > 0).to(dtype=opt_t.dtype)
    shift = (
        delta_days.to(device=opt_t.device, dtype=opt_t.dtype) / float(scale_divisor)
    ).unsqueeze(1)
    return opt_t + shift * valid


# NOTE: apply_time_offsets() and apply_hard_negative_time_shift() are kept as reusable
# primitives for potential future time-shift experiments, but are no longer called in
# the main training / eval pipeline after the offset infra removal.


def compute_time_delta_days(opt_zero_time_mjd, gw_anchor_time_mjd):
    dt = opt_zero_time_mjd.to(torch.float32) - gw_anchor_time_mjd.to(torch.float32)
    dt = torch.where(torch.isfinite(dt), dt, torch.zeros_like(dt))
    return dt


def compute_parent_relative_time_delta_days(
    opt_first_detection_mjd, parent_gw_event_time_mjd
):
    """Return candidate detection delay relative to its own parent GW event."""
    if opt_first_detection_mjd is None or parent_gw_event_time_mjd is None:
        raise ValueError(
            "parent-relative time deltas require first-detection and parent GW times."
        )
    first = opt_first_detection_mjd.to(torch.float32).reshape(-1)
    parent = parent_gw_event_time_mjd.to(
        device=first.device, dtype=torch.float32
    ).reshape(-1)
    if first.shape != parent.shape:
        raise ValueError(
            "first-detection and parent GW time arrays must have equal shapes."
        )
    dt = first - parent
    if not torch.isfinite(dt).all():
        raise ValueError("parent-relative time deltas must all be finite.")
    return dt


def sample_mixed_external_time_deltas(
    positive_dt_days,
    n_samples,
    *,
    empirical_fraction=0.5,
    window_days=30.0,
    generator=None,
):
    """Sample external-negative dt from empirical KN and operational priors."""
    pool = positive_dt_days.to(torch.float32).reshape(-1)
    pool = pool[torch.isfinite(pool)]
    if pool.numel() == 0:
        raise ValueError("mixed external dt sampling requires a finite positive pool.")
    n_samples = int(n_samples)
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative.")
    fraction = float(empirical_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("empirical_fraction must be in [0, 1].")
    window_days = float(window_days)
    if window_days <= 0.0:
        raise ValueError("window_days must be positive.")
    if n_samples == 0:
        return pool.new_empty((0,))

    n_empirical = round(n_samples * fraction)
    n_uniform = n_samples - n_empirical
    parts = []
    if n_empirical:
        indices = torch.randint(
            pool.numel(),
            (n_empirical,),
            device=pool.device,
            generator=generator,
        )
        parts.append(pool[indices])
    if n_uniform:
        parts.append(
            torch.rand(
                (n_uniform,),
                device=pool.device,
                dtype=pool.dtype,
                generator=generator,
            )
            * window_days
        )
    sampled = torch.cat(parts, dim=0)
    permutation = torch.randperm(
        sampled.numel(), device=sampled.device, generator=generator
    )
    return sampled[permutation]


def build_mismatched_classification_time_deltas(
    *,
    mode,
    batch_size,
    device,
    window_days,
    candidate_indices,
    opt_first_detection_mjd,
    parent_gw_event_time_mjd,
):
    """Build dt for in-batch mismatched-KN classification pairs."""
    if str(mode).strip().lower() == "mixed_empirical":
        return compute_parent_relative_time_delta_days(
            opt_first_detection_mjd[candidate_indices],
            parent_gw_event_time_mjd[candidate_indices],
        )
    return sample_mis_neg_dt_days(batch_size, device, window_days=window_days)


def build_external_classification_time_deltas(
    *,
    mode,
    positive_dt_pool,
    n_samples,
    empirical_fraction,
    window_days,
    negative_event_time_mjd,
    query_event_time_mjd,
    device,
    generator=None,
):
    """Build dt for external non-KN classification negatives."""
    if str(mode).strip().lower() == "mixed_empirical":
        return sample_mixed_external_time_deltas(
            positive_dt_pool,
            n_samples,
            empirical_fraction=empirical_fraction,
            window_days=window_days,
            generator=generator,
        )
    if negative_event_time_mjd is not None and query_event_time_mjd is not None:
        return compute_time_delta_days(negative_event_time_mjd, query_event_time_mjd)
    return torch.zeros((int(n_samples),), device=device, dtype=torch.float32)


def resolve_optical_candidate_time_mjd(
    opt_first_detection_mjd, fallback_event_time_mjd
):
    """Return the optical first-detection MJD used for candidate timing."""
    del fallback_event_time_mjd
    return opt_first_detection_mjd


def gather_candidate_optical_time_mjd(
    opt_first_detection_mjd, fallback_event_time_mjd, candidate_idx
):
    candidate_time = resolve_optical_candidate_time_mjd(
        opt_first_detection_mjd,
        fallback_event_time_mjd,
    )
    if candidate_time is None:
        return None
    return candidate_time[candidate_idx]


def encode_hard_negative_with_time_shift(
    model,
    *,
    opt_coords,
    opt_t,
    opt_v,
    opt_mask,
    opt_err,
    opt_ref_t,
    candidate_zero_time_mjd,
    anchor_gw_time_mjd,
    scale_divisor,
):
    """
    Re-encode hard negatives after timeline shift to preserve absolute mismatch.
    """
    if candidate_zero_time_mjd is None or anchor_gw_time_mjd is None:
        delta_days = torch.zeros(
            (opt_t.size(0),), device=opt_t.device, dtype=torch.float32
        )
        shifted_opt_t = opt_t
    else:
        delta_days = compute_time_delta_days(
            candidate_zero_time_mjd, anchor_gw_time_mjd
        )
        shifted_opt_t = apply_hard_negative_time_shift(
            opt_t, opt_mask, delta_days, scale_divisor
        )
    z_l_hard, h_l_hard = model.encode_optical(
        opt_coords, shifted_opt_t, opt_v, opt_ref_t, opt_mask, opt_err
    )
    return z_l_hard, h_l_hard, delta_days


def summarize_time_delta(dt_days):
    finite = torch.isfinite(dt_days)
    if not finite.any():
        return 0.0, 0.0
    vals = dt_days[finite].to(torch.float32)
    mean = float(vals.mean().item())
    std = float(vals.std(unbiased=False).item())
    return mean, std


def _accumulate_time_delta_stats(stats, key, dt_days):
    finite = torch.isfinite(dt_days)
    if not finite.any():
        return
    vals = dt_days[finite].to(torch.float32)
    stats[key]["sum"] += float(vals.sum().item())
    stats[key]["sum_sq"] += float((vals * vals).sum().item())
    stats[key]["count"] += int(vals.numel())


def _finalize_time_delta_stats(stats):
    out = {}
    for key, entry in stats.items():
        count = int(entry["count"])
        if count <= 0:
            out[f"{key}_mean"] = 0.0
            out[f"{key}_std"] = 0.0
            continue
        mean = entry["sum"] / float(count)
        var = max(0.0, entry["sum_sq"] / float(count) - mean * mean)
        out[f"{key}_mean"] = float(mean)
        out[f"{key}_std"] = float(var**0.5)
    return out


def summarize_abs_time_delta_quantiles(dt_days, quantiles=(0.5, 0.9, 0.95)):
    finite = torch.isfinite(dt_days)
    if not finite.any():
        return {f"p{int(q*100):02d}": 0.0 for q in quantiles}
    vals = dt_days[finite].abs().to(torch.float32)
    q = torch.tensor(list(quantiles), device=vals.device, dtype=vals.dtype)
    q_vals = torch.quantile(vals, q)
    out = {}
    for i, qq in enumerate(quantiles):
        out[f"p{int(qq * 100):02d}"] = float(q_vals[i].item())
    return out


def sample_mismatched_negatives(
    batch_size,
    device,
    samples_per_gw=1,
    gw_indices=None,
    source_types=None,
    eligible_candidate_mask=None,
):
    """
    Select an optical curve from a different GW event for each anchor.

    When GW indices and source types are available, candidates from the same
    source type are preferred. If no other GW of that type exists in the batch,
    sampling falls back to any different GW event.
    """
    if batch_size < 2:
        return None

    if gw_indices is not None:
        gw_indices = torch.as_tensor(gw_indices, device=device).reshape(-1)
        if gw_indices.numel() != batch_size:
            raise ValueError(
                f"gw_indices length ({gw_indices.numel()}) != batch_size ({batch_size})"
            )
        if torch.unique(gw_indices).numel() < 2:
            return None

        if source_types is not None:
            source_types = torch.as_tensor(source_types, device=device).reshape(-1)
            if source_types.numel() != batch_size:
                raise ValueError(
                    f"source_types length ({source_types.numel()}) "
                    f"!= batch_size ({batch_size})"
                )

        different_gw = gw_indices.unsqueeze(1) != gw_indices.unsqueeze(0)
        if eligible_candidate_mask is not None:
            eligible = torch.as_tensor(
                eligible_candidate_mask, device=device, dtype=torch.bool
            ).reshape(-1)
            if eligible.numel() != batch_size:
                raise ValueError(
                    "eligible_candidate_mask must have one value per batch row."
                )
            different_gw = different_gw & eligible.unsqueeze(0)
        if not bool(different_gw.any(dim=1).all()):
            return None
        candidate_mask = different_gw
        if source_types is not None:
            same_source = source_types.unsqueeze(1) == source_types.unsqueeze(0)
            same_source_candidates = different_gw & same_source
            has_same_source_candidate = same_source_candidates.any(dim=1, keepdim=True)
            candidate_mask = torch.where(
                has_same_source_candidate,
                same_source_candidates,
                different_gw,
            )

        random_scores = torch.rand((batch_size, batch_size), device=device)
        random_scores.masked_fill_(~candidate_mask, -1.0)
        return random_scores.argmax(dim=1)

    n_gw = batch_size // samples_per_gw
    if n_gw < 2:
        return None

    mis_idx = torch.empty(batch_size, dtype=torch.long, device=device)

    if samples_per_gw == 1:
        # n_gw == batch_size; exclude self
        mis_idx = torch.randint(0, batch_size - 1, (batch_size,), device=device)
        mis_idx = mis_idx + (mis_idx >= torch.arange(batch_size, device=device)).long()
    else:
        for i in range(batch_size):
            my_group = i // samples_per_gw
            candidates = [g for g in range(n_gw) if g != my_group]
            chosen_group = candidates[
                torch.randint(0, len(candidates), (1,), device=device).item()
            ]
            offset = torch.randint(0, samples_per_gw, (1,), device=device).item()
            mis_idx[i] = chosen_group * samples_per_gw + offset

    return mis_idx


def build_itc_pair_mask(gw_indices, pair_is_neg_gw, min_pairs_per_gw=2):
    """Keep only positive GW events with enough optical views for SupCon."""
    gw_indices = torch.as_tensor(gw_indices).reshape(-1)
    pair_is_neg_gw = torch.as_tensor(
        pair_is_neg_gw, device=gw_indices.device, dtype=torch.bool
    ).reshape(-1)
    if gw_indices.numel() != pair_is_neg_gw.numel():
        raise ValueError("gw_indices and pair_is_neg_gw must have equal length.")
    _, inverse, counts = torch.unique(
        gw_indices, return_inverse=True, return_counts=True
    )
    eligible_count = counts[inverse] >= max(1, int(min_pairs_per_gw))
    return (~pair_is_neg_gw) & eligible_count


def _parameter_stage_role(name):
    name = str(name).replace("_orig_mod.", "")
    if name == "log_temp":
        return "temperature"
    if name.startswith("fusion.") or "gw_encoder.skymap_seq_proj." in name:
        return "head"
    return "encoder"


def configure_model_for_stage(model, args, stage):
    """Freeze/unfreeze model parts for the resolved training stage."""
    if not is_staged_training_enabled(args):
        for parameter in model.parameters():
            parameter.requires_grad = True
        return
    itc_enabled = float(getattr(args, "itc_weight", 0.0)) > 0.0
    for name, parameter in model.named_parameters():
        role = _parameter_stage_role(name)
        if stage == "itc":
            parameter.requires_grad = role in {"encoder", "temperature"}
        elif stage in {"cls_intro", "retrieval_intro"}:
            parameter.requires_grad = role in (
                {"head"} if itc_enabled else {"encoder", "head"}
            )
        else:
            parameter.requires_grad = role in {"encoder", "head"}


def build_optimizer_param_groups(model, args):
    """Build stage-aware AdamW groups and exclude norm/bias/temp from decay."""
    grouped = {}
    for name, parameter in model.named_parameters():
        clean_name = str(name).replace("_orig_mod.", "")
        role = _parameter_stage_role(clean_name)
        no_decay = (
            parameter.ndim < 2
            or clean_name.endswith(".bias")
            or "norm" in clean_name.lower()
            or "bn" in clean_name.lower()
            or role == "temperature"
        )
        key = (role, no_decay)
        grouped.setdefault(key, []).append(parameter)

    groups = []
    for (role, no_decay), parameters in sorted(grouped.items()):
        groups.append(
            {
                "params": parameters,
                "lr": float(args.lr),
                "weight_decay": 0.0 if no_decay else float(args.weight_decay),
                "stage_role": role,
            }
        )
    return groups


def sample_mis_neg_dt_days(batch_size, device, window_days=30.0):
    """Synthetic time delta for mismatched negatives: Uniform(0, window_days) days."""
    return torch.rand(batch_size, device=device, dtype=torch.float32) * window_days


def augment_gw_data(
    gw_s,
    gw_m,
    training=True,
    noise_std=0.05,
    scalar_jitter=0.02,
    channel_dropout_prob=0.1,
):
    """
    GW数据增强函数，增加训练样本的有效多样性。

    Args:
        gw_s: GW scalar features [Batch, 7] (质量、自旋等参数)
        gw_m: GW skymap [Batch, 7, 19200] (MOC skymap序列)
        training: 是否在训练模式（验证时不增强）
        noise_std: skymap高斯噪声标准差
        scalar_jitter: scalar参数扰动比例
        channel_dropout_prob: 随机dropout某个skymap通道的概率

    Returns:
        augmented gw_s, gw_m
    """
    if not training:
        return gw_s, gw_m

    # 1. Skymap probability augmentation in log space
    # Channel 4 stores 100 * probdensity * dA (probability mass per pixel).
    # Convert to normalized probability, perturb in log space, renormalize.
    eps = 1e-30
    dP = gw_m[:, 4, :]  # [Batch, 19200]
    dP_sum = dP.sum(dim=-1, keepdim=True).clamp(min=eps)
    p = dP / dP_sum  # normalized probability
    noise = torch.randn_like(p)
    p_perturbed = p * torch.exp(noise * noise_std)
    p_perturbed = p_perturbed / p_perturbed.sum(dim=-1, keepdim=True).clamp(min=eps)
    # Scale back to original dP sum
    gw_m = gw_m.clone()
    gw_m[:, 4, :] = p_perturbed * dP_sum

    # 2. Scalar参数扰动
    # 对质量、自旋、距离等参数添加小幅随机扰动
    # 使用乘性噪声保持参数的量纲
    scalar_multiplier = 1.0 + (torch.rand_like(gw_s) - 0.5) * 2 * scalar_jitter
    gw_s = gw_s * scalar_multiplier

    # 3. Skymap通道Dropout（可选）
    # 随机将某个通道（如distance信息）置零，增加鲁棒性
    if torch.rand(1).item() < channel_dropout_prob:
        # 随机选择一个通道（0-6），但避免dropout概率通道(index 4)
        channel_idx = torch.randint(0, gw_m.size(1), (1,)).item()
        if channel_idx != 4:  # 保留概率通道
            gw_m[:, channel_idx, :] = 0

    return gw_s, gw_m


def compute_credible_level(gw_m, opt_coords):
    """
    For each (GW, optical) pair, compute the credible level of the optical
    transient's sky position in the GW skymap.

    Uses skymap channels 0-2 (x, y, z unit vectors of pixel centers) to find
    the nearest pixel to the optical RA/Dec, then computes the cumulative
    probability mass with higher or equal probability density than that pixel.

    Args:
        gw_m: [B, 7, 19200] — skymap channels [x, y, z, dA, dP, distmu, distsigma]
        opt_coords: [B, 2] — (RA, Dec) in degrees (dataset default). If values
            look like radians (max |coord| <= ~2π), they are treated as radians.
    Returns:
        cred_level: [B, 1] — small = in high-probability region, large = in low-probability region
    """
    # NOTE: The dataset stores RA/Dec in degrees (e.g., from SNANA HEAD.FITS),
    # while some model code paths historically assumed radians. To keep this
    # function robust, auto-detect units and convert to radians if needed.
    # Heuristic: if any coord exceeds ~2π in magnitude, it's almost certainly degrees.
    coords = opt_coords
    if coords.numel() > 0:
        max_abs = coords.detach().abs().max()
        if max_abs > (2 * math.pi + 1e-3):
            coords = coords * (math.pi / 180.0)

    ra = coords[:, 0]
    dec = coords[:, 1]
    opt_xyz = torch.stack(
        [
            torch.cos(dec) * torch.cos(ra),
            torch.cos(dec) * torch.sin(ra),
            torch.sin(dec),
        ],
        dim=-1,
    )  # [B, 3]

    pix_xyz = gw_m[:, :3, :]  # [B, 3, 19200]
    dot = torch.bmm(opt_xyz.unsqueeze(1), pix_xyz).squeeze(1)  # [B, 19200]
    nearest_idx = dot.argmax(dim=-1)  # [B]

    dA = gw_m[:, 3, :]  # [B, 19200]
    dP = gw_m[:, 4, :]  # [B, 19200]
    dA = torch.nan_to_num(
        dA.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    dP = torch.nan_to_num(
        dP.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    density = dP / dA.clamp_min(torch.finfo(dP.dtype).eps)
    density_at_opt = density[
        torch.arange(density.size(0), device=density.device), nearest_idx
    ]  # [B]

    # Credible level: posterior mass with density >= density at the optical position.
    total_probability = dP.sum(dim=-1).clamp_min(torch.finfo(dP.dtype).eps)
    cred_level = (
        torch.where(
            density >= density_at_opt.unsqueeze(-1),
            dP,
            torch.zeros_like(dP),
        ).sum(dim=-1)
        / total_probability
    )  # [B]
    return cred_level.unsqueeze(-1)  # [B, 1]


def _unwrap_compiled_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _model_requires_cred_level(model) -> bool:
    base = _unwrap_compiled_model(model)
    if hasattr(base, "uses_cred_level_input"):
        return bool(base.uses_cred_level_input())
    return bool(getattr(base, "dual_fusion", False))


def augment_optical_data(
    opt_t,
    opt_v,
    opt_mask,
    opt_err,
    training=True,
    time_jitter=0.0,
    flux_noise=0.0,
    obs_dropout=0.0,
    band_dropout=0.0,
):
    """
    Optical data augmentation to improve generalization.
    """
    if not training:
        return opt_t, opt_v, opt_mask, opt_err

    if time_jitter > 0:
        time_mask = (opt_mask.sum(dim=-1) > 0).float()
        opt_t = opt_t + torch.randn_like(opt_t) * time_jitter * time_mask

    if flux_noise > 0:
        if opt_err is not None:
            noise = torch.randn_like(opt_v) * (opt_err * flux_noise)
        else:
            noise = torch.randn_like(opt_v) * flux_noise
        opt_v = opt_v + noise * opt_mask

    if obs_dropout > 0:
        drop_mask = (torch.rand_like(opt_mask) < obs_dropout) & (opt_mask > 0)
        if drop_mask.any():
            opt_mask = opt_mask.masked_fill(drop_mask, 0)
            opt_v = opt_v.masked_fill(drop_mask, 0.0)
            if opt_err is not None:
                opt_err = opt_err.masked_fill(drop_mask, 0.0)

    if band_dropout > 0:
        band_mask = (
            torch.rand(opt_v.size(0), opt_v.size(2), device=opt_v.device) < band_dropout
        )
        if band_mask.any():
            band_mask = band_mask[:, None, :]
            opt_mask = opt_mask.masked_fill(band_mask, 0)
            opt_v = opt_v.masked_fill(band_mask, 0.0)
            if opt_err is not None:
                opt_err = opt_err.masked_fill(band_mask, 0.0)

    return opt_t, opt_v, opt_mask, opt_err


def is_staged_training_enabled(args):
    return bool(getattr(args, "staged_training_enable", False))


def get_training_schedule_total_epochs(args):
    """Return the fixed schedule horizon used across resumable HPO rungs."""
    configured = getattr(args, "training_schedule_total_epochs", None)
    if configured is None:
        configured = getattr(args, "lr_schedule_total_epochs", None)
    return int(configured if configured is not None else getattr(args, "epochs", 1))


def build_sequential_curriculum(args):
    """Build enabled loss phases in order, using 0-based half-open ranges."""
    if not is_staged_training_enabled(args):
        return []
    enabled = {
        "itc": float(getattr(args, "itc_weight", 0.0)) > 0.0,
        "cls_intro": float(getattr(args, "cls_weight", 0.0)) > 0.0,
        "retrieval_intro": float(getattr(args, "gallery_loss_weight", 0.0)) > 0.0,
    }
    durations = {
        "itc": max(0, int(getattr(args, "stage_itc_epochs", 8))),
        "cls_intro": max(0, int(getattr(args, "stage_cls_ramp_epochs", 4))),
        "retrieval_intro": max(0, int(getattr(args, "stage_retrieval_ramp_epochs", 4))),
    }
    phases = []
    start = 0
    for name in ("itc", "cls_intro", "retrieval_intro"):
        if enabled[name] and durations[name] > 0:
            phases.append(
                {"name": name, "start": start, "end": start + durations[name]}
            )
            start += durations[name]
    phases.append(
        {
            "name": "joint",
            "start": start,
            "end": get_training_schedule_total_epochs(args),
        }
    )
    return phases


def get_curriculum_phase_start(args, phase_name):
    if not is_staged_training_enabled(args):
        legacy_starts = {
            "cls_intro": int(getattr(args, "cls_start_epoch", 0)),
            "retrieval_intro": int(getattr(args, "retrieval_start_epoch", 0)),
            "itc": 0,
        }
        return legacy_starts.get(phase_name, int(getattr(args, "epochs", 1)))
    for phase in build_sequential_curriculum(args):
        if phase["name"] == phase_name:
            return phase["start"]
    return int(getattr(args, "epochs", 1))


def resolve_sequential_curriculum(args, epoch):
    """Resolve phase, weights, and trainable roles for one sequential epoch."""
    epoch = int(epoch)
    phases = build_sequential_curriculum(args)
    if not phases:
        raise ValueError("Sequential curriculum requires staged_training_enable=true.")
    phase = next(
        (item for item in phases if item["start"] <= epoch < item["end"]), phases[-1]
    )
    name = phase["name"]
    itc_target = max(0.0, float(getattr(args, "itc_weight", 0.0)))
    cls_target = max(0.0, float(getattr(args, "cls_weight", 0.0)))
    retrieval_target = max(0.0, float(getattr(args, "gallery_loss_weight", 0.0)))
    weights = {"itc": 0.0, "cls": 0.0, "retrieval": 0.0}
    if name == "itc":
        weights["itc"] = itc_target
    elif name == "cls_intro":
        progress = (epoch - phase["start"] + 1) / float(
            max(1, phase["end"] - phase["start"])
        )
        weights["cls"] = cls_target * min(1.0, progress)
    elif name == "retrieval_intro":
        progress = (epoch - phase["start"] + 1) / float(
            max(1, phase["end"] - phase["start"])
        )
        weights["cls"] = cls_target
        weights["retrieval"] = retrieval_target * min(1.0, progress)
    else:
        weights["cls"] = cls_target
        weights["retrieval"] = retrieval_target
        if itc_target > 0.0:
            joint_epochs = max(1, phase["end"] - phase["start"])
            progress = min(
                1.0,
                max(0.0, (epoch - phase["start"]) / float(max(1, joint_epochs - 1))),
            )
            start_weight = float(getattr(args, "stage_joint_itc_start_weight", 0.5))
            end_weight = float(getattr(args, "stage_joint_itc_end_weight", 0.25))
            weights["itc"] = itc_target * (
                start_weight + (end_weight - start_weight) * progress
            )
    itc_pretrained = itc_target > 0.0
    if name == "itc":
        roles = {"encoder", "temperature"}
    elif name in {"cls_intro", "retrieval_intro"} and itc_pretrained:
        roles = {"head"}
    else:
        roles = {"encoder", "head"}
    return {
        "phase": name,
        "start": phase["start"],
        "end": phase["end"],
        "weights": weights,
        "train_roles": roles,
    }


def format_sequential_curriculum(args):
    parts = []
    for phase in build_sequential_curriculum(args):
        if phase["end"] > phase["start"]:
            parts.append(f"{phase['name']}=epochs {phase['start'] + 1}-{phase['end']}")
    return ", ".join(parts)


def validate_sequential_curriculum(args):
    if not is_staged_training_enabled(args):
        return
    for key in (
        "stage_itc_epochs",
        "stage_cls_ramp_epochs",
        "stage_retrieval_ramp_epochs",
    ):
        if int(getattr(args, key, 0)) < 0:
            raise ValueError(f"{key} must be >= 0.")
    if not any(
        float(getattr(args, key, 0.0)) > 0.0
        for key in ("itc_weight", "cls_weight", "gallery_loss_weight")
    ):
        raise ValueError("Sequential curriculum has no enabled loss.")
    phases = build_sequential_curriculum(args)
    if phases[-1]["start"] >= int(getattr(args, "epochs", 1)):
        raise ValueError(
            "Sequential curriculum intro phases leave no joint-training epoch."
        )
    for phase in phases:
        if phase["end"] <= phase["start"]:
            continue
        state = resolve_sequential_curriculum(args, phase["start"])
        if not any(value > 0.0 for value in state["weights"].values()):
            raise ValueError(
                f"Sequential phase {phase['name']!r} has no effective loss."
            )
        if not state["train_roles"]:
            raise ValueError(
                f"Sequential phase {phase['name']!r} has no trainable gradient recipient."
            )


def resolve_training_stage(args, epoch):
    """Resolve the active sequential loss phase for an epoch."""
    if not is_staged_training_enabled(args):
        return "legacy"
    return resolve_sequential_curriculum(args, epoch)["phase"]


def _common_lr_scale(args, step, steps_per_epoch):
    total_steps = max(
        1, get_training_schedule_total_epochs(args) * int(steps_per_epoch)
    )
    warmup_steps = max(0, int(args.warmup_epochs) * int(steps_per_epoch))
    min_lr = args.min_lr if args.min_lr is not None else 0.0
    min_lr_ratio = min(float(min_lr) / float(args.lr), 1.0)
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_lr_scheduler(optimizer, args, steps_per_epoch, start_step):
    if args.lr_scheduler == "none":
        return None

    def make_lr_lambda(role):
        def lr_lambda(step):
            common = _common_lr_scale(args, step, steps_per_epoch)
            if not is_staged_training_enabled(args):
                return common
            epoch = min(
                get_training_schedule_total_epochs(args) - 1,
                max(0, int(step) // max(1, int(steps_per_epoch))),
            )
            stage = resolve_training_stage(args, epoch)
            if role == "encoder":
                if (
                    stage in {"cls_intro", "retrieval_intro"}
                    and float(getattr(args, "itc_weight", 0.0)) > 0.0
                ):
                    return 0.0
                if stage == "joint":
                    if float(getattr(args, "itc_weight", 0.0)) <= 0.0:
                        return common
                    return common * float(getattr(args, "encoder_lr_ratio", 0.1))
                return common
            if role == "head":
                return 0.0 if stage == "itc" else common
            if role == "temperature":
                return common if stage == "itc" else 0.0
            return common

        return lr_lambda

    lambdas = [
        make_lr_lambda(group.get("stage_role", "all"))
        for group in optimizer.param_groups
    ]
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambdas, last_epoch=start_step - 1
    )


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch_state = state["torch"]
    if isinstance(torch_state, torch.Tensor):
        torch_state = torch_state.detach().cpu()
    torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            item.detach().cpu() if isinstance(item, torch.Tensor) else item
            for item in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def load_training_checkpoint(path, map_location):
    """Load a pipeline checkpoint with a minimal NumPy RNG-state allowlist."""
    numpy_uint32_dtype = type(np.dtype(np.uint32))
    safe_globals = [
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        numpy_uint32_dtype,
    ]
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(path, map_location=map_location, weights_only=True)


def save_training_checkpoint_atomic(value, path, max_attempts=5):
    """Atomically replace a checkpoint, retrying transient filesystem errors."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    max_attempts = max(1, int(max_attempts))
    for attempt in range(1, max_attempts + 1):
        temporary_path = f"{path}.tmp.{os.getpid()}.{attempt}"
        try:
            torch.save(value, temporary_path)
            os.replace(temporary_path, path)
            return
        except (OSError, RuntimeError) as exc:
            if attempt >= max_attempts:
                raise
            delay = min(2 ** (attempt - 1), 30)
            print(
                f"[WARN] Checkpoint write attempt {attempt}/{max_attempts} failed "
                f"for {path}: {exc}; retrying in {delay}s."
            )
            time.sleep(delay)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)


def save_last_checkpoint_resilient(value, path):
    """Preserve the prior last checkpoint if all transient-write retries fail."""
    try:
        save_training_checkpoint_atomic(value, path)
        return True
    except (OSError, RuntimeError) as exc:
        print(
            f"[WARN] Could not update last checkpoint after retries: {exc}. "
            "The previous checkpoint is preserved; training will continue."
        )
        return False


def write_json_atomic(path, value):
    """Atomically replace a JSON summary in the destination directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w") as handle:
            json.dump(value, handle, indent=2)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def should_run_validation(epoch, epochs, interval):
    interval = max(1, int(interval))
    return (int(epoch) + 1) % interval == 0 or int(epoch) + 1 == int(epochs)


def compute_cls_weight(args, epoch):
    if is_staged_training_enabled(args):
        return resolve_sequential_curriculum(args, epoch)["weights"]["cls"]
    if epoch < args.cls_start_epoch:
        return 0.0
    if args.cls_ramp_epochs <= 0:
        return args.cls_weight
    progress = (epoch - args.cls_start_epoch + 1) / float(args.cls_ramp_epochs)
    return args.cls_weight * min(1.0, progress)


def is_cls_branch_enabled(args):
    return float(getattr(args, "cls_weight", 0.0)) > 0.0


def is_gallery_enabled(args):
    return float(getattr(args, "gallery_loss_weight", 0.0)) > 0.0


def compute_gallery_loss_weight(args, epoch):
    if is_staged_training_enabled(args):
        return resolve_sequential_curriculum(args, epoch)["weights"]["retrieval"]
    target = float(getattr(args, "gallery_loss_weight", 0.0))
    if target <= 0.0:
        return 0.0
    start = int(getattr(args, "retrieval_start_epoch", 0))
    epoch = int(epoch)
    if epoch < start:
        return 0.0
    ramp = int(getattr(args, "gallery_loss_ramp_epochs", 0))
    if ramp <= 0:
        return target
    progress = min(1.0, (epoch - start + 1) / float(ramp))
    return target * progress


CLS_BEST_CKPT_METRICS = {
    "auprc",
    "auroc",
    "f1_optimal",
    "acc_total",
    "cls_composite_auprc_auroc",
}
G2O_BEST_CKPT_METRICS = {
    "g2o_recall_at_1",
    "g2o_recall_at_5",
    "g2o_mrr",
}
FUSION_GALLERY_BEST_CKPT_METRICS = {
    "fusion_gallery_recall_at_1",
    "fusion_gallery_recall_at_5",
    "fusion_gallery_mrr",
}
HARD_GALLERY_BEST_CKPT_METRICS = {
    "hard_gallery_macro_retrieval_score",
}


def should_compute_cls_metrics(args):
    return bool(getattr(args, "compute_cls_metrics", True))


def should_compute_fusion_gallery_metrics(args, epoch):
    if not bool(getattr(args, "compute_fusion_gallery_metrics", True)):
        return False
    start = int(getattr(args, "fusion_gallery_metrics_start_epoch", 0))
    return int(epoch) >= start


def is_retrieval_active(args, epoch):
    """Return True when retrieval/gallery training is active at the given epoch."""
    if not is_gallery_enabled(args):
        return False
    if is_staged_training_enabled(args):
        return compute_gallery_loss_weight(args, epoch) > 0.0
    start = int(getattr(args, "retrieval_start_epoch", 0))
    return int(epoch) >= start


def is_cls_active(epoch, args):
    """Return True when classification loss is active (after ramp)."""
    return compute_cls_weight(args, epoch) > 0.0


def compute_itc_weight(args, epoch):
    if is_staged_training_enabled(args):
        return resolve_sequential_curriculum(args, epoch)["weights"]["itc"]
    if args.itc_decay_epochs <= 0 or args.itc_decay_ratio <= 0:
        return args.itc_weight
    if epoch < args.itc_decay_start_epoch:
        return args.itc_weight
    progress = (epoch - args.itc_decay_start_epoch + 1) / float(args.itc_decay_epochs)
    progress = min(1.0, progress)
    return args.itc_weight * (1.0 - args.itc_decay_ratio * progress)


def compute_curriculum_full_epoch(start_epoch, ramp_epochs):
    start_epoch = max(0, int(start_epoch))
    ramp_epochs = int(ramp_epochs)
    if ramp_epochs <= 0:
        return start_epoch
    return start_epoch + ramp_epochs - 1


def gallery_loss_full_activation_reachable(args):
    if not is_gallery_enabled(args):
        return False
    total_epochs = max(1, int(getattr(args, "epochs", 1)))
    full_epoch = compute_curriculum_full_epoch(
        getattr(args, "retrieval_start_epoch", 0),
        getattr(args, "gallery_loss_ramp_epochs", 0),
    )
    return full_epoch < total_epochs


def compute_best_ckpt_stage_epochs(args):
    """Return the final 0-based epoch of each enabled staged-training phase."""
    stage_epochs = {}

    warmup_epochs = int(getattr(args, "warmup_epochs", 0))
    if str(getattr(args, "lr_scheduler", "none")) != "none" and warmup_epochs > 0:
        stage_epochs["lr_warmup"] = compute_curriculum_full_epoch(0, warmup_epochs)

    if is_staged_training_enabled(args):
        stage_epochs["joint_finetune_start"] = build_sequential_curriculum(args)[-1][
            "start"
        ]
    elif is_cls_branch_enabled(args):
        stage_epochs["classification"] = compute_curriculum_full_epoch(
            getattr(args, "cls_start_epoch", 0),
            getattr(args, "cls_ramp_epochs", 0),
        )

    if is_gallery_enabled(args):
        retrieval_start = (
            get_curriculum_phase_start(args, "retrieval_intro")
            if is_staged_training_enabled(args)
            else int(getattr(args, "retrieval_start_epoch", 0))
        )
        if not is_staged_training_enabled(args):
            stage_epochs["fusion_gallery"] = compute_curriculum_full_epoch(
                retrieval_start,
                getattr(args, "gallery_loss_ramp_epochs", 0),
            )
        if (
            is_gallery_hard_neg_enabled(args)
            and float(getattr(args, "gallery_hard_neg_weight", 0.0)) > 0.0
        ):
            hard_start = retrieval_start + int(
                getattr(args, "gallery_hard_neg_start_after_retrieval_epochs", 2)
            )
            stage_epochs["fusion_gallery_hard_negative"] = (
                compute_curriculum_full_epoch(
                    hard_start,
                    getattr(args, "gallery_hard_neg_ramp_epochs", 2),
                )
            )

    if (
        not is_staged_training_enabled(args)
        and float(getattr(args, "itc_weight", 0.0)) > 0.0
        and int(getattr(args, "itc_decay_epochs", 0)) > 0
        and float(getattr(args, "itc_decay_ratio", 0.0)) > 0.0
    ):
        stage_epochs["itc_decay"] = compute_curriculum_full_epoch(
            getattr(args, "itc_decay_start_epoch", 0),
            getattr(args, "itc_decay_epochs", 0),
        )

    return stage_epochs


def compute_best_ckpt_stable_epoch(args):
    """Return the first 0-based epoch whose validation follows all enabled phases."""
    return max(compute_best_ckpt_stage_epochs(args).values(), default=0)


def compute_best_ckpt_metric_ready_epoch(args):
    """Return the first 0-based epoch at which the selected metric is available."""
    metric_name = str(getattr(args, "best_ckpt_metric", "fusion_gallery_mrr"))

    if metric_name in CLS_BEST_CKPT_METRICS:
        if not should_compute_cls_metrics(args):
            raise ValueError(
                f"best_ckpt_metric={metric_name!r} requires compute_cls_metrics=true."
            )
        return 0

    if metric_name in FUSION_GALLERY_BEST_CKPT_METRICS:
        if not bool(getattr(args, "compute_fusion_gallery_metrics", True)):
            raise ValueError(
                f"best_ckpt_metric={metric_name!r} requires "
                "compute_fusion_gallery_metrics=true."
            )
        return max(0, int(getattr(args, "fusion_gallery_metrics_start_epoch", 0)))

    if metric_name in HARD_GALLERY_BEST_CKPT_METRICS:
        if not bool(getattr(args, "validation_gallery_enable", True)):
            raise ValueError(
                f"best_ckpt_metric={metric_name!r} requires validation_gallery_enable=true."
            )
        return 0

    if metric_name in G2O_BEST_CKPT_METRICS:
        return 0

    raise ValueError(f"Unsupported best_ckpt_metric: {metric_name}")


def compute_best_ckpt_selection_start_epoch(args):
    """Return the effective 0-based best-checkpoint tracking start epoch."""
    automatic_epoch = compute_best_ckpt_stable_epoch(args)
    metric_ready_epoch = compute_best_ckpt_metric_ready_epoch(args)
    manual_epoch = getattr(args, "best_ckpt_start_epoch", None)
    if manual_epoch is None:
        manual_epoch = 0
    return max(automatic_epoch, metric_ready_epoch, int(manual_epoch))


def validate_best_ckpt_selection_schedule(args):
    """Validate that all required phases finish while checkpointing is possible."""
    total_epochs = int(getattr(args, "epochs", 0))
    if total_epochs <= 0:
        raise ValueError("epochs must be > 0.")

    manual_epoch = getattr(args, "best_ckpt_start_epoch", None)
    if manual_epoch is not None and int(manual_epoch) < 0:
        raise ValueError("best_ckpt_start_epoch must be >= 0 or null.")

    stage_epochs = compute_best_ckpt_stage_epochs(args)
    unreachable = {
        name: epoch
        for name, epoch in stage_epochs.items()
        if int(epoch) >= total_epochs
    }
    if unreachable:
        details = ", ".join(
            f"{name}=epoch {epoch + 1}" for name, epoch in sorted(unreachable.items())
        )
        raise ValueError(
            "Best-checkpoint stability phases do not complete within "
            f"epochs={total_epochs}: {details}."
        )

    selection_start = compute_best_ckpt_selection_start_epoch(args)
    if selection_start >= total_epochs:
        raise ValueError(
            "Best-checkpoint selection cannot start within training: "
            f"start epoch={selection_start + 1}, epochs={total_epochs}."
        )
    return selection_start


def format_best_ckpt_selection_schedule(args):
    """Format the resolved checkpoint schedule for the training log."""
    stage_epochs = compute_best_ckpt_stage_epochs(args)
    stage_text = ", ".join(
        f"{name}=epoch {epoch + 1}"
        for name, epoch in sorted(stage_epochs.items(), key=lambda item: item[1])
    )
    if not stage_text:
        stage_text = "none"
    stable_epoch = compute_best_ckpt_stable_epoch(args)
    metric_epoch = compute_best_ckpt_metric_ready_epoch(args)
    manual_epoch = getattr(args, "best_ckpt_start_epoch", None)
    selection_epoch = compute_best_ckpt_selection_start_epoch(args)
    manual_text = "none" if manual_epoch is None else f"epoch {int(manual_epoch) + 1}"
    return (
        f"phases=[{stage_text}], stable=epoch {stable_epoch + 1}, "
        f"metric_ready=epoch {metric_epoch + 1}, manual={manual_text}, "
        f"selection=epoch {selection_epoch + 1}"
    )


def is_best_ckpt_selection_eligible(args, epoch):
    """Gate selection and early stopping until all enabled phases are stable."""
    return int(epoch) >= compute_best_ckpt_selection_start_epoch(args)


def describe_best_ckpt_selection_pending(args, epoch):
    selection_epoch = compute_best_ckpt_selection_start_epoch(args)
    pending_stages = [
        f"{name} (epoch {full_epoch + 1})"
        for name, full_epoch in compute_best_ckpt_stage_epochs(args).items()
        if int(epoch) < int(full_epoch)
    ]
    if pending_stages:
        reason = "pending phases: " + ", ".join(pending_stages)
    else:
        reason = "waiting for metric/manual start constraint"
    return f"waiting until epoch {selection_epoch + 1}; {reason}."


def compute_weighted_cls_loss(
    aligned_pos_loss,
    neg_gw_loss,
    mismatched_loss,
    external_loss,
    args,
):
    """Combine four semantically distinct classification buckets.

    Buckets:
      1. aligned positive GW -> its own KN optical sample
      2. neg-GW -> source-matched KN optical sample
      3. positive GW -> another GW's optical sample (mismatched)
      4. positive GW -> external non-KN optical sample

    Each loss may be ``None`` when the corresponding sample type is absent
    from the current batch (for example, no external negatives).
    """
    weights = {
        "aligned": float(
            getattr(
                args, "cls_aligned_pos_weight", getattr(args, "cls_pos_weight", 1.0)
            )
        ),
        "neg_gw": float(
            getattr(args, "cls_neg_gw_weight", getattr(args, "cls_neg_weight", 1.0))
        ),
        "mismatched": float(
            getattr(args, "cls_mismatched_weight", getattr(args, "cls_neg_weight", 1.0))
        ),
        "external": float(
            getattr(
                args,
                "cls_external_neg_weight",
                getattr(args, "cls_extra_neg_weight", 1.0),
            )
        ),
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("Classification bucket weights must be non-negative.")

    parts = []
    if aligned_pos_loss is not None:
        parts.append((weights["aligned"], aligned_pos_loss))
    if neg_gw_loss is not None:
        parts.append((weights["neg_gw"], neg_gw_loss))
    if mismatched_loss is not None:
        parts.append((weights["mismatched"], mismatched_loss))
    if external_loss is not None:
        parts.append((weights["external"], external_loss))

    if not parts:
        return torch.zeros((), dtype=torch.float32)

    denom = sum(weight for weight, _ in parts)
    if denom <= 0.0:
        raise ValueError("At least one classification bucket weight must be > 0.")
    return sum(weight * loss for weight, loss in parts) / denom


def compute_fusion_gallery_nce_loss(scores, positive_mask):
    """Multi-positive retrieval NCE over fusion-head candidate scores."""
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError(
            "scores and positive_mask must have matching shape [n_query, n_candidate]."
        )
    positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
    valid_rows = positive_mask.any(dim=1)
    if not valid_rows.any():
        return scores.sum() * 0.0

    scores_valid = scores[valid_rows]
    pos_valid = positive_mask[valid_rows]
    neg_large = torch.finfo(scores_valid.dtype).min
    pos_scores = scores_valid.masked_fill(~pos_valid, neg_large)
    return (
        torch.logsumexp(scores_valid, dim=1) - torch.logsumexp(pos_scores, dim=1)
    ).mean()


def is_gallery_hard_neg_enabled(args):
    """Return True when retrieval hard-negative mining is enabled."""
    return bool(getattr(args, "gallery_hard_neg_enable", False))


def compute_gallery_hard_neg_weight(args, epoch):
    """Compute ramp-ed weight for the retrieval hard-negative auxiliary loss.

    Returns 0.0 when the hard-mining start epoch (``retrieval phase start +
    gallery_hard_neg_start_after_retrieval_epochs``) has not been reached, then
    ramps linearly to ``gallery_hard_neg_weight`` over
    ``gallery_hard_neg_ramp_epochs``.
    """
    if not is_gallery_hard_neg_enabled(args):
        return 0.0
    if float(getattr(args, "gallery_loss_weight", 0.0)) <= 0.0:
        return 0.0
    retrieval_start = get_curriculum_phase_start(args, "retrieval_intro")
    offset = int(getattr(args, "gallery_hard_neg_start_after_retrieval_epochs", 2))
    hard_start = retrieval_start + offset
    epoch = int(epoch)
    if epoch < hard_start:
        return 0.0
    ramp = int(getattr(args, "gallery_hard_neg_ramp_epochs", 2))
    if ramp <= 0:
        return float(args.gallery_hard_neg_weight)
    progress = min(1.0, (epoch - hard_start + 1) / float(ramp))
    return float(args.gallery_hard_neg_weight) * progress


def gallery_hard_neg_full_activation_reachable(args):
    """Return True when retrieval hard-mining can reach full weight within training."""
    if not is_gallery_hard_neg_enabled(args):
        return False
    if not is_gallery_enabled(args):
        return False
    hard_start = get_curriculum_phase_start(args, "retrieval_intro") + int(
        getattr(args, "gallery_hard_neg_start_after_retrieval_epochs", 2)
    )
    full_epoch = compute_curriculum_full_epoch(
        hard_start,
        getattr(args, "gallery_hard_neg_ramp_epochs", 2),
    )
    return full_epoch < int(getattr(args, "epochs", 0))


def compute_fusion_gallery_hard_nce_loss(scores, positive_mask, topk):
    """Top-k hard-negative NCE loss for fusion mini-gallery retrieval.

    For each query row the loss uses only the true positives and the *top-k*
    highest-scoring negative candidates (hard negatives).  When *topk* reaches
    or exceeds the number of available negatives the result is identical to the
    full-gallery NCE.
    """
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError(
            "scores and positive_mask must have matching shape [n_query, n_candidate]."
        )
    positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)

    valid_rows = positive_mask.any(dim=1) & (~positive_mask).any(dim=1)
    if not valid_rows.any():
        return scores.sum() * 0.0

    scores_v = scores[valid_rows]
    pos_v = positive_mask[valid_rows]
    neg_large = torch.finfo(scores_v.dtype).min

    keep = pos_v.clone()
    neg_scores = scores_v.masked_fill(pos_v, neg_large)

    for i in range(scores_v.size(0)):
        n_neg_i = int((~pos_v[i]).sum().item())
        if n_neg_i == 0:
            continue
        k = min(topk, n_neg_i)
        _, idx = neg_scores[i].topk(k)
        keep[i, idx] = True

    reduced = scores_v.masked_fill(~keep, neg_large)
    pos_only = scores_v.masked_fill(~pos_v, neg_large)
    return (torch.logsumexp(reduced, dim=1) - torch.logsumexp(pos_only, dim=1)).mean()


def compute_fusion_gallery_stratified_hard_nce_loss(
    scores,
    positive_mask,
    kn_negative_mask,
    nonkn_negative_mask,
    *,
    topk,
    kn_fraction,
):
    """Hard-gallery NCE with a fixed KN/non-KN negative composition."""
    if scores.ndim != 2:
        raise ValueError("scores must have shape [n_query, n_candidate].")
    for name, mask in (
        ("positive_mask", positive_mask),
        ("kn_negative_mask", kn_negative_mask),
        ("nonkn_negative_mask", nonkn_negative_mask),
    ):
        if mask.shape != scores.shape:
            raise ValueError(f"{name} must match scores shape.")
    positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
    kn_negative_mask = kn_negative_mask.to(device=scores.device, dtype=torch.bool)
    nonkn_negative_mask = nonkn_negative_mask.to(device=scores.device, dtype=torch.bool)
    if (positive_mask & (kn_negative_mask | nonkn_negative_mask)).any():
        raise ValueError("positive and negative type masks must be disjoint.")
    if (kn_negative_mask & nonkn_negative_mask).any():
        raise ValueError("KN and non-KN negative masks must be disjoint.")

    topk = int(topk)
    if topk < 2:
        raise ValueError("stratified hard gallery topk must be >= 2.")
    n_kn_target, n_nonkn_target = allocate_mixed_negative_counts(
        topk + 1, float(kn_fraction)
    )
    valid_rows = positive_mask.any(dim=1) & (
        kn_negative_mask | nonkn_negative_mask
    ).any(dim=1)
    if not valid_rows.any():
        return scores.sum() * 0.0

    scores_v = scores[valid_rows]
    pos_v = positive_mask[valid_rows]
    kn_v = kn_negative_mask[valid_rows]
    nonkn_v = nonkn_negative_mask[valid_rows]
    neg_large = torch.finfo(scores_v.dtype).min
    keep = pos_v.clone()
    for row in range(scores_v.size(0)):
        selected = []
        shortages = 0
        for mask, requested in (
            (kn_v[row], n_kn_target),
            (nonkn_v[row], n_nonkn_target),
        ):
            candidates = torch.nonzero(mask, as_tuple=False).flatten()
            take = min(int(requested), int(candidates.numel()))
            shortages += int(requested) - take
            if take:
                local = scores_v[row, candidates].topk(take).indices
                selected.append(candidates[local])
        if shortages:
            already = torch.zeros_like(pos_v[row])
            for values in selected:
                already[values] = True
            fallback = (kn_v[row] | nonkn_v[row]) & ~already
            candidates = torch.nonzero(fallback, as_tuple=False).flatten()
            take = min(shortages, int(candidates.numel()))
            if take:
                local = scores_v[row, candidates].topk(take).indices
                selected.append(candidates[local])
        for values in selected:
            keep[row, values] = True

    reduced = scores_v.masked_fill(~keep, neg_large)
    pos_only = scores_v.masked_fill(~pos_v, neg_large)
    return (torch.logsumexp(reduced, dim=1) - torch.logsumexp(pos_only, dim=1)).mean()


def sample_mixed_training_gallery_indices(
    positive_parent_ids,
    *,
    n_external,
    gallery_size,
    kn_fraction,
    max_queries=0,
):
    """Sample one-positive mixed galleries from a batch.

    KN rows are sampled without replacement per query. Multiple selected rows
    may share a non-query parent, preserving the configured ratio when the
    batch contains fewer unique parents than requested KN distractors.
    """
    parent = positive_parent_ids.reshape(-1)
    if parent.numel() == 0:
        raise ValueError("mixed gallery requires positive KN rows.")
    n_kn, n_nonkn = allocate_mixed_negative_counts(gallery_size, kn_fraction)
    if int(n_external) < n_nonkn:
        raise ValueError(
            f"mixed gallery needs {n_nonkn} non-KN rows; batch has {n_external}."
        )

    unique_parent = torch.unique(parent)
    max_queries = int(max_queries)
    if max_queries > 0 and unique_parent.numel() > max_queries:
        unique_parent = unique_parent[
            torch.randperm(unique_parent.numel(), device=parent.device)[:max_queries]
        ]
    query_rows = []
    kn_rows = []
    nonkn_rows = []
    for query_parent in unique_parent:
        positives = torch.nonzero(parent == query_parent, as_tuple=False).flatten()
        target = positives[
            torch.randint(positives.numel(), (1,), device=parent.device).item()
        ]
        eligible = torch.nonzero(parent != query_parent, as_tuple=False).flatten()
        if eligible.numel() < n_kn:
            raise ValueError(
                f"mixed gallery needs {n_kn} KN distractor rows after excluding "
                f"query parent; batch has {eligible.numel()}."
            )
        kn_negative = eligible[
            torch.randperm(eligible.numel(), device=parent.device)[:n_kn]
        ]
        external = torch.randperm(int(n_external), device=parent.device)[:n_nonkn]
        query_rows.append(target)
        kn_rows.append(torch.cat([target.reshape(1), kn_negative]))
        nonkn_rows.append(external)
    return {
        "query_rows": torch.stack(query_rows),
        "kn_rows": torch.stack(kn_rows),
        "nonkn_rows": torch.stack(nonkn_rows),
        "n_kn_negative": n_kn,
        "n_nonkn_negative": n_nonkn,
    }


def build_mixed_training_gallery_batch(
    *,
    positive_parent_ids,
    h_positive,
    z_positive,
    coords_positive,
    positive_dt_days,
    h_external,
    z_external,
    gallery_size,
    kn_fraction,
    max_queries,
    nonkn_empirical_fraction,
    nonkn_window_days,
):
    """Gather query-specific mixed candidate tensors and their type masks."""
    sampled = sample_mixed_training_gallery_indices(
        positive_parent_ids,
        n_external=int(h_external.size(0)),
        gallery_size=gallery_size,
        kn_fraction=kn_fraction,
        max_queries=max_queries,
    )
    query_rows = sampled["query_rows"]
    kn_rows = sampled["kn_rows"]
    nonkn_rows = sampled["nonkn_rows"]
    h_gallery = torch.cat([h_positive[kn_rows], h_external[nonkn_rows]], dim=1)
    z_gallery = torch.cat([z_positive[kn_rows], z_external[nonkn_rows]], dim=1)
    n_query, n_candidate = h_gallery.shape[:2]
    shared_coords = (
        coords_positive[query_rows].unsqueeze(1).expand(n_query, n_candidate, -1)
    )
    kn_dt = positive_dt_days[kn_rows]
    nonkn_dt = torch.stack(
        [
            sample_mixed_external_time_deltas(
                positive_dt_days,
                sampled["n_nonkn_negative"],
                empirical_fraction=nonkn_empirical_fraction,
                window_days=nonkn_window_days,
            )
            for _ in range(n_query)
        ]
    )
    dt_gallery = torch.cat([kn_dt, nonkn_dt], dim=1)

    positive_mask = torch.zeros(
        (n_query, n_candidate), dtype=torch.bool, device=h_gallery.device
    )
    positive_mask[:, 0] = True
    kn_negative_mask = torch.zeros_like(positive_mask)
    kn_negative_mask[:, 1 : 1 + sampled["n_kn_negative"]] = True
    nonkn_negative_mask = torch.zeros_like(positive_mask)
    nonkn_negative_mask[:, 1 + sampled["n_kn_negative"] :] = True
    return {
        "query_rows": query_rows,
        "h_candidates": h_gallery,
        "z_candidates": z_gallery,
        "coords_candidates": shared_coords,
        "dt_days": dt_gallery,
        "positive_mask": positive_mask,
        "kn_negative_mask": kn_negative_mask,
        "nonkn_negative_mask": nonkn_negative_mask,
        "n_kn_negative": sampled["n_kn_negative"],
        "n_nonkn_negative": sampled["n_nonkn_negative"],
    }


def compute_fusion_gallery_metrics(scores, positive_mask, ks=(1, 5)):
    """Compute retrieval metrics for fusion-head mini-gallery scores.

    Rows without any positive candidate are counted as zero for aggregate
    metrics. Conditional metrics report performance over valid rows only.
    """
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError(
            "scores and positive_mask must have matching shape [n_query, n_candidate]."
        )
    positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
    valid_rows = positive_mask.any(dim=1)
    out = {f"fusion_gallery_recall_at_{int(k)}": 0.0 for k in ks}
    out.update({f"fusion_gallery_conditional_recall_at_{int(k)}": 0.0 for k in ks})
    out["fusion_gallery_mrr"] = 0.0
    out["fusion_gallery_conditional_mrr"] = 0.0
    out["fusion_gallery_valid_query_fraction"] = (
        float(valid_rows.float().mean().item()) if scores.size(0) else 0.0
    )
    out["fusion_gallery_candidate_recall_at_topk"] = 0.0
    if scores.size(0) == 0 or scores.size(1) == 0:
        return out

    order = scores.argsort(dim=1, descending=True)
    sorted_pos = positive_mask.gather(1, order)
    n_candidate = int(scores.size(1))
    for k in ks:
        actual_k = min(int(k), n_candidate)
        hit = sorted_pos[:, :actual_k].any(dim=1).float()
        out[f"fusion_gallery_recall_at_{int(k)}"] = float(hit.mean().item())
        if valid_rows.any():
            out[f"fusion_gallery_conditional_recall_at_{int(k)}"] = float(
                hit[valid_rows].mean().item()
            )

    topk = min(max([int(k) for k in ks] or [1]), n_candidate)
    out["fusion_gallery_candidate_recall_at_topk"] = float(
        sorted_pos[:, :topk].any(dim=1).float().mean().item()
    )

    ranks = torch.arange(
        1, n_candidate + 1, device=scores.device, dtype=torch.float32
    ).unsqueeze(0)
    first_pos_rank = (
        torch.where(
            sorted_pos,
            ranks.expand_as(sorted_pos),
            torch.full_like(ranks.expand_as(sorted_pos), float("inf")),
        )
        .min(dim=1)
        .values
    )
    reciprocal_rank = torch.where(
        torch.isfinite(first_pos_rank),
        1.0 / first_pos_rank,
        torch.zeros_like(first_pos_rank),
    )
    out["fusion_gallery_mrr"] = float(reciprocal_rank.mean().item())
    if valid_rows.any():
        out["fusion_gallery_conditional_mrr"] = float(
            reciprocal_rank[valid_rows].mean().item()
        )
    return out


def select_gallery_topk(scores, positive_mask, topk, force_include_positives=False):
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError(
            "scores and positive_mask must have matching shape [n_query, n_candidate]."
        )
    positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
    k = min(max(1, int(topk)), int(scores.size(1)))
    topk_indices = scores.topk(k, dim=1).indices

    if force_include_positives:
        adjusted = topk_indices.clone()
        for row in range(scores.size(0)):
            selected = adjusted[row]
            row_pos = torch.nonzero(positive_mask[row], as_tuple=False).flatten()
            if row_pos.numel() == 0 or positive_mask[row, selected].any():
                continue
            pos_scores = scores[row, row_pos]
            best_pos = row_pos[pos_scores.argmax()]
            selected_pos = positive_mask[row, selected]
            replaceable = torch.nonzero(~selected_pos, as_tuple=False).flatten()
            if replaceable.numel() == 0:
                replace_slot = torch.tensor(
                    k - 1, device=scores.device, dtype=torch.long
                )
            else:
                selected_scores = scores[row, selected[replaceable]]
                replace_slot = replaceable[selected_scores.argmin()]
            adjusted[row, replace_slot] = best_pos
        topk_indices = adjusted

    topk_positive_mask = positive_mask.gather(1, topk_indices)
    candidate_recall_mask = topk_positive_mask.any(dim=1)
    return topk_indices, topk_positive_mask, candidate_recall_mask


def _row_standardize_scores(scores):
    centered = scores - scores.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return centered / scale


def compute_residual_rerank_loss(s_itc, s_fusion, positive_mask, lambda_=0.5):
    """NCE loss over row-normalized ITC and fusion scores for reranking."""
    if s_itc.shape != s_fusion.shape or positive_mask.shape != s_itc.shape:
        raise ValueError(
            "s_itc, s_fusion, and positive_mask must have matching shape [n_query, n_candidate]."
        )
    positive_mask = positive_mask.to(device=s_itc.device, dtype=torch.bool)
    valid_rows = positive_mask.any(dim=1)
    if not valid_rows.any():
        return (s_itc.sum() + s_fusion.sum()) * 0.0
    combined = _row_standardize_scores(s_itc[valid_rows]) + float(
        lambda_
    ) * _row_standardize_scores(s_fusion[valid_rows])
    return compute_fusion_gallery_nce_loss(combined, positive_mask[valid_rows])


def build_gallery_time_delta_matrix(
    query_event_time_mjd,
    candidate_event_time_mjd,
    *,
    candidate_parent_event_time_mjd=None,
    positive_mask=None,
    distractor_time_mode="actual",
    distractor_time_window_days=30.0,
):
    """Build per-query/per-candidate dt for fusion mini-gallery scoring.

    In ``actual`` mode this is the raw candidate-minus-query MJD difference.
    In ``parent_relative`` mode each candidate uses its detection delay relative
    to its own parent GW, re-anchored identically for every query.
    In ``synthetic_after_gw`` mode, matching positives keep their real dt while
    every non-matching distractor is assigned a synthetic first-detection offset
    sampled uniformly from [0, distractor_time_window_days].
    """
    if query_event_time_mjd is None or candidate_event_time_mjd is None:
        return None

    query = query_event_time_mjd.to(torch.float32).reshape(-1, 1)
    candidate = candidate_event_time_mjd.to(
        device=query.device, dtype=torch.float32
    ).reshape(1, -1)
    mode = str(distractor_time_mode).strip().lower()
    if mode in {"parent_relative", "candidate_relative", "empirical"}:
        parent_dt = compute_parent_relative_time_delta_days(
            candidate.reshape(-1), candidate_parent_event_time_mjd
        )
        return parent_dt.reshape(1, -1).expand(query.size(0), -1)

    dt = candidate - query
    dt = torch.where(torch.isfinite(dt), dt, torch.zeros_like(dt))

    if mode in {"actual", "real", "none"}:
        return dt
    if mode not in {"synthetic_after_gw", "synthetic", "after_gw"}:
        raise ValueError(
            f"Unsupported gallery_distractor_time_mode='{distractor_time_mode}'. "
            "Expected 'actual' or 'synthetic_after_gw'."
        )

    window_days = float(distractor_time_window_days)
    if window_days <= 0:
        raise ValueError(
            "gallery_distractor_time_window_days must be > 0 for synthetic_after_gw mode."
        )

    if positive_mask is None:
        positive_mask_t = torch.zeros_like(dt, dtype=torch.bool)
    else:
        positive_mask_t = positive_mask.to(device=query.device, dtype=torch.bool)
        if tuple(positive_mask_t.shape) != tuple(dt.shape):
            raise ValueError(
                "positive_mask must have shape "
                f"{tuple(dt.shape)}; got {tuple(positive_mask_t.shape)}."
            )

    synthetic_dt = torch.rand(dt.shape, device=dt.device, dtype=dt.dtype) * window_days
    return torch.where(positive_mask_t, dt, synthetic_dt)


def compute_fusion_gallery_score_matrix(
    model,
    *,
    g,
    H_gw,
    gw_s,
    gw_m,
    h_candidates,
    z_candidates,
    opt_coords_candidates,
    dt_days_matrix=None,
    need_cred_level=False,
    chunk_size=8192,
):
    """Score every GW query against a candidate optical mini-gallery with fusion logits.

    Both query and candidate dimensions are chunked so that each block processes
    at most ≈ chunk_size flattened pairs, bounding peak GPU memory.
    """
    import math

    n_query = int(g.size(0))
    query_specific = h_candidates.ndim == 4
    if query_specific:
        if h_candidates.size(0) != n_query:
            raise ValueError("query-specific candidates must match the query count.")
        n_candidate = int(h_candidates.size(1))
        if z_candidates.shape[:2] != (n_query, n_candidate):
            raise ValueError("query-specific z_candidates must have shape [Q, C, D].")
        if opt_coords_candidates.shape[:2] != (n_query, n_candidate):
            raise ValueError(
                "query-specific opt_coords_candidates must have shape [Q, C, 2]."
            )
    else:
        n_candidate = int(h_candidates.size(0))
    if n_candidate == 0:
        return torch.empty((n_query, 0), dtype=g.dtype, device=g.device)

    max_pairs = max(1, int(chunk_size))
    # Balance query and candidate chunk sizes proportionally
    q_chunk = min(
        n_query,
        max(1, int(math.sqrt(max_pairs * max(1, n_query / max(1, n_candidate))))),
    )
    c_chunk = max(1, max_pairs // q_chunk)

    scores = torch.empty((n_query, n_candidate), dtype=g.dtype, device=g.device)

    for q_start in range(0, n_query, q_chunk):
        q_end = min(q_start + q_chunk, n_query)
        n_q = q_end - q_start

        g_q = g[q_start:q_end]
        H_q = None if H_gw is None else H_gw[q_start:q_end]
        gw_s_q = None if gw_s is None else gw_s[q_start:q_end]
        gw_m_q = None if (gw_m is None or not need_cred_level) else gw_m[q_start:q_end]
        dt_rows = None if dt_days_matrix is None else dt_days_matrix[q_start:q_end, :]

        for c_start in range(0, n_candidate, c_chunk):
            c_end = min(c_start + c_chunk, n_candidate)
            n_c = c_end - c_start

            g_pair = g_q.unsqueeze(1).expand(n_q, n_c, -1).reshape(n_q * n_c, -1)
            if query_specific:
                h_pair = h_candidates[q_start:q_end, c_start:c_end].reshape(
                    n_q * n_c, h_candidates.size(2), h_candidates.size(3)
                )
                z_pair = z_candidates[q_start:q_end, c_start:c_end].reshape(
                    n_q * n_c, -1
                )
                coords_pair = opt_coords_candidates[
                    q_start:q_end, c_start:c_end
                ].reshape(n_q * n_c, -1)
            else:
                h_pair = (
                    h_candidates[c_start:c_end]
                    .unsqueeze(0)
                    .expand(n_q, n_c, -1, -1)
                    .reshape(n_q * n_c, h_candidates.size(1), h_candidates.size(2))
                )
                z_pair = (
                    z_candidates[c_start:c_end]
                    .unsqueeze(0)
                    .expand(n_q, n_c, -1)
                    .reshape(n_q * n_c, -1)
                )
                coords_pair = (
                    opt_coords_candidates[c_start:c_end]
                    .unsqueeze(0)
                    .expand(n_q, n_c, -1)
                    .reshape(n_q * n_c, -1)
                )
            H_pair = (
                None
                if H_q is None
                else H_q.unsqueeze(1)
                .expand(n_q, n_c, -1, -1)
                .reshape(n_q * n_c, H_q.size(1), H_q.size(2))
            )
            gw_s_pair = (
                None
                if gw_s_q is None
                else gw_s_q.unsqueeze(1).expand(n_q, n_c, -1).reshape(n_q * n_c, -1)
            )
            gw_m_pair = None
            if need_cred_level and gw_m_q is not None:
                gw_m_pair = (
                    gw_m_q.unsqueeze(1)
                    .expand(n_q, n_c, -1, -1)
                    .reshape(n_q * n_c, gw_m_q.size(1), gw_m_q.size(2))
                )
            dt_pair = None
            if dt_rows is not None:
                dt_pair = dt_rows[:, c_start:c_end].reshape(n_q * n_c)
            cred_pair = (
                compute_credible_level(gw_m_pair, coords_pair)
                if gw_m_pair is not None
                else None
            )

            logits = model.fusion_logits(
                g_pair,
                h_pair,
                z_l=z_pair,
                H_gw=H_pair,
                cred_level=cred_pair,
                gw_s=gw_s_pair,
                gw_m=gw_m_pair,
                opt_coords=coords_pair,
                dt_days=dt_pair,
            )
            scores[q_start:q_end, c_start:c_end] = (
                logits[:, 1] - logits[:, 0]
            ).reshape(n_q, n_c)

    return scores


def build_pairwise_time_delta_matrix(query_event_time_mjd, candidate_event_time_mjd):
    if query_event_time_mjd is None or candidate_event_time_mjd is None:
        return None
    query = query_event_time_mjd.to(torch.float32).reshape(-1, 1)
    candidate = candidate_event_time_mjd.to(
        device=query.device, dtype=torch.float32
    ).reshape(1, -1)
    dt = candidate - query
    return torch.where(torch.isfinite(dt), dt, torch.zeros_like(dt))


def apply_temperature_schedule(model, args, epoch):
    if args.temp_schedule == "learned":
        return None
    if args.temp_schedule == "fixed":
        target_temp = args.temp_init
    else:
        progress = epoch / float(max(1, args.epochs - 1))
        target_temp = args.temp_final + 0.5 * (args.temp_init - args.temp_final) * (
            1.0 + math.cos(math.pi * progress)
        )
    target_temp = max(args.temp_min, min(args.temp_max, target_temp))
    model.log_temp.data.fill_(math.log(target_temp))
    return target_temp


def clamp_temperature(model, args):
    if args.temp_schedule != "learned":
        return
    min_log = math.log(args.temp_min)
    max_log = math.log(args.temp_max)
    model.log_temp.data.clamp_(min_log, max_log)


def compute_ckpt_selection_score(val_metrics, metric_name):
    """Compute best-checkpoint selection score from in-domain validation metrics."""
    cls_m = val_metrics.get("classification", {})
    ret_m = val_metrics.get("retrieval", {})

    if metric_name == "auprc":
        return float(cls_m.get("auprc", 0.0))
    if metric_name == "auroc":
        return float(cls_m.get("auroc", 0.0))
    if metric_name == "f1_optimal":
        return float(cls_m.get("f1_optimal", 0.0))
    if metric_name == "acc_total":
        return float(cls_m.get("acc_total", 0.0))
    if metric_name == "cls_composite_auprc_auroc":
        auprc = float(cls_m.get("auprc", 0.0))
        auroc = float(cls_m.get("auroc", 0.0))
        return 0.5 * auprc + 0.5 * auroc
    if metric_name == "g2o_recall_at_1":
        return float(ret_m.get("g2o_recall_at_1", 0.0))
    if metric_name == "g2o_recall_at_5":
        return float(ret_m.get("g2o_recall_at_5", 0.0))
    if metric_name == "g2o_mrr":
        return float(ret_m.get("g2o_mrr", 0.0))
    if metric_name == "fusion_gallery_recall_at_1":
        return float(ret_m.get("fusion_gallery_recall_at_1", 0.0))
    if metric_name == "fusion_gallery_recall_at_5":
        return float(ret_m.get("fusion_gallery_recall_at_5", 0.0))
    if metric_name == "fusion_gallery_mrr":
        return float(ret_m.get("fusion_gallery_mrr", 0.0))
    if metric_name in {
        "hard_gallery_macro_retrieval_score",
        "mixed_gallery_macro_retrieval_score",
    }:
        return float(val_metrics.get("hard_gallery", {}).get("selection_score", 0.0))

    raise ValueError(f"Unsupported best_ckpt_metric: {metric_name}")


def is_checkpoint_score_improved(current_score, previous_best, min_delta):
    return previous_best is None or float(current_score) > (
        float(previous_best) + float(min_delta)
    )


NEG_GW_STRATA_KEYS = ("bns_type1", "bns_type2", "nsbh_type1", "nsbh_type2")


def resolve_neg_gw_guardrail(args, val_metrics):
    """Resolve the stratified negative-GW recall guardrail.

    The guardrail is only meaningful when the validation set contains
    negative-GW pairs. All four ``source_type x neg_type`` strata must be
    populated and every stratum recall must be at least
    ``neg_gw_guardrail_recall``.
    """
    enabled = bool(getattr(args, "neg_gw_guardrail_enable", False))
    threshold = float(getattr(args, "neg_gw_guardrail_recall", 0.90))
    strata = (val_metrics or {}).get("neg_gw_strata", {})

    recalls = []
    counts = []
    for key in NEG_GW_STRATA_KEYS:
        item = strata.get(key)
        if item is None:
            recalls.append(0.0)
            counts.append(0)
        else:
            recalls.append(float(item.get("recall", 0.0)))
            counts.append(int(item.get("count", 0)))

    min_recall = min(recalls) if recalls else 0.0
    populated = len(recalls) == len(NEG_GW_STRATA_KEYS) and all(
        count > 0 for count in counts
    )
    met = (not enabled) or (populated and min_recall + 1.0e-9 >= threshold)

    reason = None
    if enabled and not met:
        if not populated:
            reason = "missing or empty neg-GW validation stratum"
        else:
            reason = f"min recall {min_recall:.4f} < threshold {threshold:.4f}"

    return {
        "enabled": enabled,
        "threshold": threshold,
        "min_recall": min_recall,
        "met": bool(met),
        "reason": reason,
    }


def evaluate(
    model,
    val_loader,
    device,
    args,
    epoch,
    amp_dtype=torch.float32,
    gw_event_time_mjd_table=None,
    gw_source_type_table=None,
    gw_neg_type_table=None,
):
    from metrics import (
        compute_retrieval_metrics,
        compute_classification_metrics,
        compute_embedding_metrics,
    )

    model.eval()
    val_sampler = getattr(val_loader, "batch_sampler", None)
    val_has_neg_gw_pairs = bool(getattr(val_sampler, "n_neg_per_batch", 0) > 0)
    has_negatives = args.neg_data_path is not None
    cls_branch_enabled = is_cls_branch_enabled(args)
    ref_time_cache = None
    cls_weight = compute_cls_weight(args, epoch)
    cls_active = cls_weight > 0.0
    itc_weight = compute_itc_weight(args, epoch)
    gallery_loss_weight = compute_gallery_loss_weight(args, epoch)
    cls_metrics_enabled = should_compute_cls_metrics(args)
    fusion_gallery_metrics_enabled = should_compute_fusion_gallery_metrics(args, epoch)
    gallery_loss_active = is_retrieval_active(args, epoch)

    val_total = 0.0
    val_itc = 0.0
    val_cls = 0.0
    val_gallery = 0.0
    val_gallery_hard = 0.0
    val_itc_acc = 0.0
    val_pos_acc = 0.0
    val_neg_gw_acc = 0.0
    val_hard_acc = 0.0
    val_neg_acc = 0.0
    val_total_acc = 0.0
    val_batches = 0

    # Accumulators for new metrics (computed globally after all batches)
    all_cls_probs = []  # predicted P(match) for classification metrics
    all_cls_labels = []  # ground-truth labels for classification metrics
    all_cls_sources = []  # source tag per sample ('pos', 'hard', 'neg')
    all_feat_g = []  # L2-normalized GW embeddings for embedding metrics
    all_feat_o = []  # L2-normalized optical embeddings
    all_gw_indices = []  # GW event indices
    retrieval_metrics_accum = []  # per-batch retrieval metric dicts
    fusion_gallery_metrics_accum = []
    dt_stats = {
        "pos": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "hard": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "extra": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
    }
    validation_dt_generator = torch.Generator(device=device)
    validation_dt_generator.manual_seed(int(args.seed) + 104729)
    neg_gw_probabilities = {
        ("bns", 1): [],
        ("bns", 2): [],
        ("nsbh", 1): [],
        ("nsbh", 2): [],
    }
    with torch.no_grad():
        for batch_data in val_loader:
            pair_is_neg_gw = None
            if val_has_neg_gw_pairs:
                pair_is_neg_gw = batch_data[-1]
                batch_data = batch_data[:-1]
            _opt_zero_time_mjd_base = None
            _neg_zero_time_mjd_base = None
            _neg_zero_time_mjd_cls_base = None
            _opt_first_detection_mjd = None
            if has_negatives:
                if len(batch_data) >= 17:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        neg_t,
                        neg_v,
                        neg_mask,
                        neg_err,
                        neg_coords,
                        _opt_zero_time_mjd_base,
                        _neg_zero_time_mjd_base,
                        _neg_zero_time_mjd_cls_base,
                        _opt_first_detection_mjd,
                    ) = batch_data[:17]
                elif len(batch_data) >= 16:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        neg_t,
                        neg_v,
                        neg_mask,
                        neg_err,
                        neg_coords,
                        _opt_zero_time_mjd_base,
                        _neg_zero_time_mjd_base,
                        _neg_zero_time_mjd_cls_base,
                    ) = batch_data[:16]
                elif len(batch_data) >= 15:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        neg_t,
                        neg_v,
                        neg_mask,
                        neg_err,
                        neg_coords,
                        _opt_zero_time_mjd_base,
                        _neg_zero_time_mjd_base,
                    ) = batch_data[:15]
                else:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        neg_t,
                        neg_v,
                        neg_mask,
                        neg_err,
                        neg_coords,
                    ) = batch_data[:13]
            else:
                if len(batch_data) >= 10:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        _opt_zero_time_mjd_base,
                        _opt_first_detection_mjd,
                    ) = batch_data[:10]
                elif len(batch_data) >= 9:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        _opt_zero_time_mjd_base,
                    ) = batch_data[:9]
                else:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                    ) = batch_data

            gw_s = gw_s.to(device, non_blocking=True)
            gw_m = gw_m.to(device, non_blocking=True)
            opt_t = opt_t.to(device, non_blocking=True)
            opt_v = opt_v.to(device, non_blocking=True)
            opt_mask = opt_mask.to(device, non_blocking=True)
            opt_err = opt_err.to(device, non_blocking=True)
            opt_coords = opt_coords.to(device, non_blocking=True)
            gw_indices = gw_indices.to(device, non_blocking=True).long()
            if pair_is_neg_gw is None:
                pair_is_neg_gw = torch.zeros(
                    gw_indices.shape[0], device=device, dtype=torch.bool
                )
            else:
                pair_is_neg_gw = pair_is_neg_gw.to(device, non_blocking=True).bool()
            itc_pair_mask = build_itc_pair_mask(
                gw_indices,
                pair_is_neg_gw,
                min_pairs_per_gw=args.min_lc_per_gw,
            )
            if not bool(itc_pair_mask.any()):
                raise ValueError(
                    "Validation batch has no multi-positive GW event for ITC."
                )
            batch_event_time_mjd = None
            if gw_event_time_mjd_table is not None:
                batch_event_time_mjd = gw_event_time_mjd_table[gw_indices]
            batch_source_types = None
            if gw_source_type_table is not None:
                batch_source_types = gw_source_type_table[gw_indices]

            if has_negatives:
                neg_t = neg_t.to(device, non_blocking=True)
                neg_v = neg_v.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)
                if _neg_zero_time_mjd_base is not None:
                    _neg_zero_time_mjd_base = _neg_zero_time_mjd_base.to(
                        device, non_blocking=True
                    )
                if _neg_zero_time_mjd_cls_base is not None:
                    _neg_zero_time_mjd_cls_base = _neg_zero_time_mjd_cls_base.to(
                        device, non_blocking=True
                    )
            if _opt_first_detection_mjd is not None:
                _opt_first_detection_mjd = _opt_first_detection_mjd.to(
                    device, non_blocking=True
                )

            batch_size = gw_s.size(0)
            if (
                ref_time_cache is None
                or ref_time_cache.shape[0] != batch_size
                or ref_time_cache.dtype != opt_t.dtype
            ):
                ref_time_cache = build_ref_time(
                    batch_size,
                    args.n_ref,
                    args.ref_start,
                    args.ref_end,
                    device,
                    opt_t.dtype,
                )
            opt_ref_t = ref_time_cache
            need_cred_level = _model_requires_cred_level(model)

            with autocast(
                device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                g, z_l, h_l, H_gw = model.encode(
                    gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
                )
                itc_g = g[itc_pair_mask]
                itc_z_l = z_l[itc_pair_mask]
                itc_gw_indices = gw_indices[itc_pair_mask]
                itc_gw_event_time = (
                    batch_event_time_mjd[itc_pair_mask]
                    if batch_event_time_mjd is not None
                    else None
                )
                itc_opt_first_detection = (
                    _opt_first_detection_mjd[itc_pair_mask]
                    if _opt_first_detection_mjd is not None
                    else None
                )
                if args.itc_loss_type == "supcon":
                    itc_loss, sim_g2o, sim_g2o_for_loss = model.compute_supcon_loss(
                        itc_g,
                        itc_z_l,
                        itc_gw_indices,
                        margin=args.supcon_margin,
                        gw_event_time_mjd=itc_gw_event_time,
                        opt_event_time_mjd=resolve_optical_candidate_time_mjd(
                            itc_opt_first_detection, itc_gw_event_time
                        ),
                        extra_neg_z=None,
                        extra_neg_opt_event_time_mjd=None,
                    )
                else:
                    itc_loss, sim_g2o, sim_g2o_for_loss = model.compute_itc_loss(
                        itc_g,
                        itc_z_l,
                        itc_gw_indices,
                        gw_event_time_mjd=itc_gw_event_time,
                        opt_event_time_mjd=resolve_optical_candidate_time_mjd(
                            itc_opt_first_detection, itc_gw_event_time
                        ),
                        extra_neg_z=None,
                        extra_neg_opt_event_time_mjd=None,
                    )

                feat_g, feat_o = model.get_contrastive_embeddings(itc_g, itc_z_l)
                pos_mask = ~pair_is_neg_gw
                neg_gw_mask = pair_is_neg_gw
                labels_pos = pos_mask.to(dtype=torch.long)
                labels_neg = torch.zeros(batch_size, device=device, dtype=torch.long)
                gallery_loss = torch.zeros((), device=device)
                gallery_hard_loss = torch.zeros((), device=device)
                gallery_hard_weight = 0.0
                cls_logits_available = False
                cls_hard_logits_available = False
                cls_extra_neg_logits_available = False

                if cls_branch_enabled:
                    cls_logits_available = True
                    cls_hard_logits_available = True
                    cls_extra_neg_logits_available = bool(has_negatives)
                    # Compute per-pair credible level for dual fusion
                    cred_level = (
                        compute_credible_level(gw_m, opt_coords)
                        if need_cred_level
                        else None
                    )
                    if (
                        _opt_first_detection_mjd is not None
                        and batch_event_time_mjd is not None
                    ):
                        dt_pos = compute_time_delta_days(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        )
                    else:
                        dt_pos = torch.zeros(
                            (batch_size,), device=device, dtype=torch.float32
                        )
                    _accumulate_time_delta_stats(dt_stats, "pos", dt_pos[pos_mask])
                    logits_pos = model.fusion_logits(
                        g,
                        h_l,
                        z_l=z_l,
                        H_gw=H_gw,
                        cred_level=cred_level,
                        gw_s=gw_s,
                        gw_m=gw_m,
                        opt_coords=opt_coords,
                        dt_days=dt_pos,
                    )
                    if pos_mask.any():
                        aligned_pos_loss = model.cls_criterion(
                            logits_pos[pos_mask],
                            torch.ones(
                                int(pos_mask.sum()), device=device, dtype=torch.long
                            ),
                        )
                    else:
                        aligned_pos_loss = None
                    if neg_gw_mask.any():
                        neg_gw_loss = model.cls_criterion(
                            logits_pos[neg_gw_mask],
                            torch.zeros(
                                int(neg_gw_mask.sum()), device=device, dtype=torch.long
                            ),
                        )
                    else:
                        neg_gw_loss = None

                    mis_idx = sample_mismatched_negatives(
                        batch_size,
                        device,
                        samples_per_gw=args.samples_per_gw,
                        gw_indices=gw_indices,
                        source_types=batch_source_types,
                        eligible_candidate_mask=itc_pair_mask,
                    )

                    if mis_idx is None:
                        hard_loss = torch.zeros((), device=device)
                        logits_hard = torch.zeros(
                            (batch_size, 2), device=device, dtype=g.dtype
                        )
                    else:
                        h_l_mis = h_l[mis_idx].clone()
                        z_l_mis = z_l[mis_idx].clone()
                        coords_mis = opt_coords[mis_idx].clone()
                        dt_mis = build_mismatched_classification_time_deltas(
                            mode=args.cls_distractor_time_mode,
                            batch_size=batch_size,
                            device=device,
                            window_days=args.mis_neg_dt_window_days,
                            candidate_indices=mis_idx,
                            opt_first_detection_mjd=_opt_first_detection_mjd,
                            parent_gw_event_time_mjd=batch_event_time_mjd,
                        )
                        _accumulate_time_delta_stats(dt_stats, "hard", dt_mis)
                        cred_level_mis = (
                            compute_credible_level(gw_m, coords_mis)
                            if need_cred_level
                            else None
                        )
                        logits_hard = model.fusion_logits(
                            g,
                            h_l_mis,
                            z_l=z_l_mis,
                            H_gw=H_gw,
                            cred_level=cred_level_mis,
                            gw_s=gw_s,
                            gw_m=gw_m,
                            opt_coords=coords_mis,
                            dt_days=dt_mis,
                        )
                        hard_loss = model.cls_criterion(logits_hard, labels_neg)

                    gallery_extra_h_l = None
                    gallery_extra_z_l = None
                    gallery_extra_coords = None
                    gallery_extra_event_time = None
                    if has_negatives:
                        neg_zero_time_mjd_for_cls = _neg_zero_time_mjd_cls_base
                        if neg_zero_time_mjd_for_cls is None:
                            neg_zero_time_mjd_for_cls = _neg_zero_time_mjd_base
                        z_l_neg, h_l_neg = model.encode_optical(
                            neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err
                        )
                        cred_level_neg = (
                            compute_credible_level(gw_m, neg_coords)
                            if need_cred_level
                            else None
                        )
                        neg_event_time = (
                            neg_zero_time_mjd_for_cls.to(
                                device=device, dtype=torch.float32
                            )
                            if neg_zero_time_mjd_for_cls is not None
                            else None
                        )
                        dt_neg = build_external_classification_time_deltas(
                            mode=args.cls_distractor_time_mode,
                            positive_dt_pool=dt_pos[itc_pair_mask],
                            n_samples=batch_size,
                            empirical_fraction=args.cls_external_empirical_dt_fraction,
                            window_days=args.mis_neg_dt_window_days,
                            negative_event_time_mjd=neg_event_time,
                            query_event_time_mjd=batch_event_time_mjd,
                            device=device,
                            generator=validation_dt_generator,
                        )
                        logits_neg = model.fusion_logits(
                            g,
                            h_l_neg,
                            z_l=z_l_neg,
                            H_gw=H_gw,
                            cred_level=cred_level_neg,
                            gw_s=gw_s,
                            gw_m=gw_m,
                            opt_coords=neg_coords,
                            dt_days=dt_neg,
                        )
                        gallery_extra_h_l = h_l_neg
                        gallery_extra_z_l = z_l_neg
                        gallery_extra_coords = neg_coords
                        gallery_extra_event_time = neg_event_time
                        if batch_event_time_mjd is not None:
                            _accumulate_time_delta_stats(dt_stats, "extra", dt_neg)
                        neg_loss = model.cls_criterion(logits_neg, labels_neg)
                    else:
                        neg_loss = None

                    cls_loss = compute_weighted_cls_loss(
                        aligned_pos_loss, neg_gw_loss, hard_loss, neg_loss, args
                    )
                    gallery_loss = torch.zeros((), device=device)
                    gallery_hard_loss = torch.zeros((), device=device)
                    gallery_hard_weight = 0.0
                    total_loss = itc_weight * itc_loss + cls_weight * cls_loss
                else:
                    logits_pos = torch.zeros(
                        (batch_size, 2), device=device, dtype=g.dtype
                    )
                    logits_hard = torch.zeros(
                        (batch_size, 2), device=device, dtype=g.dtype
                    )
                    logits_neg = (
                        torch.zeros((batch_size, 2), device=device, dtype=g.dtype)
                        if has_negatives
                        else None
                    )
                    aligned_pos_loss = torch.zeros((), device=device)
                    neg_gw_loss = None
                    hard_loss = torch.zeros((), device=device)
                    neg_loss = torch.zeros((), device=device) if has_negatives else None
                    cls_loss = torch.zeros((), device=device)
                    gallery_loss = torch.zeros((), device=device)
                    gallery_hard_loss = torch.zeros((), device=device)
                    gallery_hard_weight = 0.0
                    need_extra_optical_features = has_negatives and (
                        gallery_loss_active
                        or fusion_gallery_metrics_enabled
                        or cls_metrics_enabled
                    )
                    if need_extra_optical_features:
                        z_l_neg, h_l_neg = model.encode_optical(
                            neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err
                        )
                        gallery_extra_h_l = h_l_neg
                        gallery_extra_z_l = z_l_neg
                        gallery_extra_coords = neg_coords
                        neg_event_time = (
                            _neg_zero_time_mjd_cls_base.to(
                                device=device, dtype=torch.float32
                            )
                            if _neg_zero_time_mjd_cls_base is not None
                            else (
                                _neg_zero_time_mjd_base.to(
                                    device=device, dtype=torch.float32
                                )
                                if _neg_zero_time_mjd_base is not None
                                else None
                            )
                        )
                        gallery_extra_event_time = neg_event_time
                    else:
                        gallery_extra_h_l = None
                        gallery_extra_z_l = None
                        gallery_extra_coords = None
                        gallery_extra_event_time = None
                    total_loss = itc_weight * itc_loss

                # --- Shared gallery computation (independent of CLS branch) ---
                compute_gallery_this_epoch = (
                    gallery_loss_active or fusion_gallery_metrics_enabled
                )
                if compute_gallery_this_epoch and itc_pair_mask.any():
                    gallery_query_g = g[itc_pair_mask]
                    gallery_query_H = H_gw[itc_pair_mask] if H_gw is not None else None
                    gallery_query_gw_s = gw_s[itc_pair_mask]
                    gallery_query_gw_m = gw_m[itc_pair_mask]
                    gallery_query_event_time = (
                        batch_event_time_mjd[itc_pair_mask]
                        if batch_event_time_mjd is not None
                        else None
                    )
                    h_candidates = [h_l[itc_pair_mask]]
                    z_candidates = [z_l[itc_pair_mask]]
                    coords_candidates = [opt_coords[itc_pair_mask]]
                    candidate_gw_indices = [itc_gw_indices]
                    candidate_time_blocks = []
                    candidate_parent_time_blocks = []
                    if batch_event_time_mjd is not None:
                        pos_gallery_time = resolve_optical_candidate_time_mjd(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        )[itc_pair_mask].to(device=device, dtype=torch.float32)
                        candidate_time_blocks.append(pos_gallery_time)
                        candidate_parent_time_blocks.append(gallery_query_event_time)
                    if (
                        has_negatives
                        and bool(args.gallery_include_extra_negatives)
                        and gallery_extra_h_l is not None
                        and gallery_extra_z_l is not None
                        and gallery_extra_coords is not None
                    ):
                        h_candidates.append(gallery_extra_h_l)
                        z_candidates.append(gallery_extra_z_l)
                        coords_candidates.append(gallery_extra_coords)
                        candidate_gw_indices.append(
                            torch.full(
                                (batch_size,), -1, device=device, dtype=gw_indices.dtype
                            )
                        )
                        if batch_event_time_mjd is not None:
                            if gallery_extra_event_time is None:
                                candidate_time_blocks.append(
                                    torch.full(
                                        (batch_size,),
                                        float("nan"),
                                        device=device,
                                        dtype=torch.float32,
                                    )
                                )
                            else:
                                candidate_time_blocks.append(
                                    gallery_extra_event_time.to(
                                        device=device, dtype=torch.float32
                                    )
                                )
                            candidate_parent_time_blocks.append(
                                torch.full(
                                    (batch_size,),
                                    float("nan"),
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )
                    h_gallery = torch.cat(h_candidates, dim=0)
                    z_gallery = torch.cat(z_candidates, dim=0)
                    coords_gallery = torch.cat(coords_candidates, dim=0)
                    gallery_gw_indices = torch.cat(candidate_gw_indices, dim=0)
                    gallery_positive_mask = itc_gw_indices.unsqueeze(
                        1
                    ) == gallery_gw_indices.unsqueeze(0)
                    gallery_candidate_time = (
                        torch.cat(candidate_time_blocks, dim=0)
                        if candidate_time_blocks
                        else None
                    )
                    gallery_candidate_parent_time = (
                        torch.cat(candidate_parent_time_blocks, dim=0)
                        if candidate_parent_time_blocks
                        else None
                    )
                    dt_gallery = build_gallery_time_delta_matrix(
                        gallery_query_event_time,
                        gallery_candidate_time,
                        candidate_parent_event_time_mjd=gallery_candidate_parent_time,
                        positive_mask=gallery_positive_mask,
                        distractor_time_mode=args.gallery_distractor_time_mode,
                        distractor_time_window_days=args.gallery_distractor_time_window_days,
                    )
                    gallery_kn_negative_mask = None
                    gallery_nonkn_negative_mask = None
                    if args.gallery_candidate_mode == "mixed_kn_nonkn":
                        if (
                            gallery_extra_h_l is None
                            or gallery_extra_z_l is None
                            or gallery_query_event_time is None
                            or not candidate_time_blocks
                        ):
                            raise ValueError(
                                "mixed training gallery requires external non-KN "
                                "features and KN first-detection/event times."
                            )
                        positive_dt_days = compute_parent_relative_time_delta_days(
                            candidate_time_blocks[0], gallery_query_event_time
                        )
                        mixed = build_mixed_training_gallery_batch(
                            positive_parent_ids=itc_gw_indices,
                            h_positive=h_l[itc_pair_mask],
                            z_positive=z_l[itc_pair_mask],
                            coords_positive=opt_coords[itc_pair_mask],
                            positive_dt_days=positive_dt_days,
                            h_external=gallery_extra_h_l,
                            z_external=gallery_extra_z_l,
                            gallery_size=args.gallery_training_size,
                            kn_fraction=args.gallery_kn_distractor_fraction,
                            max_queries=args.max_gallery_queries,
                            nonkn_empirical_fraction=(
                                args.gallery_nonkn_empirical_fraction
                            ),
                            nonkn_window_days=args.gallery_distractor_time_window_days,
                        )
                        query_rows = mixed["query_rows"]
                        gallery_query_g = gallery_query_g[query_rows]
                        gallery_query_H = (
                            gallery_query_H[query_rows]
                            if gallery_query_H is not None
                            else None
                        )
                        gallery_query_gw_s = gallery_query_gw_s[query_rows]
                        gallery_query_gw_m = gallery_query_gw_m[query_rows]
                        gallery_query_event_time = gallery_query_event_time[query_rows]
                        h_gallery = mixed["h_candidates"]
                        z_gallery = mixed["z_candidates"]
                        coords_gallery = mixed["coords_candidates"]
                        dt_gallery = mixed["dt_days"]
                        gallery_positive_mask = mixed["positive_mask"]
                        gallery_kn_negative_mask = mixed["kn_negative_mask"]
                        gallery_nonkn_negative_mask = mixed["nonkn_negative_mask"]
                    gallery_scores = compute_fusion_gallery_score_matrix(
                        model,
                        g=gallery_query_g,
                        H_gw=gallery_query_H,
                        gw_s=gallery_query_gw_s,
                        gw_m=gallery_query_gw_m,
                        h_candidates=h_gallery,
                        z_candidates=z_gallery,
                        opt_coords_candidates=coords_gallery,
                        dt_days_matrix=dt_gallery,
                        need_cred_level=need_cred_level,
                        chunk_size=args.gallery_score_chunk_size,
                    )
                    gallery_loss = compute_fusion_gallery_nce_loss(
                        gallery_scores, gallery_positive_mask
                    )
                    if fusion_gallery_metrics_enabled:
                        fusion_gallery_metrics_accum.append(
                            compute_fusion_gallery_metrics(
                                gallery_scores.detach(),
                                gallery_positive_mask,
                                ks=(1, 5),
                            )
                        )
                    if gallery_loss_active and gallery_loss_weight > 0.0:
                        total_loss = total_loss + gallery_loss_weight * gallery_loss
                    gallery_hard_weight = compute_gallery_hard_neg_weight(args, epoch)
                    if gallery_hard_weight > 0.0:
                        if gallery_kn_negative_mask is not None:
                            gallery_hard_loss = (
                                compute_fusion_gallery_stratified_hard_nce_loss(
                                    gallery_scores,
                                    gallery_positive_mask,
                                    gallery_kn_negative_mask,
                                    gallery_nonkn_negative_mask,
                                    topk=int(args.gallery_hard_neg_topk),
                                    kn_fraction=args.gallery_kn_distractor_fraction,
                                )
                            )
                        else:
                            gallery_hard_loss = compute_fusion_gallery_hard_nce_loss(
                                gallery_scores,
                                gallery_positive_mask,
                                topk=int(args.gallery_hard_neg_topk),
                            )
                        total_loss = total_loss + (
                            gallery_loss_weight
                            * gallery_hard_weight
                            * gallery_hard_loss
                        )

                if cls_metrics_enabled and not cls_branch_enabled:
                    cred_level = (
                        compute_credible_level(gw_m, opt_coords)
                        if need_cred_level
                        else None
                    )
                    if (
                        _opt_first_detection_mjd is not None
                        and batch_event_time_mjd is not None
                    ):
                        dt_pos = compute_time_delta_days(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        )
                    else:
                        dt_pos = torch.zeros(
                            (batch_size,), device=device, dtype=torch.float32
                        )
                    logits_pos = model.fusion_logits(
                        g,
                        h_l,
                        z_l=z_l,
                        H_gw=H_gw,
                        cred_level=cred_level,
                        gw_s=gw_s,
                        gw_m=gw_m,
                        opt_coords=opt_coords,
                        dt_days=dt_pos,
                    )
                    cls_logits_available = True

                    metric_neg_idx = sample_mismatched_negatives(
                        batch_size,
                        device,
                        samples_per_gw=args.samples_per_gw,
                        gw_indices=gw_indices,
                        source_types=batch_source_types,
                    )
                    if metric_neg_idx is not None:
                        h_l_mis = h_l[metric_neg_idx].clone()
                        z_l_mis = z_l[metric_neg_idx].clone()
                        coords_mis = opt_coords[metric_neg_idx].clone()
                        dt_metric_mis = sample_mis_neg_dt_days(
                            batch_size, device, window_days=args.mis_neg_dt_window_days
                        )
                        cred_level_mis = (
                            compute_credible_level(gw_m, coords_mis)
                            if need_cred_level
                            else None
                        )
                        logits_hard = model.fusion_logits(
                            g,
                            h_l_mis,
                            z_l=z_l_mis,
                            H_gw=H_gw,
                            cred_level=cred_level_mis,
                            gw_s=gw_s,
                            gw_m=gw_m,
                            opt_coords=coords_mis,
                            dt_days=dt_metric_mis,
                        )
                        cls_hard_logits_available = True

                    if (
                        has_negatives
                        and gallery_extra_h_l is not None
                        and gallery_extra_z_l is not None
                    ):
                        cred_level_neg = (
                            compute_credible_level(gw_m, neg_coords)
                            if need_cred_level
                            else None
                        )
                        dt_neg_metric = (
                            compute_time_delta_days(
                                gallery_extra_event_time, batch_event_time_mjd
                            )
                            if (
                                gallery_extra_event_time is not None
                                and batch_event_time_mjd is not None
                            )
                            else torch.zeros(
                                (batch_size,), device=device, dtype=torch.float32
                            )
                        )
                        logits_neg = model.fusion_logits(
                            g,
                            gallery_extra_h_l,
                            z_l=gallery_extra_z_l,
                            H_gw=H_gw,
                            cred_level=cred_level_neg,
                            gw_s=gw_s,
                            gw_m=gw_m,
                            opt_coords=gallery_extra_coords,
                            dt_days=dt_neg_metric,
                        )
                        cls_extra_neg_logits_available = True

            val_total += total_loss.item()
            val_itc += itc_loss.item()
            val_cls += cls_loss.item()
            val_gallery += gallery_loss.item()
            val_gallery_hard += gallery_hard_loss.item()

            # --- Per-batch retrieval metrics (unified gallery incl. external negs) ---
            sim_g2o_d = sim_g2o.detach()
            sim_ext = sim_g2o_for_loss.detach()
            if sim_ext.size(1) > sim_g2o_d.size(1):
                ext_count = int(sim_ext.size(1) - sim_g2o_d.size(1))
                ext_gw_indices = torch.full(
                    (ext_count,),
                    -1,
                    device=gw_indices.device,
                    dtype=gw_indices.dtype,
                )
                batch_ret_gw = torch.cat([itc_gw_indices, ext_gw_indices], dim=0)
                batch_ret = compute_retrieval_metrics(
                    sim_ext, batch_ret_gw, ks=(1, 5, 10)
                )
            else:
                batch_ret = compute_retrieval_metrics(
                    sim_ext, itc_gw_indices, ks=(1, 5, 10)
                )
            retrieval_metrics_accum.append(batch_ret)

            # --- ITC accuracy (in-batch only) ---
            itc_preds = sim_g2o_d.argmax(dim=1)
            anchor_gw = itc_gw_indices
            pred_gw = itc_gw_indices[itc_preds]
            itc_acc = (anchor_gw == pred_gw).float().mean().item()

            if cls_logits_available:
                pos_acc = 0.0
                neg_gw_acc = 0.0
                if pos_mask.any():
                    pos_acc = (
                        (logits_pos[pos_mask].argmax(dim=1) == 1).float().mean().item()
                    )
                if neg_gw_mask.any():
                    neg_gw_acc = (
                        (logits_pos[neg_gw_mask].argmax(dim=1) == 0)
                        .float()
                        .mean()
                        .item()
                    )
                acc_parts = [pos_acc, neg_gw_acc]
                hard_acc = 0.0
                if cls_hard_logits_available:
                    hard_acc = (
                        (logits_hard.argmax(dim=1) == labels_neg).float().mean().item()
                    )
                    acc_parts.append(hard_acc)
                neg_acc = 0.0
                if has_negatives and cls_extra_neg_logits_available:
                    neg_acc = (
                        (logits_neg.argmax(dim=1) == labels_neg).float().mean().item()
                    )
                    acc_parts.append(neg_acc)
                total_acc = sum(acc_parts) / float(max(1, len(acc_parts)))
            else:
                pos_acc = 0.0
                neg_gw_acc = 0.0
                hard_acc = 0.0
                neg_acc = 0.0
                total_acc = 0.0

            val_itc_acc += itc_acc
            val_pos_acc += pos_acc
            val_neg_gw_acc += neg_gw_acc
            val_hard_acc += hard_acc
            val_neg_acc += neg_acc
            val_total_acc += total_acc
            val_batches += 1

            # --- Accumulate for global classification + embedding metrics ---
            if cls_metrics_enabled and cls_logits_available:
                pair_probs = torch.softmax(logits_pos.detach().float(), dim=1)[:, 1]
                if pos_mask.any():
                    pos_probs = pair_probs[pos_mask]
                    all_cls_probs.append(pos_probs.cpu())
                    all_cls_labels.append(
                        torch.ones(int(pos_mask.sum()), dtype=torch.long).cpu()
                    )
                    all_cls_sources.extend(["aligned"] * int(pos_mask.sum()))
                if neg_gw_mask.any():
                    neg_gw_probs = pair_probs[neg_gw_mask]
                    all_cls_probs.append(neg_gw_probs.cpu())
                    all_cls_labels.append(
                        torch.zeros(int(neg_gw_mask.sum()), dtype=torch.long).cpu()
                    )
                    all_cls_sources.extend(["neg_gw"] * int(neg_gw_mask.sum()))
                if pair_is_neg_gw.any():
                    if gw_source_type_table is None or gw_neg_type_table is None:
                        raise ValueError(
                            "Negative-GW validation requires source_type and neg_type tables."
                        )
                    source_codes = gw_source_type_table[gw_indices]
                    neg_types_batch = gw_neg_type_table[gw_indices]
                    for source_code, source_name in ((0, "bns"), (1, "nsbh")):
                        for neg_type in (1, 2):
                            stratum_mask = (
                                pair_is_neg_gw
                                & (source_codes == source_code)
                                & (neg_types_batch == neg_type)
                            )
                            if stratum_mask.any():
                                neg_gw_probabilities[(source_name, neg_type)].append(
                                    pair_probs[stratum_mask].cpu()
                                )

                if cls_hard_logits_available:
                    hard_probs = torch.softmax(logits_hard.detach().float(), dim=1)[
                        :, 1
                    ]
                    all_cls_probs.append(hard_probs.cpu())
                    all_cls_labels.append(labels_neg[:batch_size].cpu())
                    all_cls_sources.extend(["mismatched"] * batch_size)

                if has_negatives and cls_extra_neg_logits_available:
                    neg_probs_val = torch.softmax(logits_neg.detach().float(), dim=1)[
                        :, 1
                    ]
                    all_cls_probs.append(neg_probs_val.cpu())
                    all_cls_labels.append(labels_neg[:batch_size].cpu())
                    all_cls_sources.extend(["external"] * batch_size)

            all_feat_g.append(feat_g.detach().cpu())
            all_feat_o.append(feat_o.detach().cpu())
            all_gw_indices.append(itc_gw_indices.cpu())

            # Memory cleanup in validation loop
            del g, z_l, h_l, sim_g2o, sim_g2o_d, itc_loss, cls_loss, total_loss
            del (
                logits_pos,
                logits_hard,
                aligned_pos_loss,
                neg_gw_loss,
                hard_loss,
                feat_g,
                feat_o,
            )
            if has_negatives:
                del logits_neg, neg_loss

    if val_batches == 0:
        model.train()
        return None

    # --- Compute global metrics ---
    # Average per-batch retrieval metrics
    retrieval = {}
    if retrieval_metrics_accum:
        for key in retrieval_metrics_accum[0]:
            retrieval[key] = sum(d[key] for d in retrieval_metrics_accum) / len(
                retrieval_metrics_accum
            )
    if fusion_gallery_metrics_accum:
        for key in fusion_gallery_metrics_accum[0]:
            retrieval[key] = sum(d[key] for d in fusion_gallery_metrics_accum) / len(
                fusion_gallery_metrics_accum
            )

    # Global classification metrics
    if all_cls_probs:
        cls_metrics = compute_classification_metrics(
            torch.cat(all_cls_probs),
            torch.cat(all_cls_labels),
            all_sources=all_cls_sources,
        )
    else:
        cls_metrics = {}

    neg_gw_strata = {}
    populated_recalls = []
    for (source, neg_type), probability_blocks in neg_gw_probabilities.items():
        key = f"{source}_type{neg_type}"
        if probability_blocks:
            probabilities = torch.cat(probability_blocks)
            recall = float((probabilities < 0.5).float().mean().item())
            count = int(probabilities.numel())
            populated_recalls.append(recall)
        else:
            recall = 0.0
            count = 0
        neg_gw_strata[key] = {"recall": recall, "count": count}
    neg_gw_strata["min_recall"] = (
        min(populated_recalls) if len(populated_recalls) == 4 else 0.0
    )
    # Global embedding metrics
    emb_metrics = compute_embedding_metrics(
        torch.cat(all_feat_g),
        torch.cat(all_feat_o),
        torch.cat(all_gw_indices),
    )

    metrics = {
        "total": val_total / val_batches,
        "itc": val_itc / val_batches,
        "cls": val_cls / val_batches,
        "gallery": val_gallery / val_batches,
        "gallery_hard": val_gallery_hard / val_batches,
        "itc_acc": val_itc_acc / val_batches,
        "pos_acc": val_pos_acc / val_batches,
        "hard_acc": val_hard_acc / val_batches,
        "neg_acc": val_neg_acc / val_batches,
        "neg_gw_acc": val_neg_gw_acc / val_batches,
        "total_acc": val_total_acc / val_batches,
        "retrieval": retrieval,
        "classification": cls_metrics,
        "embedding": emb_metrics,
        "neg_gw_strata": neg_gw_strata,
    }
    if any(int(v.get("count", 0)) > 0 for v in dt_stats.values()):
        metrics["time_delta"] = _finalize_time_delta_stats(dt_stats)
    model.train()
    # Memory cleanup after validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return metrics


def train(args):
    args.validation_gallery_sizes = list(
        parse_validation_gallery_sizes(args.validation_gallery_sizes)
    )
    if (
        not bool(args.validation_gallery_enable)
        and args.best_ckpt_metric == "hard_gallery_macro_retrieval_score"
    ):
        args.best_ckpt_metric = "fusion_gallery_mrr"
        print(
            "validation_gallery_enable=false: falling back to "
            "best_ckpt_metric='fusion_gallery_mrr'."
        )
    if bool(args.validation_gallery_enable):
        if args.val_split is None or not (0.0 < float(args.val_split) < 1.0):
            raise ValueError(
                "validation_gallery_enable requires val_split to be in (0, 1)."
            )
        if int(args.validation_gallery_queries_per_source) < 1:
            raise ValueError("validation_gallery_queries_per_source must be >= 1.")
        if int(args.validation_gallery_trials) < 1:
            raise ValueError("validation_gallery_trials must be >= 1.")
        if float(args.validation_gallery_time_window_days) <= 0.0:
            raise ValueError("validation_gallery_time_window_days must be > 0.")
        if not (0.0 < float(args.validation_gallery_credible_level_max) <= 1.0):
            raise ValueError("validation_gallery_credible_level_max must be in (0, 1].")
        metric_weight_sum = float(args.validation_gallery_mrr_weight) + float(
            args.validation_gallery_recall_at_1_weight
        )
        if (
            float(args.validation_gallery_mrr_weight) < 0.0
            or float(args.validation_gallery_recall_at_1_weight) < 0.0
            or metric_weight_sum <= 0.0
        ):
            raise ValueError(
                "validation gallery metric weights must be non-negative and sum to > 0."
            )
    if args.temp_final is None:
        args.temp_final = args.temp_init
    if args.temp_min <= 0 or args.temp_max <= 0:
        raise ValueError("temp_min and temp_max must be > 0.")
    if args.temp_min >= args.temp_max:
        raise ValueError("temp_min must be < temp_max.")

    if args.hardneg_min_candidates < 1:
        raise ValueError("hardneg_min_candidates must be >= 1.")
    if not 0.0 < float(getattr(args, "encoder_lr_ratio", 0.1)) <= 1.0:
        raise ValueError("encoder_lr_ratio must be in (0, 1].")
    for key in ("stage_joint_itc_start_weight", "stage_joint_itc_end_weight"):
        value = float(getattr(args, key, 0.5 if key.endswith("start_weight") else 0.25))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be in [0, 1].")
    if not 0.0 <= float(args.neg_gw_guardrail_recall) <= 1.0:
        raise ValueError("neg_gw_guardrail_recall must be in [0, 1].")
    for key in (
        "cls_aligned_pos_weight",
        "cls_neg_gw_weight",
        "cls_mismatched_weight",
        "cls_external_neg_weight",
    ):
        if float(getattr(args, key, 0.0)) < 0.0:
            raise ValueError(f"{key} must be >= 0.")
    if args.gallery_hard_neg_topk < 1:
        raise ValueError("gallery_hard_neg_topk must be >= 1.")
    if args.gallery_candidate_mode == "mixed_kn_nonkn":
        if args.gallery_training_size < 3:
            raise ValueError("gallery_training_size must be >= 3 in mixed mode.")
        if not 0.0 < float(args.gallery_kn_distractor_fraction) < 1.0:
            raise ValueError("gallery_kn_distractor_fraction must be in (0, 1).")
        if not 0.0 <= float(args.gallery_nonkn_empirical_fraction) <= 1.0:
            raise ValueError("gallery_nonkn_empirical_fraction must be in [0, 1].")
        if args.gallery_candidate_coordinate_mode != "positive_shared":
            raise ValueError("mixed gallery requires positive_shared coordinates.")
        if args.gallery_kn_distractor_time_mode != "parent_relative":
            raise ValueError("mixed gallery requires parent-relative KN timing.")
        if args.gallery_nonkn_distractor_time_mode != "empirical_uniform_mixture":
            raise ValueError("mixed gallery requires empirical/uniform non-KN timing.")
        if args.gallery_hard_neg_enable and args.gallery_hard_neg_topk < 2:
            raise ValueError("mixed stratified hard-negative topk must be >= 2.")
        if args.neg_data_path is None:
            raise ValueError("mixed training gallery requires neg_data_path.")
    if args.gallery_hard_neg_weight < 0:
        raise ValueError("gallery_hard_neg_weight must be >= 0.")
    if args.gallery_hard_neg_start_after_retrieval_epochs < 0:
        raise ValueError("gallery_hard_neg_start_after_retrieval_epochs must be >= 0.")
    if args.gallery_hard_neg_ramp_epochs < 0:
        raise ValueError("gallery_hard_neg_ramp_epochs must be >= 0.")
    if args.gallery_loss_weight < 0:
        raise ValueError("gallery_loss_weight must be >= 0.")
    if args.gallery_loss_ramp_epochs < 0:
        raise ValueError("gallery_loss_ramp_epochs must be >= 0.")
    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be >= 0.")
    if args.best_ckpt_min_delta is None:
        args.best_ckpt_min_delta = float(args.early_stop_min_delta)
    if args.best_ckpt_min_delta < 0:
        raise ValueError("best_ckpt_min_delta must be >= 0.")
    if args.validation_gallery_eval_interval < 1:
        raise ValueError("validation_gallery_eval_interval must be >= 1.")
    if get_training_schedule_total_epochs(args) < int(args.epochs):
        raise ValueError("training schedule horizon must be >= epochs.")
    if args.cls_start_epoch < 0:
        raise ValueError("cls_start_epoch must be >= 0.")
    if args.cls_ramp_epochs < 0:
        raise ValueError("cls_ramp_epochs must be >= 0.")
    if args.retrieval_start_epoch < 0:
        raise ValueError("retrieval_start_epoch must be >= 0.")
    if args.fusion_gallery_metrics_start_epoch < 0:
        raise ValueError("fusion_gallery_metrics_start_epoch must be >= 0.")
    if args.itc_decay_start_epoch < 0:
        raise ValueError("itc_decay_start_epoch must be >= 0.")
    if args.itc_decay_epochs < 0:
        raise ValueError("itc_decay_epochs must be >= 0.")
    if args.gallery_score_chunk_size < 1:
        raise ValueError("gallery_score_chunk_size must be >= 1.")
    validate_sequential_curriculum(args)
    if is_staged_training_enabled(args):
        print(f"Sequential loss curriculum: {format_sequential_curriculum(args)}")
    validate_best_ckpt_selection_schedule(args)
    print(
        "Best-checkpoint stability schedule: "
        f"{format_best_ckpt_selection_schedule(args)}"
    )
    args.gallery_distractor_time_mode = (
        str(args.gallery_distractor_time_mode).strip().lower()
    )
    if args.gallery_distractor_time_mode in {"synthetic", "after_gw"}:
        args.gallery_distractor_time_mode = "synthetic_after_gw"
    if args.gallery_distractor_time_mode in {"real", "none"}:
        args.gallery_distractor_time_mode = "actual"
    if args.gallery_distractor_time_mode in {"candidate_relative", "empirical"}:
        args.gallery_distractor_time_mode = "parent_relative"
    if args.gallery_distractor_time_mode not in {
        "actual",
        "synthetic_after_gw",
        "parent_relative",
    }:
        raise ValueError(
            "gallery_distractor_time_mode must be 'actual', "
            "'synthetic_after_gw', or 'parent_relative'."
        )
    args.cls_distractor_time_mode = str(args.cls_distractor_time_mode).strip().lower()
    if args.cls_distractor_time_mode not in {"legacy", "mixed_empirical"}:
        raise ValueError(
            "cls_distractor_time_mode must be 'legacy' or 'mixed_empirical'."
        )
    if not 0.0 <= float(args.cls_external_empirical_dt_fraction) <= 1.0:
        raise ValueError("cls_external_empirical_dt_fraction must be in [0, 1].")
    if (
        args.gallery_distractor_time_mode == "parent_relative"
        and bool(args.gallery_include_extra_negatives)
        and args.gallery_candidate_mode == "legacy"
    ):
        raise ValueError(
            "parent_relative gallery timing requires "
            "gallery_include_extra_negatives=false."
        )
    if args.gallery_distractor_time_window_days <= 0:
        raise ValueError("gallery_distractor_time_window_days must be > 0.")
    if args.time_delta_cls_scale_days is None:
        args.time_delta_cls_scale_days = float(args.time_compat_tau_days)
    if args.time_delta_cls_scale_days <= 0:
        raise ValueError("time_delta_cls_scale_days must be > 0.")
    if args.time_delta_cls_clip < 0:
        raise ValueError("time_delta_cls_clip must be >= 0.")
    if args.mis_neg_dt_window_days <= 0:
        raise ValueError("mis_neg_dt_window_days must be > 0.")
    args.fusion_mode = normalize_fusion_mode(
        getattr(args, "fusion_mode", None), dual_fusion=args.dual_fusion
    )
    args.dual_fusion = args.fusion_mode != "legacy_g2o"
    if args.gallery_candidate_mode == "mixed_kn_nonkn":
        if args.fusion_mode != "physical_dual_hgw":
            raise ValueError("mixed training gallery requires physical_dual_hgw.")
        if bool(args.use_similarity_as_cls_input):
            raise ValueError(
                "mixed positive-shared coordinates require "
                "use_similarity_as_cls_input=false."
            )
    args._hardneg_window_days = parse_day_windows(args.hardneg_time_window_days)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args._dataset_window_metadata = _collect_dataset_window_metadata(args)
    args._effective_input_window_metadata = _build_effective_input_window_metadata(args)
    print(
        f"Dataset window metadata: {json.dumps(args._dataset_window_metadata, indent=2)}"
    )
    print(
        f"Effective input window metadata: {json.dumps(args._effective_input_window_metadata, indent=2)}"
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler = GradScaler(enabled=(not use_bf16))
        print(f"Running on device: {device} | AMP dtype={amp_dtype}")
    else:
        amp_dtype = torch.float32
        scaler = GradScaler(enabled=False)
        print(f"Running on device: {device} | No AMP (CPU)")
    need_zero_time_mjd_for_cls = bool(
        args.use_time_delta_cls_feature
        or args.gallery_loss_weight > 0
        or args.time_compat_weight > 0
    )
    if not 0.0 <= float(args.neg_gw_pair_ratio) < 1.0:
        raise ValueError("neg_gw_pair_ratio must be in [0, 1).")
    train_neg_gw_ratio = 0.0
    if float(args.neg_gw_pair_ratio) > 0.0 and is_cls_branch_enabled(args):
        with h5py.File(args.data_path, "r") as f:
            required = (
                "events/gw_data/has_kn",
                "events/gw_data/neg_type",
                "events/gw_data/source_type",
                "events/gw_data/event_time_mjd",
            )
            if all(name in f for name in required):
                has_kn = np.asarray(f["events/gw_data/has_kn"][:])
                if np.any(has_kn == 0):
                    train_neg_gw_ratio = float(args.neg_gw_pair_ratio)
            if train_neg_gw_ratio == 0.0:
                print(
                    "Negative-GW pairs disabled: the training H5 has no complete negative-GW schema/events."
                )
    elif float(args.neg_gw_pair_ratio) > 0.0:
        print(
            "Negative-GW pairs disabled because the classification branch is inactive."
        )
    print(f"Train negative-GW pair ratio: {train_neg_gw_ratio:.3f}")
    print(
        f"CLS/gallery/time-compat optical first-detection metadata requested: {int(need_zero_time_mjd_for_cls)}"
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(
        os.path.dirname(args.ckpt_path),
        "tb_logs",
        f"run_magiks_{timestamp}_pid{os.getpid()}",
    )
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging started at: {log_dir}")
    extra_neg_timeaware_enable = bool(args.neg_data_path is not None)
    extra_neg_timeaware_seed = int(getattr(args, "split_seed", args.seed))
    if extra_neg_timeaware_enable:
        print(
            "Extra-negative time-aware sampling enabled "
            f"(reuse hardneg windows={args._hardneg_window_days}, "
            f"min_candidates={args.hardneg_min_candidates}, "
            f"seed={extra_neg_timeaware_seed})"
        )
    else:
        print("Extra-negative time-aware sampling disabled.")

    val_loader = None
    val_gw_map = None
    if args.val_split is not None and 0 < args.val_split < 1:
        if args.itc_loss_type == "supcon":
            loader_result = create_supcon_dataloaders(
                h5_path=args.data_path,
                batch_size=args.batch_size,
                samples_per_gw=args.samples_per_gw,
                val_batch_size=args.val_batch_size,
                steps_per_epoch=args.steps_per_epoch,
                val_steps_per_epoch=args.val_steps_per_epoch,
                val_split=args.val_split,
                split_seed=args.split_seed,
                val_split_stratify_by_source=bool(args.val_split_stratify_by_source),
                return_split_maps=True,
                num_workers=args.num_workers,
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=args.prefetch_factor,
                cache_in_memory=bool(args.cache_in_memory),
                negative_h5_path=args.neg_data_path,
                negative_group=args.neg_group,
                min_lc_per_gw=args.min_lc_per_gw,
                return_zero_time_mjd=need_zero_time_mjd_for_cls,
                nonkn_cls_base_field=args.nonkn_cls_base_field,
                extra_negative_timeaware_enable=extra_neg_timeaware_enable,
                extra_negative_timeaware_windows_days=args._hardneg_window_days,
                extra_negative_timeaware_min_candidates=args.hardneg_min_candidates,
                extra_negative_timeaware_seed=extra_neg_timeaware_seed,
                opt_input_window_start=args.ref_start,
                opt_input_window_end=args.ref_end,
                train_neg_gw_ratio=train_neg_gw_ratio,
            )
            (
                train_loader,
                val_loader,
                steps_per_epoch,
                val_steps,
                _train_gw_map,
                val_gw_map,
            ) = loader_result
            print(
                f"SupCon mode: {args.samples_per_gw} samples/GW, "
                f"{args.batch_size // args.samples_per_gw} GW/batch"
            )
        else:
            loader_result = create_train_val_dataloaders(
                h5_path=args.data_path,
                batch_size=args.batch_size,
                val_batch_size=args.val_batch_size,
                steps_per_epoch=args.steps_per_epoch,
                val_steps_per_epoch=args.val_steps_per_epoch,
                val_split=args.val_split,
                split_seed=args.split_seed,
                val_split_stratify_by_source=bool(args.val_split_stratify_by_source),
                return_split_maps=True,
                num_workers=args.num_workers,
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=args.prefetch_factor,
                cache_in_memory=bool(args.cache_in_memory),
                negative_h5_path=args.neg_data_path,
                negative_group=args.neg_group,
                return_zero_time_mjd=need_zero_time_mjd_for_cls,
                nonkn_cls_base_field=args.nonkn_cls_base_field,
                extra_negative_timeaware_enable=extra_neg_timeaware_enable,
                extra_negative_timeaware_windows_days=args._hardneg_window_days,
                extra_negative_timeaware_min_candidates=args.hardneg_min_candidates,
                extra_negative_timeaware_seed=extra_neg_timeaware_seed,
                opt_input_window_start=args.ref_start,
                opt_input_window_end=args.ref_end,
                train_neg_gw_ratio=train_neg_gw_ratio,
            )
            (
                train_loader,
                val_loader,
                steps_per_epoch,
                val_steps,
                _train_gw_map,
                val_gw_map,
            ) = loader_result
        print(f"Train Steps/Epoch: {steps_per_epoch} | Val Steps/Epoch: {val_steps}")
    else:
        if args.steps_per_epoch is not None:
            steps_per_epoch = args.steps_per_epoch
        else:
            with h5py.File(args.data_path, "r") as f:
                total_optical = f["events/optical_data/values"].shape[0]
            steps_per_epoch = total_optical // args.batch_size
            print(f"Dataset Size: {total_optical} | Steps/Epoch: {steps_per_epoch}")

        train_loader = create_training_dataloader(
            h5_path=args.data_path,
            batch_size=args.batch_size,
            steps_per_epoch=steps_per_epoch,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=args.prefetch_factor,
            cache_in_memory=bool(args.cache_in_memory),
            negative_h5_path=args.neg_data_path,
            negative_group=args.neg_group,
            return_zero_time_mjd=need_zero_time_mjd_for_cls,
            nonkn_cls_base_field=args.nonkn_cls_base_field,
            extra_negative_timeaware_enable=extra_neg_timeaware_enable,
            extra_negative_timeaware_windows_days=args._hardneg_window_days,
            extra_negative_timeaware_min_candidates=args.hardneg_min_candidates,
            extra_negative_timeaware_seed=extra_neg_timeaware_seed,
            opt_input_window_start=args.ref_start,
            opt_input_window_end=args.ref_end,
            loader_usage="train",
            loader_label="Train DataLoader",
            train_neg_gw_ratio=train_neg_gw_ratio,
        )

    mtan_cfg = resolve_mtan_runtime_config(args)
    print("mTAN runtime config:")
    print(
        json.dumps(
            {k: (list(v) if isinstance(v, tuple) else v) for k, v in mtan_cfg.items()},
            indent=2,
        )
    )

    model = MAGIKSModel(
        gw_scalar_dim=7,
        gw_skymap_channels=7,
        optical_input_dim=6,
        ref_time_dim=args.ref_dim,
        enc_dim=args.enc_dim,
        proj_dim=args.proj_dim,
        optical_curve_dim=args.optical_curve_dim,
        optical_coord_dim=args.optical_coord_dim,
        optical_curve_hidden_dim=args.optical_curve_hidden_dim,
        contrastive_hidden_dim=args.contrastive_hidden_dim,
        fusion_attn_dim=args.fusion_attn_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        temp_init=args.temp_init,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
        gw_dropout=args.gw_dropout,
        opt_dropout=args.opt_dropout,
        proj_dropout=args.proj_dropout,
        feature_dropout=args.feature_dropout,
        fusion_dropout=args.fusion_dropout,
        label_smoothing=args.label_smoothing,
        itc_label_smoothing=args.itc_label_smoothing,
        use_lightweight_gw=args.use_lightweight_gw,
        dual_fusion=args.dual_fusion,
        fusion_mode=args.fusion_mode,
        use_cred_level_feature=args.use_cred_level_feature,
        use_similarity_as_cls_input=args.use_similarity_as_cls_input,
        time_compat_weight=args.time_compat_weight,
        time_compat_tau_days=args.time_compat_tau_days,
        time_compat_power=args.time_compat_power,
        time_compat_max_penalty=args.time_compat_max_penalty,
        use_time_delta_cls_feature=args.use_time_delta_cls_feature,
        time_delta_cls_scale_days=args.time_delta_cls_scale_days,
        time_delta_cls_clip=args.time_delta_cls_clip,
        fusion_physical_weight=args.fusion_physical_weight,
        fusion_spatial_weight=args.fusion_spatial_weight,
        mtan_snr_s0=float(mtan_cfg["mtan_snr_s0"]),
        mtan_snr_beta=float(mtan_cfg["mtan_snr_beta"]),
        mtan_snr_clip_min=float(mtan_cfg["mtan_snr_clip_min"]),
        mtan_snr_clip_max=float(mtan_cfg["mtan_snr_clip_max"]),
        mtan_snr_eps=float(mtan_cfg["mtan_snr_eps"]),
        mtan_lupt_psfflux_zp=float(mtan_cfg["mtan_lupt_psfflux_zp"]),
        mtan_lupt_k=float(mtan_cfg["mtan_lupt_k"]),
        mtan_lupt_m5_mag=tuple(mtan_cfg["mtan_lupt_m5_mag"]),
    ).to(device)

    cls_branch_enabled = is_cls_branch_enabled(args)

    if args.use_lightweight_gw:
        print("Using lightweight GW encoder (~100K params) to prevent overfitting.")
    print(f"Using fusion_mode={args.fusion_mode}.")
    if args.fusion_mode == "legacy_dual":
        print(
            "Legacy dual cross-attention fusion enabled with per-pair credible level."
        )
    elif args.fusion_mode == "physical_dual_hgw":
        print(
            "Physical dual fusion enabled: GW-parameter->optical and "
            "coord-query->H_gw sequence retrieval "
            f"(use_similarity_as_cls_input={int(bool(args.use_similarity_as_cls_input))}, "
            f"use_cred_level_feature={int(bool(args.use_cred_level_feature))}, "
            f"use_time_delta_cls_feature={int(bool(args.use_time_delta_cls_feature))}, "
            f"fusion_physical_weight={float(args.fusion_physical_weight):.3g}, "
            f"fusion_spatial_weight={float(args.fusion_spatial_weight):.3g})."
        )
    elif args.fusion_mode == "concat_proj":
        print("Ablation mode: no cross-attention, classifier on [proj_gw; proj_opt].")
    if not cls_branch_enabled:
        print(
            "Retrieval-only mode enabled: fusion/classification branch disabled; extra negatives go into contrastive loss."
        )

    if args.gallery_candidate_mode == "mixed_kn_nonkn":
        n_kn, n_nonkn = allocate_mixed_negative_counts(
            args.gallery_training_size, args.gallery_kn_distractor_fraction
        )
        hard_kn, hard_nonkn = allocate_mixed_negative_counts(
            args.gallery_hard_neg_topk + 1,
            args.gallery_kn_distractor_fraction,
        )
        print(
            "Mixed training gallery: "
            f"1 target + {n_kn} KN + {n_nonkn} non-KN; "
            f"stratified hard negatives={hard_kn} KN + {hard_nonkn} non-KN."
        )

    if hasattr(torch, "compile"):
        model = torch.compile(model)
        print("Model compiled with torch.compile() for optimized execution.")

    optimizer = torch.optim.AdamW(build_optimizer_param_groups(model, args))
    start_epoch = 0
    global_step = 0
    resume_training_state = {}
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = load_training_checkpoint(args.resume, map_location=device)
        migrate_time_embed_state_dict(ckpt["model_state_dict"])
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError as exc:
            msg = str(exc)
            if "size mismatch" in msg:
                raise RuntimeError(
                    "Resume checkpoint is incompatible with the current classifier head shape. "
                    "If use_time_delta_cls_feature changed, start from scratch or use a matching checkpoint."
                ) from exc
            raise
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", -1) + 1
        resume_training_state = ckpt.get("training_state", {})
        global_step = int(
            resume_training_state.get("global_step", start_epoch * steps_per_epoch)
        )
        restore_rng_state(ckpt.get("rng_state"))
        print(f"Resumed from {args.resume} at epoch {start_epoch}.")
    model.train()
    has_negatives = args.neg_data_path is not None
    print(
        "Fusion gallery distractor time policy: "
        f"mode={args.gallery_distractor_time_mode}, "
        f"window_days={float(args.gallery_distractor_time_window_days):.3g}"
    )
    gw_event_time_mjd_table = load_gw_event_time_mjd_table(args.data_path, device)
    if gw_event_time_mjd_table is None:
        raise ValueError(
            "Hard-negative time-shift re-encoding requires 'events/gw_data/event_time_mjd' "
            "in training dataset."
        )
    gw_source_type_table = load_gw_source_type_table(
        args.data_path, device, required=True
    )
    gw_neg_type_table = load_gw_neg_type_table(args.data_path, device, required=True)
    print("Mismatched-negative policy: prefer a different GW of the same source type.")
    print(
        "Extra-negative time-aware sampling config: "
        f"windows_days={args._hardneg_window_days}, "
        f"min_candidates={args.hardneg_min_candidates}"
    )

    validation_gallery_context = None
    if bool(args.validation_gallery_enable):
        if val_gw_map is None:
            raise ValueError(
                "validation_gallery_enable requires an event-level validation split."
            )
        validation_gallery_context = ValidationGalleryContext.build(
            args,
            val_gw_map,
        )

    pbar_update_every = 500
    tb_log_interval = 50
    best_val_score = resume_training_state.get("best_val_score")
    best_epoch_idx = resume_training_state.get("best_epoch_idx")
    best_val_metrics = dict(resume_training_state.get("best_val_metrics", {}))
    epochs_no_improve = int(resume_training_state.get("epochs_no_improve", 0))
    best_tracking_started = bool(
        resume_training_state.get("best_tracking_started", False)
    )
    best_tracking_start_epoch = resume_training_state.get("best_tracking_start_epoch")
    last_selection_score = resume_training_state.get("last_selection_score")
    lr_scheduler = build_lr_scheduler(optimizer, args, steps_per_epoch, global_step)
    last_completed_epoch = start_epoch - 1

    for epoch in range(start_epoch, args.epochs):
        last_completed_epoch = epoch
        epoch_total = 0.0
        training_stage = resolve_training_stage(args, epoch)
        configure_model_for_stage(model, args, training_stage)
        batch_sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_stage"):
            batch_sampler.set_stage(
                "alignment" if training_stage in {"alignment", "itc"} else "joint"
            )
        print(
            f"Training stage: {training_stage} | "
            f"ITC weight={compute_itc_weight(args, epoch):.4f} | "
            f"CLS weight={compute_cls_weight(args, epoch):.4f} | "
            f"Retrieval weight={compute_gallery_loss_weight(args, epoch):.4f}"
        )
        epoch_itc = 0.0
        epoch_cls = 0.0
        ref_time_cache = None
        apply_temperature_schedule(model, args, epoch)
        cls_weight = compute_cls_weight(args, epoch)
        itc_weight = compute_itc_weight(args, epoch)
        gallery_loss_weight = compute_gallery_loss_weight(args, epoch)
        best_selection_eligible = is_best_ckpt_selection_eligible(args, epoch)
        if best_selection_eligible and not best_tracking_started:
            best_tracking_started = True
            best_tracking_start_epoch = int(epoch)
            best_val_score = None
            best_epoch_idx = None
            epochs_no_improve = 0
            if not cls_branch_enabled:
                print(
                    "Best-checkpoint selection activated "
                    f"at epoch {epoch+1} (retrieval-only / fusion branch disabled)."
                )
            else:
                print("Best-checkpoint selection activated " f"at epoch {epoch+1}.")

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            mininterval=0,
            miniters=pbar_update_every,
        )

        for batch_idx, batch_data in enumerate(pbar):
            pair_is_neg_gw = None
            if train_neg_gw_ratio > 0.0:
                pair_is_neg_gw = batch_data[-1]
                batch_data = batch_data[:-1]
            _opt_zero_time_mjd_base = None
            _neg_zero_time_mjd_base = None
            _neg_zero_time_mjd_cls_base = None
            _opt_first_detection_mjd = None
            if has_negatives:
                if len(batch_data) >= 17:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        neg_t,
                        neg_v,
                        neg_mask,
                        neg_err,
                        neg_coords,
                        _opt_zero_time_mjd_base,
                        _neg_zero_time_mjd_base,
                        _neg_zero_time_mjd_cls_base,
                        _opt_first_detection_mjd,
                    ) = batch_data[:17]
                elif len(batch_data) >= 16:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        neg_t,
                        neg_v,
                        neg_mask,
                        neg_err,
                        neg_coords,
                        _opt_zero_time_mjd_base,
                        _neg_zero_time_mjd_base,
                        _neg_zero_time_mjd_cls_base,
                    ) = batch_data[:16]
                elif len(batch_data) >= 15:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        neg_t,
                        neg_v,
                        neg_mask,
                        neg_err,
                        neg_coords,
                        _opt_zero_time_mjd_base,
                        _neg_zero_time_mjd_base,
                    ) = batch_data[:15]
                else:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        neg_t,
                        neg_v,
                        neg_mask,
                        neg_err,
                        neg_coords,
                    ) = batch_data[:13]
            else:
                if len(batch_data) >= 10:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        _opt_zero_time_mjd_base,
                        _opt_first_detection_mjd,
                    ) = batch_data[:10]
                elif len(batch_data) >= 9:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                        _opt_zero_time_mjd_base,
                    ) = batch_data[:9]
                else:
                    (
                        gw_s,
                        gw_m,
                        opt_t,
                        opt_v,
                        opt_mask,
                        opt_err,
                        opt_coords,
                        gw_indices,
                    ) = batch_data

            gw_s = gw_s.to(device, non_blocking=True)
            gw_m = gw_m.to(device, non_blocking=True)
            gw_s, gw_m = augment_gw_data(
                gw_s,
                gw_m,
                training=True,
                noise_std=args.gw_aug_noise,
                scalar_jitter=args.gw_aug_jitter,
                channel_dropout_prob=args.gw_aug_dropout,
            )
            opt_t = opt_t.to(device, non_blocking=True)
            opt_v = opt_v.to(device, non_blocking=True)
            opt_mask = opt_mask.to(device, non_blocking=True)
            opt_err = opt_err.to(device, non_blocking=True)
            opt_coords = opt_coords.to(device, non_blocking=True)
            gw_indices = gw_indices.to(device, non_blocking=True).long()
            if pair_is_neg_gw is None:
                pair_is_neg_gw = torch.zeros(
                    (gw_indices.shape[0],), device=device, dtype=torch.bool
                )
            else:
                pair_is_neg_gw = pair_is_neg_gw.to(device, non_blocking=True).bool()
            itc_pair_mask = build_itc_pair_mask(
                gw_indices, pair_is_neg_gw, min_pairs_per_gw=args.min_lc_per_gw
            )
            if not bool(itc_pair_mask.any()):
                raise ValueError(
                    "Each training batch must retain at least one positive GW-optical pair"
                )
            batch_event_time_mjd = None
            if gw_event_time_mjd_table is not None:
                batch_event_time_mjd = gw_event_time_mjd_table[gw_indices]
            batch_source_types = None
            if gw_source_type_table is not None:
                batch_source_types = gw_source_type_table[gw_indices]

            if has_negatives:
                neg_t = neg_t.to(device, non_blocking=True)
                neg_v = neg_v.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)
                if _neg_zero_time_mjd_base is not None:
                    _neg_zero_time_mjd_base = _neg_zero_time_mjd_base.to(
                        device, non_blocking=True
                    )
                if _neg_zero_time_mjd_cls_base is not None:
                    _neg_zero_time_mjd_cls_base = _neg_zero_time_mjd_cls_base.to(
                        device, non_blocking=True
                    )
            if _opt_first_detection_mjd is not None:
                _opt_first_detection_mjd = _opt_first_detection_mjd.to(
                    device, non_blocking=True
                )
            opt_t_raw = opt_t.clone()
            opt_v_raw = opt_v.clone()
            opt_mask_raw = opt_mask.clone()
            opt_err_raw = opt_err.clone()

            opt_t, opt_v, opt_mask, opt_err = augment_optical_data(
                opt_t,
                opt_v,
                opt_mask,
                opt_err,
                training=True,
                time_jitter=args.opt_aug_time_jitter,
                flux_noise=args.opt_aug_noise,
                obs_dropout=args.opt_aug_dropout,
                band_dropout=args.opt_aug_band_dropout,
            )
            if has_negatives:
                neg_t, neg_v, neg_mask, neg_err = augment_optical_data(
                    neg_t,
                    neg_v,
                    neg_mask,
                    neg_err,
                    training=True,
                    time_jitter=args.opt_aug_time_jitter,
                    flux_noise=args.opt_aug_noise,
                    obs_dropout=args.opt_aug_dropout,
                    band_dropout=args.opt_aug_band_dropout,
                )

            batch_size = gw_s.size(0)

            if (
                ref_time_cache is None
                or ref_time_cache.shape[0] != batch_size
                or ref_time_cache.dtype != opt_t.dtype
            ):
                ref_time_cache = build_ref_time(
                    batch_size,
                    args.n_ref,
                    args.ref_start,
                    args.ref_end,
                    device,
                    opt_t.dtype,
                )
            opt_ref_t = ref_time_cache

            should_tb_log = global_step % tb_log_interval == 0
            should_perf_log = should_tb_log
            step_cpu_start = time.perf_counter() if should_perf_log else None
            optimizer.zero_grad(set_to_none=True)
            need_cred_level = _model_requires_cred_level(model)

            with autocast(
                device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")
            ):
                g, z_l, h_l, H_gw = model.encode(
                    gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
                )
                extra_neg_z_itc = None
                extra_neg_event_time_mjd = None
                if has_negatives and bool(
                    getattr(args, "itc_extra_negative_enable", False)
                ):
                    neg_zero_time_mjd_for_itc = _neg_zero_time_mjd_cls_base
                    if neg_zero_time_mjd_for_itc is None:
                        neg_zero_time_mjd_for_itc = _neg_zero_time_mjd_base
                    with torch.no_grad():
                        extra_neg_z_itc, _ = model.encode_optical(
                            neg_coords,
                            neg_t,
                            neg_v,
                            opt_ref_t,
                            neg_mask,
                            neg_err,
                        )
                    extra_neg_event_time_mjd = neg_zero_time_mjd_for_itc
                itc_g = g[itc_pair_mask]
                itc_z_l = z_l[itc_pair_mask]
                itc_gw_indices = gw_indices[itc_pair_mask]
                itc_gw_event_time = (
                    batch_event_time_mjd[itc_pair_mask]
                    if batch_event_time_mjd is not None
                    else None
                )
                itc_opt_first_detection = (
                    _opt_first_detection_mjd[itc_pair_mask]
                    if _opt_first_detection_mjd is not None
                    else None
                )
                if args.itc_loss_type == "supcon":
                    itc_loss, sim_g2o, sim_g2o_for_loss = model.compute_supcon_loss(
                        itc_g,
                        itc_z_l,
                        itc_gw_indices,
                        margin=args.supcon_margin,
                        gw_event_time_mjd=itc_gw_event_time,
                        opt_event_time_mjd=resolve_optical_candidate_time_mjd(
                            itc_opt_first_detection, itc_gw_event_time
                        ),
                        extra_neg_z=extra_neg_z_itc,
                        extra_neg_opt_event_time_mjd=extra_neg_event_time_mjd,
                    )
                else:
                    itc_loss, sim_g2o, sim_g2o_for_loss = model.compute_itc_loss(
                        itc_g,
                        itc_z_l,
                        itc_gw_indices,
                        gw_event_time_mjd=itc_gw_event_time,
                        opt_event_time_mjd=resolve_optical_candidate_time_mjd(
                            itc_opt_first_detection, itc_gw_event_time
                        ),
                        extra_neg_z=extra_neg_z_itc,
                        extra_neg_opt_event_time_mjd=extra_neg_event_time_mjd,
                    )

                pos_mask = ~pair_is_neg_gw
                neg_gw_mask = pair_is_neg_gw
                labels_pos = pos_mask.to(dtype=torch.long)
                labels_neg = torch.zeros(batch_size, device=device, dtype=torch.long)
                gallery_loss = torch.zeros((), device=device)
                gallery_hard_loss = torch.zeros((), device=device)
                gallery_hard_weight = 0.0

                if cls_branch_enabled:
                    # Compute per-pair credible level for dual fusion
                    cred_level = (
                        compute_credible_level(gw_m, opt_coords)
                        if need_cred_level
                        else None
                    )
                    if (
                        _opt_first_detection_mjd is not None
                        and batch_event_time_mjd is not None
                    ):
                        dt_pos = compute_time_delta_days(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        )
                    else:
                        dt_pos = torch.zeros(
                            (batch_size,), device=device, dtype=torch.float32
                        )
                    logits_pos = model.fusion_logits(
                        g,
                        h_l,
                        z_l=z_l,
                        H_gw=H_gw,
                        cred_level=cred_level,
                        gw_s=gw_s,
                        gw_m=gw_m,
                        opt_coords=opt_coords,
                        dt_days=dt_pos,
                    )
                    if pos_mask.any():
                        aligned_pos_loss = model.cls_criterion(
                            logits_pos[pos_mask],
                            torch.ones(
                                int(pos_mask.sum()), device=device, dtype=torch.long
                            ),
                        )
                    else:
                        aligned_pos_loss = None
                    if neg_gw_mask.any():
                        neg_gw_loss = model.cls_criterion(
                            logits_pos[neg_gw_mask],
                            torch.zeros(
                                int(neg_gw_mask.sum()), device=device, dtype=torch.long
                            ),
                        )
                    else:
                        neg_gw_loss = None

                    mis_idx = sample_mismatched_negatives(
                        batch_size,
                        device,
                        samples_per_gw=args.samples_per_gw,
                        gw_indices=gw_indices,
                        source_types=batch_source_types,
                        eligible_candidate_mask=itc_pair_mask,
                    )

                    if mis_idx is None:
                        hard_loss = torch.zeros((), device=device)
                        logits_hard = torch.zeros(
                            (batch_size, 2), device=device, dtype=g.dtype
                        )
                        dt_mis = None
                    else:
                        h_l_mis = h_l[mis_idx].clone()
                        z_l_mis = z_l[mis_idx].clone()
                        coords_mis = opt_coords[mis_idx].clone()
                        dt_mis = build_mismatched_classification_time_deltas(
                            mode=args.cls_distractor_time_mode,
                            batch_size=batch_size,
                            device=device,
                            window_days=args.mis_neg_dt_window_days,
                            candidate_indices=mis_idx,
                            opt_first_detection_mjd=_opt_first_detection_mjd,
                            parent_gw_event_time_mjd=batch_event_time_mjd,
                        )

                        cred_level_mis = (
                            compute_credible_level(gw_m, coords_mis)
                            if need_cred_level
                            else None
                        )
                        logits_hard = model.fusion_logits(
                            g,
                            h_l_mis,
                            z_l=z_l_mis,
                            H_gw=H_gw,
                            cred_level=cred_level_mis,
                            gw_s=gw_s,
                            gw_m=gw_m,
                            opt_coords=coords_mis,
                            dt_days=dt_mis,
                        )
                        hard_loss = model.cls_criterion(logits_hard, labels_neg)
                    gallery_extra_h_l = None
                    gallery_extra_z_l = None
                    gallery_extra_coords = None
                    gallery_extra_event_time = None
                    if has_negatives:
                        z_l_neg, h_l_neg = model.encode_optical(
                            neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err
                        )
                        cred_level_neg = (
                            compute_credible_level(gw_m, neg_coords)
                            if need_cred_level
                            else None
                        )
                        neg_zero_time_mjd_for_cls = _neg_zero_time_mjd_cls_base
                        if neg_zero_time_mjd_for_cls is None:
                            neg_zero_time_mjd_for_cls = _neg_zero_time_mjd_base
                        neg_event_time = (
                            neg_zero_time_mjd_for_cls.to(
                                device=device, dtype=torch.float32
                            )
                            if neg_zero_time_mjd_for_cls is not None
                            else None
                        )
                        dt_neg = build_external_classification_time_deltas(
                            mode=args.cls_distractor_time_mode,
                            positive_dt_pool=dt_pos[itc_pair_mask],
                            n_samples=batch_size,
                            empirical_fraction=args.cls_external_empirical_dt_fraction,
                            window_days=args.mis_neg_dt_window_days,
                            negative_event_time_mjd=neg_event_time,
                            query_event_time_mjd=batch_event_time_mjd,
                            device=device,
                        )
                        logits_neg = model.fusion_logits(
                            g,
                            h_l_neg,
                            z_l=z_l_neg,
                            H_gw=H_gw,
                            cred_level=cred_level_neg,
                            gw_s=gw_s,
                            gw_m=gw_m,
                            opt_coords=neg_coords,
                            dt_days=dt_neg,
                        )
                        neg_loss = model.cls_criterion(logits_neg, labels_neg)
                        gallery_extra_h_l = h_l_neg
                        gallery_extra_z_l = z_l_neg
                        gallery_extra_coords = neg_coords
                        gallery_extra_event_time = neg_event_time
                    else:
                        neg_loss = None

                    cls_loss = compute_weighted_cls_loss(
                        aligned_pos_loss, neg_gw_loss, hard_loss, neg_loss, args
                    )

                    gallery_loss = torch.zeros((), device=device)
                    gallery_hard_loss = torch.zeros((), device=device)
                    gallery_hard_weight = 0.0
                    total_loss = itc_weight * itc_loss + cls_weight * cls_loss
                else:
                    logits_pos = torch.zeros(
                        (batch_size, 2), device=device, dtype=g.dtype
                    )
                    logits_hard = torch.zeros(
                        (batch_size, 2), device=device, dtype=g.dtype
                    )
                    logits_neg = (
                        torch.zeros((batch_size, 2), device=device, dtype=g.dtype)
                        if has_negatives
                        else None
                    )
                    aligned_pos_loss = torch.zeros((), device=device)
                    neg_gw_loss = None
                    hard_loss = torch.zeros((), device=device)
                    neg_loss = torch.zeros((), device=device) if has_negatives else None
                    cls_loss = torch.zeros((), device=device)
                    gallery_loss = torch.zeros((), device=device)
                    gallery_hard_loss = torch.zeros((), device=device)
                    gallery_hard_weight = 0.0
                    dt_mis = None
                    if is_retrieval_active(args, epoch) and has_negatives:
                        z_l_neg, h_l_neg = model.encode_optical(
                            neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err
                        )
                        gallery_extra_h_l = h_l_neg
                        gallery_extra_z_l = z_l_neg
                        gallery_extra_coords = neg_coords
                        neg_event_time = (
                            _neg_zero_time_mjd_cls_base.to(
                                device=device, dtype=torch.float32
                            )
                            if _neg_zero_time_mjd_cls_base is not None
                            else (
                                _neg_zero_time_mjd_base.to(
                                    device=device, dtype=torch.float32
                                )
                                if _neg_zero_time_mjd_base is not None
                                else None
                            )
                        )
                        gallery_extra_event_time = neg_event_time
                    else:
                        gallery_extra_h_l = None
                        gallery_extra_z_l = None
                        gallery_extra_coords = None
                        gallery_extra_event_time = None
                    total_loss = itc_weight * itc_loss

                # --- Shared gallery computation (independent of CLS branch) ---
                if is_retrieval_active(args, epoch) and itc_pair_mask.any():
                    gallery_query_g = g[itc_pair_mask]
                    gallery_query_H = H_gw[itc_pair_mask] if H_gw is not None else None
                    gallery_query_gw_s = gw_s[itc_pair_mask]
                    gallery_query_gw_m = gw_m[itc_pair_mask]
                    gallery_query_event_time = (
                        batch_event_time_mjd[itc_pair_mask]
                        if batch_event_time_mjd is not None
                        else None
                    )
                    h_candidates = [h_l[itc_pair_mask]]
                    z_candidates = [z_l[itc_pair_mask]]
                    coords_candidates = [opt_coords[itc_pair_mask]]
                    candidate_gw_indices = [itc_gw_indices]
                    candidate_time_blocks = []
                    candidate_parent_time_blocks = []
                    if batch_event_time_mjd is not None:
                        pos_gallery_time = resolve_optical_candidate_time_mjd(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        )[itc_pair_mask].to(device=device, dtype=torch.float32)
                        candidate_time_blocks.append(pos_gallery_time)
                        candidate_parent_time_blocks.append(gallery_query_event_time)
                    if (
                        has_negatives
                        and bool(args.gallery_include_extra_negatives)
                        and gallery_extra_h_l is not None
                        and gallery_extra_z_l is not None
                        and gallery_extra_coords is not None
                    ):
                        h_candidates.append(gallery_extra_h_l)
                        z_candidates.append(gallery_extra_z_l)
                        coords_candidates.append(gallery_extra_coords)
                        candidate_gw_indices.append(
                            torch.full(
                                (batch_size,), -1, device=device, dtype=gw_indices.dtype
                            )
                        )
                        if batch_event_time_mjd is not None:
                            if gallery_extra_event_time is None:
                                candidate_time_blocks.append(
                                    torch.full(
                                        (batch_size,),
                                        float("nan"),
                                        device=device,
                                        dtype=torch.float32,
                                    )
                                )
                            else:
                                candidate_time_blocks.append(
                                    gallery_extra_event_time.to(
                                        device=device, dtype=torch.float32
                                    )
                                )
                            candidate_parent_time_blocks.append(
                                torch.full(
                                    (batch_size,),
                                    float("nan"),
                                    device=device,
                                    dtype=torch.float32,
                                )
                            )
                    h_gallery = torch.cat(h_candidates, dim=0)
                    z_gallery = torch.cat(z_candidates, dim=0)
                    coords_gallery = torch.cat(coords_candidates, dim=0)
                    gallery_gw_indices = torch.cat(candidate_gw_indices, dim=0)
                    gallery_pos_mask_full = itc_gw_indices.unsqueeze(
                        1
                    ) == gallery_gw_indices.unsqueeze(0)
                    gallery_candidate_time = (
                        torch.cat(candidate_time_blocks, dim=0)
                        if candidate_time_blocks
                        else None
                    )
                    gallery_candidate_parent_time = (
                        torch.cat(candidate_parent_time_blocks, dim=0)
                        if candidate_parent_time_blocks
                        else None
                    )
                    dt_gallery = build_gallery_time_delta_matrix(
                        gallery_query_event_time,
                        gallery_candidate_time,
                        candidate_parent_event_time_mjd=gallery_candidate_parent_time,
                        positive_mask=gallery_pos_mask_full,
                        distractor_time_mode=args.gallery_distractor_time_mode,
                        distractor_time_window_days=args.gallery_distractor_time_window_days,
                    )
                    gallery_kn_negative_mask_full = None
                    gallery_nonkn_negative_mask_full = None
                    if args.gallery_candidate_mode == "mixed_kn_nonkn":
                        if (
                            gallery_extra_h_l is None
                            or gallery_extra_z_l is None
                            or gallery_query_event_time is None
                            or not candidate_time_blocks
                        ):
                            raise ValueError(
                                "mixed training gallery requires external non-KN "
                                "features and KN first-detection/event times."
                            )
                        positive_dt_days = compute_parent_relative_time_delta_days(
                            candidate_time_blocks[0], gallery_query_event_time
                        )
                        mixed = build_mixed_training_gallery_batch(
                            positive_parent_ids=itc_gw_indices,
                            h_positive=h_l[itc_pair_mask],
                            z_positive=z_l[itc_pair_mask],
                            coords_positive=opt_coords[itc_pair_mask],
                            positive_dt_days=positive_dt_days,
                            h_external=gallery_extra_h_l,
                            z_external=gallery_extra_z_l,
                            gallery_size=args.gallery_training_size,
                            kn_fraction=args.gallery_kn_distractor_fraction,
                            max_queries=args.max_gallery_queries,
                            nonkn_empirical_fraction=(
                                args.gallery_nonkn_empirical_fraction
                            ),
                            nonkn_window_days=args.gallery_distractor_time_window_days,
                        )
                        query_rows = mixed["query_rows"]
                        gallery_query_g = gallery_query_g[query_rows]
                        gallery_query_H = (
                            gallery_query_H[query_rows]
                            if gallery_query_H is not None
                            else None
                        )
                        gallery_query_gw_s = gallery_query_gw_s[query_rows]
                        gallery_query_gw_m = gallery_query_gw_m[query_rows]
                        gallery_query_event_time = gallery_query_event_time[query_rows]
                        h_gallery = mixed["h_candidates"]
                        z_gallery = mixed["z_candidates"]
                        coords_gallery = mixed["coords_candidates"]
                        dt_gallery = mixed["dt_days"]
                        gallery_pos_mask_full = mixed["positive_mask"]
                        gallery_kn_negative_mask_full = mixed["kn_negative_mask"]
                        gallery_nonkn_negative_mask_full = mixed["nonkn_negative_mask"]

                    # Query subsampling: randomly select a subset of queries to
                    # bound GPU memory when scoring against the full candidate gallery.
                    max_queries = int(getattr(args, "max_gallery_queries", 0) or 0)
                    n_q_total = int(gallery_query_g.size(0))
                    if max_queries > 0 and n_q_total > max_queries:
                        q_idx = torch.randperm(n_q_total, device=device)[:max_queries]
                        g_q = gallery_query_g[q_idx]
                        H_q = (
                            gallery_query_H[q_idx]
                            if gallery_query_H is not None
                            else None
                        )
                        gw_s_q = gallery_query_gw_s[q_idx]
                        gw_m_q = gallery_query_gw_m[q_idx]
                        dt_q = dt_gallery[q_idx, :] if dt_gallery is not None else None
                        gallery_pos_mask = gallery_pos_mask_full[q_idx, :]
                        gallery_kn_negative_mask = (
                            gallery_kn_negative_mask_full[q_idx, :]
                            if gallery_kn_negative_mask_full is not None
                            else None
                        )
                        gallery_nonkn_negative_mask = (
                            gallery_nonkn_negative_mask_full[q_idx, :]
                            if gallery_nonkn_negative_mask_full is not None
                            else None
                        )
                    else:
                        g_q = gallery_query_g
                        H_q = gallery_query_H
                        gw_s_q = gallery_query_gw_s
                        gw_m_q = gallery_query_gw_m
                        dt_q = dt_gallery
                        gallery_pos_mask = gallery_pos_mask_full
                        gallery_kn_negative_mask = gallery_kn_negative_mask_full
                        gallery_nonkn_negative_mask = gallery_nonkn_negative_mask_full

                    gallery_scores = compute_fusion_gallery_score_matrix(
                        model,
                        g=g_q,
                        H_gw=H_q,
                        gw_s=gw_s_q,
                        gw_m=gw_m_q,
                        h_candidates=h_gallery,
                        z_candidates=z_gallery,
                        opt_coords_candidates=coords_gallery,
                        dt_days_matrix=dt_q,
                        need_cred_level=need_cred_level,
                        chunk_size=args.gallery_score_chunk_size,
                    )
                    gallery_loss = compute_fusion_gallery_nce_loss(
                        gallery_scores, gallery_pos_mask
                    )
                    if gallery_loss_weight > 0.0:
                        total_loss = total_loss + gallery_loss_weight * gallery_loss
                    gallery_hard_weight = compute_gallery_hard_neg_weight(args, epoch)
                    if gallery_hard_weight > 0.0:
                        if gallery_kn_negative_mask is not None:
                            gallery_hard_loss = (
                                compute_fusion_gallery_stratified_hard_nce_loss(
                                    gallery_scores,
                                    gallery_pos_mask,
                                    gallery_kn_negative_mask,
                                    gallery_nonkn_negative_mask,
                                    topk=int(args.gallery_hard_neg_topk),
                                    kn_fraction=args.gallery_kn_distractor_fraction,
                                )
                            )
                        else:
                            gallery_hard_loss = compute_fusion_gallery_hard_nce_loss(
                                gallery_scores,
                                gallery_pos_mask,
                                topk=int(args.gallery_hard_neg_topk),
                            )
                        total_loss = total_loss + (
                            gallery_loss_weight
                            * gallery_hard_weight
                            * gallery_hard_loss
                        )

            scaler.scale(total_loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=args.grad_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
            if lr_scheduler is not None:
                lr_scheduler.step()
            clamp_temperature(model, args)

            total_val = total_loss.item()
            itc_val = itc_loss.item()
            aligned_pos_loss_val = aligned_pos_loss.item()
            neg_gw_loss_val = neg_gw_loss.item() if neg_gw_loss is not None else 0.0
            hard_loss_val = hard_loss.item()
            if has_negatives:
                neg_loss_val = neg_loss.item()
            cls_val = cls_loss.item()
            gallery_loss_val = gallery_loss.item()
            gallery_hard_loss_val = gallery_hard_loss.item()
            epoch_total += total_val
            epoch_itc += itc_val
            epoch_cls += cls_val

            # Detach similarity matrix to prevent gradient graph retention
            sim_g2o_detached = sim_g2o.detach()
            with torch.no_grad():
                # ITC accuracy: accept any optical sample from the same GW event
                itc_preds = sim_g2o_detached.argmax(dim=1)
                pred_gw = itc_gw_indices[itc_preds]
                itc_acc = (itc_gw_indices == pred_gw).float().mean().item()

                if cls_branch_enabled:
                    pos_acc = 0.0
                    neg_gw_acc = 0.0
                    if pos_mask.any():
                        pos_acc = (
                            (logits_pos[pos_mask].argmax(dim=1) == 1)
                            .float()
                            .mean()
                            .item()
                        )
                    if neg_gw_mask.any():
                        neg_gw_acc = (
                            (logits_pos[neg_gw_mask].argmax(dim=1) == 0)
                            .float()
                            .mean()
                            .item()
                        )
                    hard_acc = (
                        (logits_hard.argmax(dim=1) == labels_neg).float().mean().item()
                    )
                    neg_acc = None
                    if has_negatives:
                        neg_acc = (
                            (logits_neg.argmax(dim=1) == labels_neg)
                            .float()
                            .mean()
                            .item()
                        )
                else:
                    pos_acc = 0.0
                    neg_gw_acc = 0.0
                    hard_acc = 0.0
                    neg_acc = 0.0 if has_negatives else None
                current_temp = model.log_temp.exp().item()

            step_time_ms = 0.0
            if should_perf_log and step_cpu_start is not None:
                step_time_ms = (time.perf_counter() - step_cpu_start) * 1000.0

            if should_tb_log:
                writer.add_scalar("Train/Batch_Total_Loss", total_val, global_step)
                writer.add_scalar("Train/Batch_ITC_Loss", itc_val, global_step)
                writer.add_scalar("Train/Batch_CLS_Loss", cls_val, global_step)
                writer.add_scalar(
                    "Train/Batch_Gallery_Loss", gallery_loss_val, global_step
                )
                writer.add_scalar(
                    "Train/Batch_GalleryHard_Loss", gallery_hard_loss_val, global_step
                )
                writer.add_scalar(
                    "Train/Batch_GalleryHard_Weight", gallery_hard_weight, global_step
                )
                writer.add_scalar(
                    "Train/Batch_AlignedPos_Loss", aligned_pos_loss_val, global_step
                )
                writer.add_scalar(
                    "Train/Batch_NegGW_Loss", neg_gw_loss_val, global_step
                )
                writer.add_scalar(
                    "Train/Batch_HardNeg_Loss", hard_loss_val, global_step
                )
                if has_negatives:
                    writer.add_scalar(
                        "Train/Batch_ExtraNeg_Loss", neg_loss_val, global_step
                    )
                writer.add_scalar("Train/Batch_ITC_Acc", itc_acc, global_step)
                writer.add_scalar("Train/Batch_AlignedPos_Acc", pos_acc, global_step)
                writer.add_scalar("Train/Batch_NegGW_Acc", neg_gw_acc, global_step)
                writer.add_scalar("Train/Batch_HardNeg_Acc", hard_acc, global_step)
                if has_negatives and neg_acc is not None:
                    writer.add_scalar("Train/Batch_ExtraNeg_Acc", neg_acc, global_step)
                writer.add_scalar("Train/Itc_Weight", itc_weight, global_step)
                writer.add_scalar("Train/Cls_Weight", cls_weight, global_step)
                writer.add_scalar("Train/Temperature", current_temp, global_step)
                writer.add_scalar(
                    "Train/Learning_Rate", optimizer.param_groups[0]["lr"], global_step
                )
                if dt_mis is not None and batch_event_time_mjd is not None:
                    dt_mis_mean, dt_mis_std = summarize_time_delta(dt_mis)
                    writer.add_scalar(
                        "Train/TimeShift/mis_delta_days_mean", dt_mis_mean, global_step
                    )
                    writer.add_scalar(
                        "Train/TimeShift/mis_delta_days_std", dt_mis_std, global_step
                    )
                    mis_q = summarize_abs_time_delta_quantiles(
                        dt_mis, quantiles=(0.5, 0.9, 0.95)
                    )
                    writer.add_scalar(
                        "Train/TimeShift/mis_delta_abs_p50", mis_q["p50"], global_step
                    )
                    writer.add_scalar(
                        "Train/TimeShift/mis_delta_abs_p90", mis_q["p90"], global_step
                    )
                    writer.add_scalar(
                        "Train/TimeShift/mis_delta_abs_p95", mis_q["p95"], global_step
                    )
                writer.add_scalar("Train/Perf/step_time_ms", step_time_ms, global_step)

            if batch_idx % 100 == 0:
                postfix = {
                    "Total": f"{total_val:.4f}",
                    "ITC": f"{itc_val:.4f}",
                    "CLS": f"{cls_val:.4f}",
                    "ITC_Acc": f"{itc_acc:.2f}",
                }
                if dt_mis is not None and batch_event_time_mjd is not None:
                    dt_mis_mean, _ = summarize_time_delta(dt_mis)
                    postfix["mis_dt"] = f"{dt_mis_mean:.2f}"
                if args.gallery_loss_weight > 0:
                    postfix["Gallery"] = f"{gallery_loss_val:.4f}"
                    if gallery_hard_weight > 0:
                        postfix["GalleryHard"] = f"{gallery_hard_loss_val:.4f}"
                if last_selection_score is not None:
                    postfix[f"{args.best_ckpt_metric}"] = f"{last_selection_score:.4f}"
                pbar.set_postfix(postfix)

            # Memory cleanup: delete large tensors to free memory
            del g, z_l, h_l, sim_g2o, sim_g2o_detached, itc_loss, total_loss
            del (
                logits_pos,
                logits_hard,
                aligned_pos_loss,
                neg_gw_loss,
                hard_loss,
                cls_loss,
                gallery_loss,
                gallery_hard_loss,
            )
            if has_negatives:
                del logits_neg, neg_loss

            # Periodic aggressive memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            global_step += 1

        # End of epoch cleanup
        pbar.close()
        del pbar

        avg_total = epoch_total / len(train_loader)
        avg_itc = epoch_itc / len(train_loader)
        avg_cls = epoch_cls / len(train_loader)
        print(
            f"Epoch {epoch+1} Complete. Avg Total: {avg_total:.4f} | "
            f"ITC: {avg_itc:.4f} | CLS: {avg_cls:.4f}"
        )

        writer.add_scalar("Train/Epoch_Total_Loss", avg_total, epoch)
        writer.add_scalar("Train/Epoch_ITC_Loss", avg_itc, epoch)
        writer.add_scalar("Train/Epoch_CLS_Loss", avg_cls, epoch)

        # Flush tensorboard to prevent memory accumulation
        writer.flush()

        stop_early = False
        if val_loader is not None and should_run_validation(
            epoch, args.epochs, args.validation_gallery_eval_interval
        ):
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args,
                epoch,
                amp_dtype=amp_dtype,
                gw_event_time_mjd_table=gw_event_time_mjd_table,
                gw_neg_type_table=gw_neg_type_table,
                gw_source_type_table=gw_source_type_table,
            )
            _close_loader_dataset_handles(
                val_loader,
                cache_in_memory=bool(args.cache_in_memory),
                label="Validation DataLoader",
            )
            if val_metrics is not None:
                if validation_gallery_context is not None:
                    hard_gallery_metrics = validation_gallery_context.evaluate(
                        model,
                        args,
                        device,
                        amp_dtype,
                        gw_event_time_mjd_table,
                    )
                    val_metrics["hard_gallery"] = hard_gallery_metrics
                print(
                    f"Val Avg Total: {val_metrics['total']:.4f} | "
                    f"ITC: {val_metrics['itc']:.4f} | CLS: {val_metrics['cls']:.4f}"
                )
                writer.add_scalar("Val/Epoch_Total_Loss", val_metrics["total"], epoch)
                writer.add_scalar("Val/Epoch_ITC_Loss", val_metrics["itc"], epoch)
                writer.add_scalar("Val/Epoch_CLS_Loss", val_metrics["cls"], epoch)
                writer.add_scalar(
                    "Val/Epoch_Gallery_Loss", val_metrics.get("gallery", 0.0), epoch
                )
                writer.add_scalar(
                    "Val/Epoch_GalleryHard_Loss",
                    val_metrics.get("gallery_hard", 0.0),
                    epoch,
                )
                # Retrieval metrics
                ret = val_metrics.get("retrieval", {})
                for k, v in ret.items():
                    writer.add_scalar(f"Val/Retrieval/{k}", v, epoch)

                # Classification metrics
                cls_m = val_metrics.get("classification", {})
                for k in ("auroc", "auprc", "f1_optimal", "f1_threshold", "ece"):
                    if k in cls_m:
                        writer.add_scalar(f"Val/Classification/{k}", cls_m[k], epoch)
                for k in (
                    "acc_aligned",
                    "acc_neg_gw",
                    "acc_mismatched",
                    "acc_external",
                    "acc_total",
                ):
                    if k in cls_m:
                        writer.add_scalar(f"Val/Classification/{k}", cls_m[k], epoch)

                # Embedding quality metrics
                emb = val_metrics.get("embedding", {})
                for k, v in emb.items():
                    writer.add_scalar(f"Val/Embedding/{k}", v, epoch)
                dt_m = val_metrics.get("time_delta", {})
                for key, value in dt_m.items():
                    writer.add_scalar(f"Val/TimeShift/{key}", value, epoch)
                hard_gallery = val_metrics.get("hard_gallery", {})
                if hard_gallery:
                    writer.add_scalar(
                        "Val/HardGallery/MacroMRR",
                        hard_gallery["macro_mrr"],
                        epoch,
                    )
                    writer.add_scalar(
                        "Val/HardGallery/MacroR1",
                        hard_gallery["macro_recall_at_1"],
                        epoch,
                    )
                    writer.add_scalar(
                        "Val/HardGallery/SelectionScore",
                        hard_gallery["selection_score"],
                        epoch,
                    )
                    for source, source_metrics in hard_gallery["by_source"].items():
                        for key, value in source_metrics.items():
                            writer.add_scalar(
                                f"Val/HardGallery/{source}/{key}",
                                value,
                                epoch,
                            )
                    for gallery_size in args.validation_gallery_sizes:
                        bns = hard_gallery["by_source"]["bns"]
                        nsbh = hard_gallery["by_source"]["nsbh"]
                        print(
                            f"  HardGallery G={gallery_size}: "
                            f"BNS R@1={bns[f'gallery_{gallery_size}_recall_at_1']:.4f} "
                            f"MRR={bns[f'gallery_{gallery_size}_mrr']:.4f} | "
                            f"NSBH R@1={nsbh[f'gallery_{gallery_size}_recall_at_1']:.4f} "
                            f"MRR={nsbh[f'gallery_{gallery_size}_mrr']:.4f}"
                        )
                    print(
                        "  HardGallery macro: "
                        f"R@1={hard_gallery['macro_recall_at_1']:.4f} "
                        f"MRR={hard_gallery['macro_mrr']:.4f} "
                        f"score={hard_gallery['selection_score']:.4f}"
                    )
                # Print summary of new metrics
                r1 = ret.get("g2o_recall_at_1", 0)
                r5 = ret.get("g2o_recall_at_5", 0)
                mrr = ret.get("g2o_mrr", 0)
                fr1 = ret.get("fusion_gallery_recall_at_1", 0)
                fmrr = ret.get("fusion_gallery_mrr", 0)
                auroc = cls_m.get("auroc", 0)
                auprc = cls_m.get("auprc", 0)
                align = emb.get("alignment", 0)
                print(
                    f"  Retrieval: R@1={r1:.4f} R@5={r5:.4f} MRR={mrr:.4f} | "
                    f"FusionGallery: R@1={fr1:.4f} MRR={fmrr:.4f} | "
                    f"CLS: AUROC={auroc:.4f} AUPRC={auprc:.4f} | "
                    f"Emb: align={align:.4f}"
                )

                current_selection_score = compute_ckpt_selection_score(
                    val_metrics, args.best_ckpt_metric
                )
                last_selection_score = float(current_selection_score)
                is_best_epoch = False
                writer.add_scalar(
                    f"Val/BestCkpt/{args.best_ckpt_metric}",
                    float(current_selection_score),
                    epoch,
                )

                neg_gw_guardrail = resolve_neg_gw_guardrail(args, val_metrics)
                guardrail_active = (
                    bool(neg_gw_guardrail["enabled"]) and train_neg_gw_ratio > 0.0
                )
                strata_m = val_metrics.get("neg_gw_strata", {})
                for key in NEG_GW_STRATA_KEYS:
                    item = strata_m.get(key)
                    if item:
                        writer.add_scalar(
                            f"Val/NegGW/{key}_recall",
                            float(item.get("recall", 0.0)),
                            epoch,
                        )
                        writer.add_scalar(
                            f"Val/NegGW/{key}_count",
                            int(item.get("count", 0)),
                            epoch,
                        )
                writer.add_scalar(
                    "Val/NegGW/MinRecall",
                    float(neg_gw_guardrail["min_recall"]),
                    epoch,
                )
                writer.add_scalar(
                    "Val/NegGW/GuardrailMet",
                    int(neg_gw_guardrail["met"]),
                    epoch,
                )

                if not best_selection_eligible:
                    print(
                        "  Best-CKPT selection pending: "
                        f"{describe_best_ckpt_selection_pending(args, epoch)}"
                    )
                else:
                    guardrail_blocked = guardrail_active and not neg_gw_guardrail["met"]
                    if guardrail_blocked:
                        reason = neg_gw_guardrail["reason"] or "unknown"
                        print(
                            "  Neg-GW guardrail not met; checkpoint selection and "
                            f"early stopping paused: {reason} "
                            f"(min recall={neg_gw_guardrail['min_recall']:.4f}, "
                            f"threshold={neg_gw_guardrail['threshold']:.4f})"
                        )
                        improved = False
                    else:
                        prev_best = best_val_score
                        improved = is_checkpoint_score_improved(
                            current_selection_score,
                            prev_best,
                            args.best_ckpt_min_delta,
                        )
                    if improved:
                        best_val_score = current_selection_score
                        best_epoch_idx = int(epoch)
                        is_best_epoch = True
                        retrieval_metrics = val_metrics.get("retrieval", {})
                        classification_metrics = val_metrics.get("classification", {})
                        best_val_metrics = {
                            "best_ckpt_metric": args.best_ckpt_metric,
                            "best_ckpt_score": current_selection_score,
                            "best_val_acc_total": classification_metrics.get(
                                "acc_total", 0
                            ),
                            "best_val_loss": val_metrics["total"],
                            "best_epoch": epoch,
                            "best_epoch_1based": epoch + 1,
                            "best_tracking_start_epoch": best_tracking_start_epoch,
                            "best_tracking_start_epoch_1based": (
                                best_tracking_start_epoch + 1
                                if best_tracking_start_epoch is not None
                                else None
                            ),
                            "val_itc_loss": val_metrics.get("itc", 0),
                            "val_cls_loss": val_metrics.get("cls", 0),
                            "val_recall_at_1": val_metrics.get("retrieval", {}).get(
                                "g2o_recall_at_1", 0
                            ),
                            "val_recall_at_5": val_metrics.get("retrieval", {}).get(
                                "g2o_recall_at_5", 0
                            ),
                            "val_mrr": val_metrics.get("retrieval", {}).get(
                                "g2o_mrr", 0
                            ),
                            "val_auroc": val_metrics.get("classification", {}).get(
                                "auroc", 0
                            ),
                            "val_auprc": val_metrics.get("classification", {}).get(
                                "auprc", 0
                            ),
                        }
                        for metric_name in sorted(FUSION_GALLERY_BEST_CKPT_METRICS):
                            best_val_metrics[f"val_{metric_name}"] = (
                                retrieval_metrics.get(metric_name, 0)
                            )
                        hard_gallery = val_metrics.get("hard_gallery", {})
                        if hard_gallery:
                            best_val_metrics["val_hard_gallery_macro_mrr"] = (
                                hard_gallery["macro_mrr"]
                            )
                            best_val_metrics["val_hard_gallery_macro_recall_at_1"] = (
                                hard_gallery["macro_recall_at_1"]
                            )
                            best_val_metrics[
                                "val_hard_gallery_macro_retrieval_score"
                            ] = hard_gallery["selection_score"]
                            best_val_metrics["val_hard_gallery"] = hard_gallery
                            if (
                                hard_gallery.get("mode") == "mixed_kn_nonkn"
                                and hard_gallery.get("condition")
                                == "training_aligned"
                            ):
                                best_val_metrics["val_mixed_gallery_macro_mrr"] = (
                                    hard_gallery["macro_mrr"]
                                )
                                best_val_metrics[
                                    "val_mixed_gallery_macro_recall_at_1"
                                ] = hard_gallery["macro_recall_at_1"]
                                best_val_metrics[
                                    "val_mixed_gallery_macro_retrieval_score"
                                ] = hard_gallery["selection_score"]
                                best_val_metrics["val_mixed_gallery"] = hard_gallery
                        best_val_metrics["neg_gw_strata"] = strata_m
                        best_val_metrics["neg_gw_guardrail"] = neg_gw_guardrail
                        best_val_metrics["val_neg_gw_min_recall"] = float(
                            neg_gw_guardrail["min_recall"]
                        )
                        best_val_metrics["val_neg_gw_guardrail_met"] = int(
                            bool(neg_gw_guardrail["met"])
                        )
                        epochs_no_improve = 0
                        best_ckpt = os.path.join(
                            args.ckpt_path, "ALBEF", "albef_best.pth"
                        )
                        os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)
                        save_training_checkpoint_atomic(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scaler_state_dict": scaler.state_dict(),
                                "acc_total": val_metrics.get("classification", {}).get(
                                    "acc_total", 0
                                ),
                                "selection_metric": args.best_ckpt_metric,
                                "selection_score": current_selection_score,
                                "selection_eligible": bool(best_selection_eligible),
                                "best_tracking_start_epoch": best_tracking_start_epoch,
                                "loss": val_metrics["total"],
                                "args": vars(args),
                                "dataset_window_metadata": getattr(
                                    args, "_dataset_window_metadata", None
                                ),
                                "effective_input_window_metadata": getattr(
                                    args, "_effective_input_window_metadata", None
                                ),
                                "validation_hard_gallery": val_metrics.get(
                                    "hard_gallery"
                                ),
                                "neg_gw_strata": strata_m,
                                "neg_gw_guardrail": neg_gw_guardrail,
                                "rng_state": capture_rng_state(),
                            },
                            best_ckpt,
                        )
                        print(
                            f"Saved best checkpoint ({args.best_ckpt_metric}="
                            f"{current_selection_score:.4f}, epoch={epoch+1}): {best_ckpt}"
                        )
                    else:
                        if guardrail_blocked:
                            print(
                                "  Neg-GW guardrail blocked; early stopping patience not consumed."
                            )
                        else:
                            epochs_no_improve += 1
                            if (
                                args.early_stop_patience > 0
                                and epochs_no_improve >= args.early_stop_patience
                            ):
                                print("Early stopping triggered.")
                                stop_early = True

                best_score_str = (
                    "N/A" if best_val_score is None else f"{best_val_score:.4f}"
                )
                best_epoch_str = (
                    "N/A" if best_epoch_idx is None else str(best_epoch_idx + 1)
                )
                print(
                    f"  Best-CKPT metric[{args.best_ckpt_metric}]: current={current_selection_score:.4f}, "
                    f"best={best_score_str}@epoch{best_epoch_str}, "
                    f"eligible={int(best_selection_eligible)}, is_best={int(is_best_epoch)}"
                )

        if not bool(getattr(args, "skip_epoch_checkpoints", False)):
            checkpoint_path = os.path.join(
                args.ckpt_path, "ALBEF", f"albef_epoch_{epoch+1}.pth"
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            save_training_checkpoint_atomic(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": avg_total,
                    "args": vars(args),
                    "dataset_window_metadata": getattr(
                        args, "_dataset_window_metadata", None
                    ),
                    "effective_input_window_metadata": getattr(
                        args, "_effective_input_window_metadata", None
                    ),
                },
                checkpoint_path,
            )

        if bool(getattr(args, "save_last_checkpoint", False)):
            last_checkpoint_path = os.path.join(
                args.ckpt_path, "ALBEF", "albef_last.pth"
            )
            os.makedirs(os.path.dirname(last_checkpoint_path), exist_ok=True)
            save_last_checkpoint_resilient(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "loss": avg_total,
                    "args": vars(args),
                    "rng_state": capture_rng_state(),
                    "training_state": {
                        "global_step": global_step,
                        "best_val_score": best_val_score,
                        "best_epoch_idx": best_epoch_idx,
                        "best_val_metrics": best_val_metrics,
                        "epochs_no_improve": epochs_no_improve,
                        "best_tracking_started": best_tracking_started,
                        "best_tracking_start_epoch": best_tracking_start_epoch,
                        "last_selection_score": last_selection_score,
                    },
                },
                last_checkpoint_path,
            )

        # End-of-epoch memory cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if stop_early:
            break

    if bool(args.validation_confirmation_gallery_enable) and best_val_metrics:
        validation_gallery_context = None
        gc.collect()
        confirmation_args = copy.copy(args)
        confirmation_args.validation_gallery_partition = "confirmation"
        confirmation_args.validation_gallery_queries_per_source = int(
            args.validation_confirmation_gallery_queries_per_source
        )
        confirmation_args.validation_gallery_trials = int(
            args.validation_confirmation_gallery_trials
        )
        validation_confirmation_gallery_context = ValidationGalleryContext.build(
            confirmation_args, val_gw_map
        )
        best_ckpt_path = os.path.join(args.ckpt_path, "ALBEF", "albef_best.pth")
        checkpoint = load_training_checkpoint(best_ckpt_path, map_location=device)
        migrate_time_embed_state_dict(checkpoint["model_state_dict"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        best_val_metrics["val_confirmation_hard_gallery"] = (
            validation_confirmation_gallery_context.evaluate(
                model, args, device, amp_dtype, gw_event_time_mjd_table
            )
        )

    _close_loader_dataset_handles(
        train_loader,
        cache_in_memory=bool(args.cache_in_memory),
        label="Train DataLoader",
    )
    _close_loader_dataset_handles(
        val_loader,
        cache_in_memory=bool(args.cache_in_memory),
        label="Validation DataLoader",
    )

    writer.close()

    # Write trial results JSON for HPO collection
    if best_val_metrics:
        best_val_metrics["final_epoch"] = last_completed_epoch
        best_val_metrics["dataset_window_metadata"] = getattr(
            args, "_dataset_window_metadata", None
        )
        best_val_metrics["effective_input_window_metadata"] = getattr(
            args, "_effective_input_window_metadata", None
        )
        result_path = os.path.join(args.ckpt_path, "ALBEF", "trial_results.json")
        summary_dir = os.path.dirname(result_path)
        os.makedirs(summary_dir, exist_ok=True)
        for summary_name in (
            "trial_results.json",
            "train_summary.json",
            "best_checkpoint_summary.json",
        ):
            summary_path = os.path.join(summary_dir, summary_name)
            write_json_atomic(summary_path, best_val_metrics)
        print(f"Trial results saved to: {result_path}")


if __name__ == "__main__":
    """
    Example usage:
    python Model/scripts/train/train.py --data_path data/LSST_KN_BNS/combined_dataset.h5 \
        --epochs 2 --batch_size 32 --steps_per_epoch 10 --ckpt_path data/model/checkpoints
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="training_data.h5")
    parser.add_argument("--neg_data_path", type=str, default=None)
    parser.add_argument("--neg_group", type=str, default="events/optical_data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Max norm for gradient clipping (0 to disable)",
    )
    parser.add_argument(
        "--lr_scheduler", type=str, default="none", choices=["none", "cosine"]
    )
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_schedule_total_epochs", type=int, default=None)
    parser.add_argument("--training_schedule_total_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--persistent_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--cache_in_memory", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--val_steps_per_epoch", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="Minimum improvement required on best_ckpt_metric to reset early stopping.",
    )
    parser.add_argument(
        "--best_ckpt_min_delta",
        type=float,
        default=None,
        help="Minimum checkpoint-score improvement; defaults to early_stop_min_delta.",
    )
    parser.add_argument(
        "--validation_gallery_eval_interval",
        type=int,
        default=1,
        help="Run validation every N epochs and always on the final epoch.",
    )
    parser.add_argument(
        "--neg_gw_guardrail_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require all four neg-GW validation strata recalls to pass before selecting checkpoints.",
    )
    parser.add_argument(
        "--neg_gw_guardrail_recall",
        type=float,
        default=0.9,
        help="Minimum per-stratum recall required by the neg-GW checkpoint guardrail.",
    )
    parser.add_argument(
        "--best_ckpt_metric",
        type=str,
        default="hard_gallery_macro_retrieval_score",
        choices=[
            "auprc",
            "auroc",
            "f1_optimal",
            "acc_total",
            "cls_composite_auprc_auroc",
            "g2o_recall_at_1",
            "g2o_recall_at_5",
            "g2o_mrr",
            "fusion_gallery_recall_at_1",
            "fusion_gallery_recall_at_5",
            "fusion_gallery_mrr",
            "hard_gallery_macro_retrieval_score",
            "mixed_gallery_macro_retrieval_score",
        ],
        help="In-domain validation metric used for selecting best checkpoint. Supports classification and retrieval metrics.",
    )
    parser.add_argument(
        "--best_ckpt_start_epoch",
        type=int,
        default=None,
        help=(
            "Optional 0-based lower bound for best-checkpoint selection. "
            "The automatic staged-training stability epoch always takes precedence."
        ),
    )
    parser.add_argument(
        "--val_split_stratify_by_source",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stratify the event-level train/validation split by GW source_type.",
    )
    parser.add_argument(
        "--validation_gallery_enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute fixed test-style hard-gallery validation every epoch.",
    )
    parser.add_argument(
        "--validation_gallery_mode",
        type=str,
        default="synthetic_time_sky_hard",
    )
    parser.add_argument(
        "--validation_gallery_condition",
        choices=["positive_shared", "training_aligned"],
        default="positive_shared",
        help="Coordinate condition for mixed KN/non-KN validation galleries.",
    )
    parser.add_argument(
        "--validation_gallery_kn_distractor_fraction",
        type=float,
        default=None,
        help="Fixed KN distractor fraction for validation; defaults to the training value.",
    )
    parser.add_argument(
        "--validation_gallery_nonkn_empirical_fraction",
        type=float,
        default=None,
        help=(
            "Fixed empirical-time fraction for non-KN validation distractors; "
            "defaults to the training value."
        ),
    )
    parser.add_argument(
        "--validation_gallery_sizes",
        type=parse_validation_gallery_sizes,
        default=[100, 500, 1000],
    )
    parser.add_argument("--validation_gallery_queries_per_source", type=int, default=64)
    parser.add_argument("--validation_gallery_trials", type=int, default=1)
    parser.add_argument("--validation_gallery_seed", type=int, default=42)
    parser.add_argument(
        "--validation_gallery_partition",
        choices=["all", "tune", "confirmation"],
        default="all",
    )
    parser.add_argument("--validation_gallery_tune_fraction", type=float, default=0.75)
    parser.add_argument("--validation_gallery_partition_seed", type=int, default=42)
    parser.add_argument(
        "--validation_gallery_size_weights",
        type=json.loads,
        default=None,
        help="JSON mapping from gallery size to objective weight.",
    )
    parser.add_argument(
        "--validation_confirmation_gallery_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--validation_confirmation_gallery_queries_per_source",
        type=int,
        default=128,
    )
    parser.add_argument("--validation_confirmation_gallery_trials", type=int, default=5)
    parser.add_argument(
        "--validation_gallery_time_window_days", type=float, default=30.0
    )
    parser.add_argument(
        "--validation_gallery_credible_level_max", type=float, default=0.9
    )
    parser.add_argument(
        "--validation_gallery_include_undersized",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--validation_gallery_mrr_weight", type=float, default=0.8)
    parser.add_argument(
        "--validation_gallery_recall_at_1_weight",
        type=float,
        default=0.2,
    )
    parser.add_argument("--n_ref", type=int, default=64)
    parser.add_argument("--ref_start", type=float, default=-0.3)
    parser.add_argument("--ref_end", type=float, default=0.6)
    parser.add_argument("--ref_dim", type=int, default=64)
    parser.add_argument("--enc_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument(
        "--optical_curve_dim",
        type=int,
        default=None,
        help="Physical-dual optical curve feature dimension (default: enc_dim).",
    )
    parser.add_argument(
        "--optical_coord_dim",
        type=int,
        default=None,
        help="Physical-dual optical coordinate feature dimension (default: enc_dim).",
    )
    parser.add_argument(
        "--optical_curve_hidden_dim",
        type=int,
        default=None,
        help="Hidden size for residual optical curve refiner; <=0 disables it.",
    )
    parser.add_argument(
        "--contrastive_hidden_dim",
        type=int,
        default=None,
        help="Hidden size for physical-dual contrastive heads (default: max(input_dim, proj_dim)).",
    )
    parser.add_argument("--fusion_attn_dim", type=int, default=None)
    parser.add_argument("--fusion_hidden_dim", type=int, default=None)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for classification loss (0 to disable)",
    )
    parser.add_argument("--temp_init", type=float, default=0.07)
    parser.add_argument("--temp_final", type=float, default=None)
    parser.add_argument("--temp_min", type=float, default=0.01)
    parser.add_argument("--temp_max", type=float, default=100.0)
    parser.add_argument(
        "--temp_schedule",
        type=str,
        default="learned",
        choices=["learned", "fixed", "cosine"],
    )
    parser.add_argument(
        "--time_compat_weight",
        type=float,
        default=0.6,
        help="Weight for time-compatibility penalty on ITC logits (<=0 to disable).",
    )
    parser.add_argument(
        "--time_compat_tau_days",
        type=float,
        default=30.0,
        help="Time scale tau (days) for time-compatibility penalty.",
    )
    parser.add_argument(
        "--time_compat_power",
        type=float,
        default=2.0,
        help="Power for |Δt/tau|^power in time-compatibility penalty.",
    )
    parser.add_argument(
        "--time_compat_max_penalty",
        type=float,
        default=8.0,
        help="Maximum absolute logit penalty for time compatibility.",
    )
    parser.add_argument(
        "--use_time_delta_cls_feature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append normalized |delta time| to the physical_dual_hgw classifier input.",
    )
    parser.add_argument(
        "--time_delta_cls_scale_days",
        type=float,
        default=None,
        help="Scale in days for the CLS time-delta feature (default: time_compat_tau_days).",
    )
    parser.add_argument(
        "--time_delta_cls_clip",
        type=float,
        default=10.0,
        help="Maximum normalized |delta time| value appended to the CLS fusion head.",
    )
    parser.add_argument(
        "--fusion_physical_weight",
        type=float,
        default=1.0,
        help="Fixed multiplier for the physical-consistency fused feature.",
    )
    parser.add_argument(
        "--fusion_spatial_weight",
        type=float,
        default=1.0,
        help="Fixed multiplier for the spatial-consistency fused feature.",
    )
    parser.add_argument(
        "--mtan_snr_s0",
        type=float,
        default=3.0,
        help="mTAN SNR compatibility threshold s0.",
    )
    parser.add_argument(
        "--mtan_snr_beta",
        type=float,
        default=1.0,
        help="mTAN SNR compatibility slope beta.",
    )
    parser.add_argument(
        "--mtan_snr_clip_min",
        type=float,
        default=-8.0,
        help="mTAN SNR clipping lower bound.",
    )
    parser.add_argument(
        "--mtan_snr_clip_max",
        type=float,
        default=20.0,
        help="mTAN SNR clipping upper bound.",
    )
    parser.add_argument(
        "--mtan_snr_eps",
        type=float,
        default=1e-9,
        help="Numerical epsilon for mTAN SNR denominator.",
    )
    parser.add_argument(
        "--mtan_lupt_psfflux_zp",
        type=float,
        default=None,
        help="Optional override for luptitude psfFlux zero point used by mTAN.",
    )
    parser.add_argument(
        "--mtan_lupt_k",
        type=float,
        default=None,
        help="Optional override for luptitude softening scale k used by mTAN.",
    )
    parser.add_argument(
        "--mtan_lupt_m5_mag",
        type=str,
        default=None,
        help="Optional override: 6 comma-separated m5 mags in order u,g,r,i,z,Y.",
    )
    parser.add_argument(
        "--nonkn_cls_base_field",
        type=str,
        default="zero_time_mjd_cls_base",
        help="Negative H5 field used as base absolute time for extra-negative dt.",
    )
    parser.add_argument("--gw_dropout", type=float, default=0.1)
    parser.add_argument("--opt_dropout", type=float, default=0.1)
    parser.add_argument(
        "--proj_dropout",
        type=float,
        default=0.0,
        help="Projection head dropout to prevent ITC overfitting (default: 0.0)",
    )
    parser.add_argument(
        "--feature_dropout",
        type=float,
        default=0.0,
        help="Dropout applied to encoder features before ITC/CLS heads",
    )
    parser.add_argument("--itc_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument(
        "--cls_pos_weight",
        type=float,
        default=1.0,
        help="Fallback positive class weight for CLS loss (legacy)",
    )
    parser.add_argument(
        "--cls_neg_weight",
        type=float,
        default=1.0,
        help="Fallback negative class weight for CLS loss (legacy)",
    )
    parser.add_argument(
        "--cls_extra_neg_weight",
        type=float,
        default=1.0,
        help="Fallback extra negative class weight for CLS loss (legacy)",
    )
    parser.add_argument(
        "--cls_aligned_pos_weight",
        type=float,
        default=0.5,
        help="Aligned positive GW->own-KN bucket weight for CLS loss",
    )
    parser.add_argument(
        "--cls_neg_gw_weight",
        type=float,
        default=0.2,
        help="neg-GW->source-matched KN bucket weight for CLS loss",
    )
    parser.add_argument(
        "--cls_mismatched_weight",
        type=float,
        default=0.15,
        help="GW->other-GW optical mismatched bucket weight for CLS loss",
    )
    parser.add_argument(
        "--cls_external_neg_weight",
        type=float,
        default=0.15,
        help="GW->external non-KN optical bucket weight for CLS loss",
    )
    parser.add_argument(
        "--neg_gw_pair_ratio",
        type=float,
        default=0.2,
        help="Training batch fraction of classification-only negative-GW/positive-optical pairs.",
    )
    parser.add_argument(
        "--cls_ramp_epochs",
        type=int,
        default=0,
        help="Epochs to ramp CLS weight from 0 to cls_weight (0 to disable)",
    )
    parser.add_argument(
        "--staged_training_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable A/B/C staged training with frozen-head warm-up and differential LR.",
    )
    parser.add_argument(
        "--stage_itc_epochs",
        type=int,
        default=8,
        help="ITC-only epochs in sequential_loss mode.",
    )
    parser.add_argument(
        "--stage_cls_ramp_epochs",
        type=int,
        default=4,
        help="CLS introduction/ramp epochs in sequential_loss mode.",
    )
    parser.add_argument(
        "--stage_retrieval_ramp_epochs",
        type=int,
        default=4,
        help="Retrieval introduction/ramp epochs in sequential_loss mode.",
    )
    parser.add_argument(
        "--stage_joint_itc_start_weight",
        type=float,
        default=0.5,
        help="ITC weight fraction at the start of joint fine-tuning.",
    )
    parser.add_argument(
        "--stage_joint_itc_end_weight",
        type=float,
        default=0.25,
        help="ITC weight fraction at the end of joint fine-tuning.",
    )
    parser.add_argument(
        "--encoder_lr_ratio",
        type=float,
        default=0.1,
        help="Encoder LR multiplier relative to head LR during joint fine-tuning.",
    )
    parser.add_argument(
        "--retrieval_start_epoch",
        type=int,
        default=0,
        help="Epoch to start retrieval/gallery training (0 = start from epoch 0).",
    )
    parser.add_argument(
        "--gallery_loss_weight",
        type=float,
        default=0.0,
        help="Weight for batch-level fusion mini-gallery retrieval loss (0 to disable).",
    )
    parser.add_argument(
        "--gallery_loss_ramp_epochs",
        type=int,
        default=0,
        help="Epochs to ramp gallery loss from 0 to gallery_loss_weight after retrieval_start_epoch.",
    )
    parser.add_argument(
        "--compute_cls_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute validation pair/triplet classification metrics even when CLS loss is disabled.",
    )
    parser.add_argument(
        "--compute_fusion_gallery_metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute validation fusion-gallery retrieval metrics even when gallery loss is disabled.",
    )
    parser.add_argument(
        "--fusion_gallery_metrics_start_epoch",
        type=int,
        default=0,
        help="Epoch to start validation-only fusion-gallery metrics.",
    )
    parser.add_argument(
        "--gallery_score_chunk_size",
        type=int,
        default=8192,
        help="Approximate number of flattened query-candidate pairs scored per gallery chunk.",
    )
    parser.add_argument(
        "--max_gallery_queries",
        type=int,
        default=0,
        help="Max queries for fusion gallery scoring during training (0 = use all queries).",
    )
    parser.add_argument(
        "--gallery_candidate_mode",
        choices=["legacy", "mixed_kn_nonkn"],
        default="legacy",
        help="Candidate construction for the fusion training gallery.",
    )
    parser.add_argument(
        "--gallery_training_size",
        type=int,
        default=1000,
        help="Total candidates per query in mixed training galleries.",
    )
    parser.add_argument(
        "--gallery_kn_distractor_fraction",
        type=float,
        default=0.25,
        help="Fraction of mixed-gallery distractors that are KN light curves.",
    )
    parser.add_argument(
        "--gallery_candidate_coordinate_mode",
        choices=["native", "positive_shared"],
        default="native",
        help="Coordinate policy for training gallery candidates.",
    )
    parser.add_argument(
        "--gallery_kn_distractor_time_mode",
        choices=["parent_relative"],
        default="parent_relative",
        help="Time policy for KN distractors in mixed galleries.",
    )
    parser.add_argument(
        "--gallery_nonkn_distractor_time_mode",
        choices=["empirical_uniform_mixture"],
        default="empirical_uniform_mixture",
        help="Time policy for non-KN distractors in mixed galleries.",
    )
    parser.add_argument(
        "--gallery_nonkn_empirical_fraction",
        type=float,
        default=0.5,
        help="Empirical-KN fraction of mixed-gallery non-KN time offsets.",
    )
    parser.add_argument(
        "--gallery_include_extra_negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include external non-KN negatives in the fusion mini-gallery loss when available.",
    )
    parser.add_argument(
        "--gallery_distractor_time_mode",
        type=str,
        default="actual",
        choices=["actual", "synthetic_after_gw", "parent_relative"],
        help="Time-delta policy for non-matching fusion mini-gallery distractors.",
    )
    parser.add_argument(
        "--gallery_distractor_time_window_days",
        type=float,
        default=30.0,
        help="Upper bound in days for synthetic_after_gw gallery distractor dt sampling.",
    )
    parser.add_argument(
        "--gallery_hard_neg_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable top-k hard-negative mining inside the fusion mini-gallery retrieval loss.",
    )
    parser.add_argument(
        "--gallery_hard_neg_topk",
        type=int,
        default=32,
        help="Number of top-scoring negative candidates to keep per query in the hard gallery NCE term.",
    )
    parser.add_argument(
        "--gallery_hard_neg_weight",
        type=float,
        default=0.5,
        help="Weight of the hard-gallery NCE auxiliary loss term.",
    )
    parser.add_argument(
        "--gallery_hard_neg_start_after_retrieval_epochs",
        type=int,
        default=2,
        help="Epochs after retrieval_start_epoch to begin gallery hard-negative mining.",
    )
    parser.add_argument(
        "--gallery_hard_neg_ramp_epochs",
        type=int,
        default=2,
        help="Epochs to ramp gallery hard-negative weight from 0 to gallery_hard_neg_weight.",
    )
    parser.add_argument(
        "--itc_decay_start_epoch",
        type=int,
        default=0,
        help="Epoch to start decaying ITC weight (ignored if itc_decay_epochs <= 0)",
    )
    parser.add_argument(
        "--itc_decay_epochs",
        type=int,
        default=0,
        help="Epochs to decay ITC weight (0 to disable)",
    )
    parser.add_argument(
        "--itc_decay_ratio",
        type=float,
        default=0.0,
        help="Fractional decay of ITC weight by the end of itc_decay_epochs",
    )
    parser.add_argument(
        "--itc_extra_negative_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include external non-KN optical samples as ITC negatives (default: off).",
    )
    parser.add_argument(
        "--itc_label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing for ITC loss (0 to disable)",
    )
    parser.add_argument(
        "--itc_loss_type",
        type=str,
        default="infonce",
        choices=["infonce", "supcon"],
        help="ITC loss type: 'infonce' (original) or 'supcon' (supervised contrastive)",
    )
    parser.add_argument(
        "--supcon_margin",
        type=float,
        default=0.0,
        help="Margin for SupCon loss to enforce separation between positives and negatives (default: 0.0)",
    )
    parser.add_argument(
        "--samples_per_gw",
        type=int,
        default=4,
        help="Number of optical samples per GW event for SupCon (default: 4)",
    )
    parser.add_argument(
        "--min_lc_per_gw",
        type=int,
        default=2,
        help="Minimum light curves required for a GW to be eligible for SupCon",
    )
    parser.add_argument(
        "--mis_neg_dt_window_days",
        type=float,
        default=30.0,
        help="Synthetic time delta for in-batch mismatched negatives: Uniform(0, window_days) days. "
        "Models the realistic scenario where optical first detection follows GW by 0–window days.",
    )
    parser.add_argument(
        "--cls_distractor_time_mode",
        type=str,
        default="legacy",
        choices=["legacy", "mixed_empirical"],
        help=(
            "Time-delta policy for mismatched-KN and external classification "
            "negatives. mixed_empirical uses parent-relative KN delays and a "
            "mixture of empirical-positive and operational-uniform external dt."
        ),
    )
    parser.add_argument(
        "--cls_external_empirical_dt_fraction",
        type=float,
        default=0.5,
        help="Fraction of external-negative dt drawn from positive KN delays.",
    )
    parser.add_argument(
        "--hardneg_time_window_days",
        type=str,
        default="30,60,120",
        help="Comma-separated adaptive windows (days) for hard-negative mining.",
    )
    parser.add_argument(
        "--hardneg_min_candidates",
        type=int,
        default=4,
        help="Minimum candidate count before accepting current hard-negative time window.",
    )
    parser.add_argument(
        "--cls_start_epoch",
        type=int,
        default=0,
        help="Epoch to start CLS training. Before this epoch, only ITC loss is used (for staged training)",
    )
    parser.add_argument(
        "--use_lightweight_gw",
        action="store_true",
        help="Use lightweight GW encoder (~100K params) instead of ResNet-18 (~11M params) to prevent overfitting on small GW datasets",
    )
    parser.add_argument(
        "--dual_fusion",
        action="store_true",
        help="Use dual cross-attention fusion (optical→GW + GW→optical) with per-pair credible level",
    )
    parser.add_argument(
        "--fusion_mode",
        type=str,
        default=None,
        help="Fusion mode: legacy_g2o | legacy_dual | physical_dual_hgw | concat_proj. If unset, inferred from --dual_fusion.",
    )
    parser.add_argument(
        "--use_similarity_as_cls_input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append pair ITC similarity to classifier input in physical_dual_hgw mode.",
    )
    parser.add_argument(
        "--use_cred_level_feature",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append cred_level to classifier input in physical_dual_hgw mode.",
    )
    # GW数据增强参数
    parser.add_argument(
        "--gw_aug_noise",
        type=float,
        default=0.05,
        help="GW skymap augmentation noise std (default: 0.05)",
    )
    parser.add_argument(
        "--gw_aug_jitter",
        type=float,
        default=0.02,
        help="GW scalar augmentation jitter ratio (default: 0.02)",
    )
    parser.add_argument(
        "--gw_aug_dropout",
        type=float,
        default=0.1,
        help="GW channel dropout probability (default: 0.1)",
    )
    # Optical data augmentation parameters
    parser.add_argument(
        "--opt_aug_noise",
        type=float,
        default=0.0,
        help="Optical flux noise scale relative to errors (default: 0.0)",
    )
    parser.add_argument(
        "--opt_aug_time_jitter",
        type=float,
        default=0.0,
        help="Optical time jitter std (default: 0.0)",
    )
    parser.add_argument(
        "--opt_aug_dropout",
        type=float,
        default=0.0,
        help="Optical observation dropout probability (default: 0.0)",
    )
    parser.add_argument(
        "--opt_aug_band_dropout",
        type=float,
        default=0.0,
        help="Optical band dropout probability (default: 0.0)",
    )
    parser.add_argument(
        "--hpo_trial_number",
        type=int,
        default=None,
        help="Optuna trial number (set automatically by HPO, not for manual use)",
    )
    parser.add_argument(
        "--skip_epoch_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip per-epoch checkpoint files while still writing best checkpoint and summaries.",
    )
    parser.add_argument(
        "--save_last_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Overwrite albef_last.pth each epoch with resumable training state.",
    )
    parser.add_argument(
        "--default_json_config",
        type=str,
        default=None,
        help="Default JSON config file path. Loaded before --json_config.",
    )
    parser.add_argument(
        "--json_config",
        type=str,
        default=None,
        help="JSON config file path. Values set defaults; CLI args override them.",
    )

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--default_json_config", type=str, default=None)
    config_parser.add_argument("--json_config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()
    apply_json_config_defaults(
        parser,
        default_json_config=config_args.default_json_config,
        json_config=config_args.json_config,
    )

    args = parser.parse_args()

    if os.path.exists(args.data_path):
        if args.ckpt_path is None:
            raise ValueError("ckpt_path must be provided.")
        os.makedirs(args.ckpt_path, exist_ok=True)
        train(args)
    else:
        print("Data file not found.")
