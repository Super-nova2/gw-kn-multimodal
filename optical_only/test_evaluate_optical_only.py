#!/usr/bin/env python3
"""
Evaluation script for optical-only KN classifier (fd_t0 variant).

Supports optional K-quantile time-offset ensemble with logits averaging.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import (
    OpticalBinaryDataset,
    OpticalPrefixEvalDataset,
    _build_prefix_manifest_for_binary_dataset,
    _filter_indices_by_meta_constraints,
    _filter_indices_by_min_detection_count,
)
from metrics import compute_classification_metrics
from model import OpticalKNClassifier
from optical_prefix import parse_prefix_det_support


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
        raise ValueError("offset_eval_quantiles produced empty list.")
    return vals


def parse_relax_t_span_thresholds(value, default_train: int = 500000, default_eval: int = 20000) -> Dict[str, int]:
    out = {
        "train": int(default_train),
        "eval": int(default_eval),
    }
    if value is None:
        return out
    if isinstance(value, dict):
        for key in ("train", "eval"):
            if key in value and value[key] is not None:
                out[key] = int(value[key])
        return out
    text = str(value).strip()
    if text == "":
        return out
    if text.isdigit():
        n = int(text)
        out["train"] = n
        out["eval"] = n
        return out
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, raw_val = part.split(":", 1)
        elif "=" in part:
            key, raw_val = part.split("=", 1)
        else:
            raise ValueError(
                "meta_filter_relax_t_span_if_below_rows must be an int or "
                "'train:500000,eval:20000' style mapping."
            )
        key = key.strip().lower()
        if key not in out:
            raise ValueError(
                f"Unsupported key '{key}' in meta_filter_relax_t_span_if_below_rows; expected train/eval."
            )
        out[key] = int(raw_val.strip())
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate optical-only KN classifier (fd_t0)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--config", type=str, default=None, help="Optional JSON config for path fallback.")

    p.add_argument("--pos_data_path", type=str, default=None)
    p.add_argument("--neg_data_path", type=str, default=None)
    p.add_argument("--neg_group", type=str, default=None)

    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=0)
    p.add_argument("--prefetch_factor", type=int, default=2)

    p.add_argument("--max_pos_samples", type=int, default=100000)
    p.add_argument("--max_neg_samples", type=int, default=100000)
    p.add_argument("--sample_seed", type=int, default=42)

    p.add_argument("--target_recall", type=float, default=None)
    p.add_argument("--output_dir", type=str, default="eval_results/optical_only_fd_t0")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--prefix_eval_enable", action="store_true", default=None)
    p.add_argument("--prefix_min_det", type=int, default=None)
    p.add_argument("--prefix_eval_det_support", type=str, default=None)
    p.add_argument("--prefix_manifest_out", type=str, default=None)
    p.add_argument("--meta_filter_n_det_min", type=int, default=None)
    p.add_argument("--meta_filter_n_det_max", type=int, default=None)
    p.add_argument("--meta_filter_n_bands_max", type=int, default=None)
    p.add_argument("--meta_filter_t_span_max", type=float, default=None)
    p.add_argument("--meta_filter_relax_t_span_if_below_rows", type=str, default=None)
    p.add_argument("--real_stream_profile_path", type=str, default=None)

    p.add_argument("--time_offset_enable", action="store_true", default=None)
    p.add_argument("--offset_dist_npz", type=str, default=None)
    p.add_argument("--offset_dist_key", type=str, default=None)
    p.add_argument("--offset_eval_mode", type=str, default=None)
    p.add_argument("--offset_eval_quantiles", type=str, default=None)
    p.add_argument("--offset_scale_days_divisor", type=float, default=None)
    p.add_argument("--offset_seed", type=int, default=None)
    args = p.parse_args()
    if args.prefix_min_det is not None and int(args.prefix_min_det) < 1:
        raise ValueError("--prefix_min_det must be >= 1.")
    if args.prefix_eval_det_support is not None:
        parse_prefix_det_support(args.prefix_eval_det_support)
    if args.meta_filter_n_det_min is not None and args.meta_filter_n_det_max is not None:
        if int(args.meta_filter_n_det_min) > int(args.meta_filter_n_det_max):
            raise ValueError("--meta_filter_n_det_min must be <= --meta_filter_n_det_max.")
    if args.meta_filter_n_bands_max is not None and int(args.meta_filter_n_bands_max) < 1:
        raise ValueError("--meta_filter_n_bands_max must be >= 1.")
    if args.meta_filter_t_span_max is not None and float(args.meta_filter_t_span_max) <= 0:
        raise ValueError("--meta_filter_t_span_max must be > 0.")
    parse_relax_t_span_thresholds(args.meta_filter_relax_t_span_if_below_rows)
    return args


def load_json(path):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be JSON object: {path}")
    return data


def choose_value(cli_value, config_dict, ckpt_args, key, default=None):
    if cli_value is not None:
        return cli_value
    if key in config_dict and config_dict[key] is not None:
        return config_dict[key]
    if key in ckpt_args and ckpt_args[key] is not None:
        return ckpt_args[key]
    return default


def _preview_keys(keys: List[str], limit: int = 8) -> str:
    if not keys:
        return "[]"
    shown = keys[:limit]
    suffix = "" if len(keys) <= limit else ", ..."
    return "[" + ", ".join(shown) + suffix + "]"


def _default_neg_group_from_path(neg_data_path: str) -> str:
    # Keep v2 defaults without relying on old *_TRAIN groups.
    name = Path(neg_data_path).name.lower()
    if "tutorial_negative_dataset" in name:
        return "Tutorial/optical_data"
    return "ELASTICC2/optical_data"


def build_ref_time(batch_size, n_ref, ref_start, ref_end, device, dtype):
    ref = torch.linspace(ref_start, ref_end, n_ref, dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)


def apply_time_offsets(opt_t, opt_mask, delta_days, scale_divisor):
    valid = (opt_mask.sum(dim=-1) > 0).to(dtype=opt_t.dtype)
    shift = (delta_days.to(device=opt_t.device, dtype=opt_t.dtype) / float(scale_divisor)).unsqueeze(1)
    return opt_t - shift * valid


def select_threshold_for_target_recall(probs, labels, target_recall=0.98):
    thresholds = torch.linspace(1.0, 0.0, 1001, device=probs.device)
    best = None
    fallback = None

    for thr in thresholds:
        preds = probs >= thr
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        fpr = fp / max(1, fp + tn)

        candidate = {
            "threshold": float(thr.item()),
            "recall": float(recall),
            "precision": float(precision),
            "fpr": float(fpr),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }

        if fallback is None or candidate["recall"] > fallback["recall"] or (
            abs(candidate["recall"] - fallback["recall"]) < 1e-12
            and candidate["precision"] > fallback["precision"]
        ):
            fallback = candidate

        if recall >= target_recall:
            best = candidate
            break

    if best is None:
        best = fallback
        best["meets_target_recall"] = False
    else:
        best["meets_target_recall"] = True
    best["target_recall"] = float(target_recall)
    return best


def unpack_optical_batch(batch):
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Expected batch to be tuple/list, got {type(batch)!r}")
    if len(batch) == 5:
        opt_t, opt_v, opt_mask, opt_err, labels = batch
        return {
            "opt_t": opt_t,
            "opt_v": opt_v,
            "opt_mask": opt_mask,
            "opt_err": opt_err,
            "labels": labels,
            "slot_is_detection": None,
            "actual_target_k": None,
            "is_terminal_prefix": None,
        }
    if len(batch) == 8:
        opt_t, opt_v, opt_mask, opt_err, labels, slot_is_detection, actual_target_k, is_terminal_prefix = batch
        return {
            "opt_t": opt_t,
            "opt_v": opt_v,
            "opt_mask": opt_mask,
            "opt_err": opt_err,
            "labels": labels,
            "slot_is_detection": slot_is_detection,
            "actual_target_k": actual_target_k,
            "is_terminal_prefix": is_terminal_prefix,
        }
    raise ValueError(f"Unsupported optical eval batch length: {len(batch)}")


def build_prefix_bucket_metrics(probs, labels, actual_target_k, is_terminal_prefix):
    if actual_target_k is None or is_terminal_prefix is None:
        return {}
    probs = probs.detach().cpu()
    labels = labels.detach().cpu().long()
    actual_target_k = actual_target_k.detach().cpu().long()
    is_terminal_prefix = is_terminal_prefix.detach().cpu().long()

    bucket_defs = [
        ("k2", actual_target_k == 2),
        ("k3", actual_target_k == 3),
        ("k4", actual_target_k == 4),
        ("k5", actual_target_k == 5),
        ("k6plus", actual_target_k >= 6),
        ("terminal", is_terminal_prefix > 0),
    ]
    out: Dict[str, Dict[str, float]] = {}
    for name, mask in bucket_defs:
        mask = mask.bool()
        if int(mask.sum().item()) <= 0:
            continue
        p = probs[mask]
        y = labels[mask]
        cls = compute_classification_metrics(p, y)
        out[name] = {
            "n_samples": int(mask.sum().item()),
            "n_pos": int((y == 1).sum().item()),
            "n_neg": int((y == 0).sum().item()),
            "auroc": float(cls.get("auroc", 0.0)),
            "auprc": float(cls.get("auprc", 0.0)),
            "f1_optimal": float(cls.get("f1_optimal", 0.0)),
            "ece": float(cls.get("ece", 0.0)),
            "mean_prob": float(p.mean().item()),
        }
    return out


def save_prefix_manifest_csv(path: str, manifest_rows: List[Dict[str, object]]) -> None:
    if not manifest_rows:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "base_idx",
        "label",
        "prefix_det_target_k",
        "prefix_cut_time",
        "actual_n_det_snr5",
        "actual_n_obs",
        "actual_n_bands",
        "is_terminal_prefix",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


class EvalTimeOffsetPolicy:
    def __init__(
        self,
        enabled: bool,
        dist_npz: Optional[str],
        dist_key: str,
        eval_mode: str,
        eval_quantiles: str,
        scale_divisor: float,
    ):
        self.enabled = bool(enabled)
        self.scale_divisor = float(scale_divisor)
        if self.scale_divisor <= 0:
            raise ValueError("offset_scale_days_divisor must be > 0.")

        self.eval_mode = str(eval_mode).strip().lower()
        self.eval_offsets_days: List[float] = [0.0]
        self.info: Dict[str, object] = {
            "enabled": self.enabled,
            "scale_divisor": self.scale_divisor,
            "eval_mode": self.eval_mode,
        }

        if not self.enabled:
            self.info["eval_offsets_days"] = [0.0]
            return

        if dist_npz is None:
            raise ValueError("time_offset_enable=true requires offset_dist_npz.")
        npz_path = Path(dist_npz)
        if not npz_path.exists():
            raise FileNotFoundError(f"Offset distribution file not found: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as npz:
            if dist_key not in npz:
                raise KeyError(
                    f"offset_dist_key '{dist_key}' not found in {npz_path}. "
                    f"Available keys: {list(npz.keys())}"
                )
            values = np.asarray(npz[dist_key], dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"Offset distribution is empty after filtering NaN/Inf: {npz_path}:{dist_key}")

        if self.eval_mode == "quantile_ensemble":
            q = parse_quantiles(eval_quantiles)
            q_vals = np.quantile(values, np.asarray(q, dtype=np.float64))
            self.eval_offsets_days = [float(v) for v in q_vals.tolist()]
        elif self.eval_mode == "median":
            self.eval_offsets_days = [float(np.quantile(values, 0.5))]
        elif self.eval_mode == "zero":
            self.eval_offsets_days = [0.0]
        else:
            raise ValueError(f"Unsupported offset_eval_mode: {self.eval_mode}")

        self.info.update(
            {
                "dist_path": str(npz_path),
                "dist_key": str(dist_key),
                "dist_count": int(values.shape[0]),
                "dist_min_days": float(np.min(values)),
                "dist_max_days": float(np.max(values)),
                "dist_mean_days": float(np.mean(values)),
                "dist_std_days": float(np.std(values)),
                "eval_offsets_days": [float(v) for v in self.eval_offsets_days],
            }
        )


def load_model(checkpoint_path, device, config_dict):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    def _get(key, default):
        if key in ckpt_args and ckpt_args[key] is not None:
            return ckpt_args[key]
        if key in config_dict and config_dict[key] is not None:
            return config_dict[key]
        return default

    model = OpticalKNClassifier(
        optical_input_dim=6,
        ref_time_dim=_get("ref_dim", 64),
        enc_dim=_get("enc_dim", 64),
        opt_dropout=_get("opt_dropout", 0.1),
        feature_dropout=_get("feature_dropout", 0.0),
        head_hidden_dim=_get("head_hidden_dim", None),
        head_dropout=_get("head_dropout", 0.2),
    )

    state_dict = ckpt.get("model_state_dict", ckpt)
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint does not contain valid state_dict.")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model_state = model.state_dict()
    missing = sorted(set(model_state.keys()) - set(state_dict.keys()))
    unexpected = sorted(set(state_dict.keys()) - set(model_state.keys()))
    shape_mismatch = sorted(
        k
        for k in (set(model_state.keys()) & set(state_dict.keys()))
        if tuple(model_state[k].shape) != tuple(state_dict[k].shape)
    )
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Strict checkpoint loading failed for optical-only eval. "
            f"missing={len(missing)} {_preview_keys(missing)} | "
            f"unexpected={len(unexpected)} {_preview_keys(unexpected)} | "
            f"shape_mismatch={len(shape_mismatch)} {_preview_keys(shape_mismatch)}"
        ) from exc

    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path} | epoch={ckpt.get('epoch', '?')} | strict=True")
    return model, ckpt_args


def build_eval_datasets(args, ckpt_args, config_dict):
    pos_data_path = choose_value(args.pos_data_path, config_dict, ckpt_args, "pos_data_path", default=None)
    neg_data_path = choose_value(args.neg_data_path, config_dict, ckpt_args, "neg_data_path", default=None)
    neg_group = choose_value(args.neg_group, config_dict, ckpt_args, "neg_group", default=None)
    prefix_eval_enable = choose_value(
        args.prefix_eval_enable,
        config_dict,
        ckpt_args,
        "prefix_eval_enable",
        default=None,
    )
    if prefix_eval_enable is None:
        prefix_eval_enable = choose_value(
            None,
            config_dict,
            ckpt_args,
            "prefix_train_enable",
            default=False,
        )
    prefix_eval_enable = bool(prefix_eval_enable)
    prefix_min_det = int(choose_value(args.prefix_min_det, config_dict, ckpt_args, "prefix_min_det", default=2))
    prefix_eval_det_support = parse_prefix_det_support(
        choose_value(
            args.prefix_eval_det_support,
            config_dict,
            ckpt_args,
            "prefix_eval_det_support",
            default="2,3,4,5,6,8,10,12",
        )
    )
    meta_filter_n_det_min = choose_value(
        args.meta_filter_n_det_min,
        config_dict,
        ckpt_args,
        "meta_filter_n_det_min",
        default=None,
    )
    meta_filter_n_det_max = choose_value(
        args.meta_filter_n_det_max,
        config_dict,
        ckpt_args,
        "meta_filter_n_det_max",
        default=None,
    )
    meta_filter_n_bands_max = choose_value(
        args.meta_filter_n_bands_max,
        config_dict,
        ckpt_args,
        "meta_filter_n_bands_max",
        default=None,
    )
    meta_filter_t_span_max = choose_value(
        args.meta_filter_t_span_max,
        config_dict,
        ckpt_args,
        "meta_filter_t_span_max",
        default=None,
    )
    meta_filter_relax_cfg = choose_value(
        args.meta_filter_relax_t_span_if_below_rows,
        config_dict,
        ckpt_args,
        "meta_filter_relax_t_span_if_below_rows",
        default=None,
    )
    meta_filter_relax_thresholds = parse_relax_t_span_thresholds(meta_filter_relax_cfg)
    real_stream_profile_path = choose_value(
        args.real_stream_profile_path,
        config_dict,
        ckpt_args,
        "real_stream_profile_path",
        default=None,
    )
    meta_filter_n_det_min = None if meta_filter_n_det_min is None else int(meta_filter_n_det_min)
    meta_filter_n_det_max = None if meta_filter_n_det_max is None else int(meta_filter_n_det_max)
    meta_filter_n_bands_max = None if meta_filter_n_bands_max is None else int(meta_filter_n_bands_max)
    meta_filter_t_span_max = None if meta_filter_t_span_max is None else float(meta_filter_t_span_max)

    if pos_data_path is None or neg_data_path is None:
        raise ValueError(
            "pos_data_path/neg_data_path missing. Provide via CLI or --config, "
            "or ensure they are stored in checkpoint args."
        )
    if neg_group is None:
        neg_group = _default_neg_group_from_path(neg_data_path)
    if not os.path.exists(pos_data_path):
        raise FileNotFoundError(f"Positive data file not found: {pos_data_path}")
    if not os.path.exists(neg_data_path):
        raise FileNotFoundError(f"Negative data file not found: {neg_data_path}")

    pos_indices = np.arange(
        OpticalBinaryDataset(
            pos_h5_path=pos_data_path,
            neg_h5_path=neg_data_path,
            neg_group=neg_group,
            cache_in_memory=False,
        ).n_pos,
        dtype=np.int64,
    )
    neg_indices = np.arange(
        OpticalBinaryDataset(
            pos_h5_path=pos_data_path,
            neg_h5_path=neg_data_path,
            neg_group=neg_group,
            cache_in_memory=False,
        ).n_neg,
        dtype=np.int64,
    )
    pos_indices_raw = np.asarray(pos_indices, dtype=np.int64)
    neg_indices_raw = np.asarray(neg_indices, dtype=np.int64)

    meta_filter_enabled = any(
        v is not None
        for v in (
            meta_filter_n_det_min,
            meta_filter_n_det_max,
            meta_filter_n_bands_max,
            meta_filter_t_span_max,
        )
    )
    meta_filter_t_span_used = meta_filter_t_span_max
    meta_filter_relaxed = False
    if meta_filter_enabled:
        pos_indices, _ = _filter_indices_by_meta_constraints(
            pos_data_path,
            "events/optical_data",
            pos_indices_raw,
            n_det_min=meta_filter_n_det_min,
            n_det_max=meta_filter_n_det_max,
            n_bands_max=meta_filter_n_bands_max,
            t_span_max=meta_filter_t_span_max,
        )
        neg_indices, _ = _filter_indices_by_meta_constraints(
            neg_data_path,
            neg_group,
            neg_indices_raw,
            n_det_min=meta_filter_n_det_min,
            n_det_max=meta_filter_n_det_max,
            n_bands_max=meta_filter_n_bands_max,
            t_span_max=meta_filter_t_span_max,
        )
        if (
            meta_filter_t_span_max is not None
            and pos_indices.size < int(meta_filter_relax_thresholds["eval"])
            and float(meta_filter_t_span_max) < 0.05
        ):
            meta_filter_relaxed = True
            meta_filter_t_span_used = 0.05
            pos_indices, _ = _filter_indices_by_meta_constraints(
                pos_data_path,
                "events/optical_data",
                pos_indices_raw,
                n_det_min=meta_filter_n_det_min,
                n_det_max=meta_filter_n_det_max,
                n_bands_max=meta_filter_n_bands_max,
                t_span_max=meta_filter_t_span_used,
            )
            neg_indices, _ = _filter_indices_by_meta_constraints(
                neg_data_path,
                neg_group,
                neg_indices_raw,
                n_det_min=meta_filter_n_det_min,
                n_det_max=meta_filter_n_det_max,
                n_bands_max=meta_filter_n_bands_max,
                t_span_max=meta_filter_t_span_used,
            )
        print(
            "Evaluation meta filter: "
            f"n_det=[{meta_filter_n_det_min},{meta_filter_n_det_max}] "
            f"| n_bands<={meta_filter_n_bands_max} "
            f"| t_span<={meta_filter_t_span_used} "
            f"| relaxed={meta_filter_relaxed} "
            f"| pos={pos_indices.size} neg={neg_indices.size}"
        )

    if prefix_eval_enable:
        pos_indices = _filter_indices_by_min_detection_count(
            pos_data_path, "events/optical_data", pos_indices, prefix_min_det
        )
        neg_indices = _filter_indices_by_min_detection_count(
            neg_data_path, neg_group, neg_indices, prefix_min_det
        )
        if pos_indices.size == 0 or neg_indices.size == 0:
            raise ValueError("prefix_eval_enable=true left an empty class after prefix_min_det filtering.")

    rng = np.random.default_rng(args.sample_seed)

    if args.max_pos_samples is not None and args.max_pos_samples < len(pos_indices):
        pos_indices = rng.choice(pos_indices, size=int(args.max_pos_samples), replace=False)
        pos_indices = np.sort(pos_indices)
    if args.max_neg_samples is not None and args.max_neg_samples < len(neg_indices):
        neg_indices = rng.choice(neg_indices, size=int(args.max_neg_samples), replace=False)
        neg_indices = np.sort(neg_indices)

    base_dataset = OpticalBinaryDataset(
        pos_h5_path=pos_data_path,
        neg_h5_path=neg_data_path,
        neg_group=neg_group,
        pos_indices=pos_indices,
        neg_indices=neg_indices,
        cache_in_memory=False,
        return_prefix_aux=prefix_eval_enable,
    )

    print(
        "Evaluation base dataset: "
        f"n_pos={base_dataset.n_pos}, n_neg={base_dataset.n_neg}, total={len(base_dataset)} | "
        f"task_mode={'prefix_right_censored' if prefix_eval_enable else 'full_window'}"
    )

    primary_dataset = base_dataset
    legacy_dataset = None
    manifest_rows: List[Dict[str, object]] = []
    if prefix_eval_enable:
        manifest_rows = _build_prefix_manifest_for_binary_dataset(
            base_dataset=base_dataset,
            det_support=prefix_eval_det_support,
            prefix_min_det=prefix_min_det,
            include_terminal=True,
        )
        if not manifest_rows:
            raise ValueError("Prefix evaluation manifest is empty.")
        primary_dataset = OpticalPrefixEvalDataset(
            base_dataset=base_dataset,
            manifest_rows=manifest_rows,
            prefix_min_det=prefix_min_det,
        )
        legacy_dataset = OpticalBinaryDataset(
            pos_h5_path=pos_data_path,
            neg_h5_path=neg_data_path,
            neg_group=neg_group,
            pos_indices=pos_indices,
            neg_indices=neg_indices,
            cache_in_memory=False,
            return_prefix_aux=False,
        )
        print(
            "Prefix evaluation manifest: "
            f"n_rows={len(primary_dataset)}, det_support={prefix_eval_det_support}, include_terminal=True"
        )

    return primary_dataset, legacy_dataset, {
        "pos_data_path": pos_data_path,
        "neg_data_path": neg_data_path,
        "neg_group": neg_group,
        "arch_version": str(choose_value(None, config_dict, ckpt_args, "arch_version", default="unknown")),
        "task_mode": "prefix_right_censored" if prefix_eval_enable else "full_window",
        "prefix_eval_enable": bool(prefix_eval_enable),
        "prefix_min_det": int(prefix_min_det),
        "prefix_eval_det_support": [int(v) for v in prefix_eval_det_support],
        "prefix_manifest_rows": int(len(manifest_rows)),
        "meta_filter": {
            "n_det_min": (None if meta_filter_n_det_min is None else int(meta_filter_n_det_min)),
            "n_det_max": (None if meta_filter_n_det_max is None else int(meta_filter_n_det_max)),
            "n_bands_max": (None if meta_filter_n_bands_max is None else int(meta_filter_n_bands_max)),
            "t_span_max": (None if meta_filter_t_span_max is None else float(meta_filter_t_span_max)),
            "t_span_used": (None if meta_filter_t_span_used is None else float(meta_filter_t_span_used)),
            "relaxed": bool(meta_filter_relaxed),
            "relax_t_span_if_below_rows": meta_filter_relax_thresholds,
        },
        "real_stream_profile_path": real_stream_profile_path,
    }


def build_eval_loader(dataset, args):
    if args.num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=args.prefetch_factor,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bool(args.pin_memory),
    )


@torch.no_grad()
def run_evaluation(model, loader, device, n_ref, ref_start, ref_end, target_recall, offset_policy):
    criterion = nn.BCEWithLogitsLoss()

    if device.type == "cuda":
        use_bf16 = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    else:
        amp_dtype = torch.float32

    all_probs = []
    all_logits = []
    all_labels = []
    all_actual_target_k = []
    all_is_terminal_prefix = []
    total_loss = 0.0
    n_batches = 0
    ref_time_cache = None

    for batch in tqdm(loader, desc="Evaluating"):
        batch_dict = unpack_optical_batch(batch)
        opt_t = batch_dict["opt_t"]
        opt_v = batch_dict["opt_v"]
        opt_mask = batch_dict["opt_mask"]
        opt_err = batch_dict["opt_err"]
        labels = batch_dict["labels"]
        actual_target_k = batch_dict["actual_target_k"]
        is_terminal_prefix = batch_dict["is_terminal_prefix"]
        opt_t = opt_t.to(device, non_blocking=True)
        opt_v = opt_v.to(device, non_blocking=True)
        opt_mask = opt_mask.to(device, non_blocking=True)
        opt_err = opt_err.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        if actual_target_k is not None:
            actual_target_k = actual_target_k.to(device, non_blocking=True)
        if is_terminal_prefix is not None:
            is_terminal_prefix = is_terminal_prefix.to(device, non_blocking=True)

        batch_size = opt_t.size(0)
        if (
            ref_time_cache is None
            or ref_time_cache.shape[0] != batch_size
            or ref_time_cache.dtype != opt_t.dtype
        ):
            ref_time_cache = build_ref_time(batch_size, n_ref, ref_start, ref_end, device, opt_t.dtype)

        if not offset_policy.enabled:
            with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                logits = model(opt_t, opt_v, ref_time_cache, opt_mask, opt_err).squeeze(-1)
            logits = logits.float()
        else:
            logits_accum = None
            for off_days in offset_policy.eval_offsets_days:
                delta_days = torch.full((batch_size,), float(off_days), device=device, dtype=torch.float32)
                shifted_opt_t = apply_time_offsets(opt_t, opt_mask, delta_days, offset_policy.scale_divisor)
                with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
                    logits_i = model(shifted_opt_t, opt_v, ref_time_cache, opt_mask, opt_err).squeeze(-1)
                logits_i = logits_i.float()
                logits_accum = logits_i if logits_accum is None else logits_accum + logits_i
            logits = logits_accum / float(max(1, len(offset_policy.eval_offsets_days)))

        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        n_batches += 1
        all_probs.append(probs.detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu().long())
        if actual_target_k is not None:
            all_actual_target_k.append(actual_target_k.detach().cpu())
        if is_terminal_prefix is not None:
            all_is_terminal_prefix.append(is_terminal_prefix.detach().cpu())

    if n_batches == 0:
        raise RuntimeError("No batches were evaluated.")

    probs = torch.cat(all_probs, dim=0)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    cls = compute_classification_metrics(probs, labels)
    op = select_threshold_for_target_recall(probs, labels, target_recall=target_recall)

    results = {
        "loss": float(total_loss / n_batches),
        **cls,
        "op_threshold": op["threshold"],
        "op_recall": op["recall"],
        "op_precision": op["precision"],
        "op_fpr": op["fpr"],
        "op_meets_target_recall": bool(op["meets_target_recall"]),
        "target_recall": float(target_recall),
        "offset_eval_count": int(len(offset_policy.eval_offsets_days)),
    }
    prefix_bucket_metrics = build_prefix_bucket_metrics(
        probs=probs,
        labels=labels,
        actual_target_k=(torch.cat(all_actual_target_k, dim=0) if all_actual_target_k else None),
        is_terminal_prefix=(torch.cat(all_is_terminal_prefix, dim=0) if all_is_terminal_prefix else None),
    )
    if prefix_bucket_metrics:
        results["prefix_bucket_metrics"] = prefix_bucket_metrics
    return results, probs.numpy(), labels.numpy(), logits.numpy()


def generate_plots(probs, labels, logits, results, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)

    sorted_idx = np.argsort(-probs)
    sorted_labels = labels[sorted_idx]
    n_pos = int(sorted_labels.sum())
    n_neg = int(len(sorted_labels) - n_pos)
    if n_pos > 0 and n_neg > 0:
        tpr = np.cumsum(sorted_labels) / n_pos
        fpr = np.cumsum(1 - sorted_labels) / n_neg
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"ROC Curve (AUROC={results.get('auroc', 0.0):.4f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if n_pos > 0:
        tp_cum = np.cumsum(sorted_labels)
        ranks = np.arange(1, len(sorted_labels) + 1)
        precision = tp_cum / np.maximum(ranks, 1)
        recall = tp_cum / max(n_pos, 1)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, lw=2)
        ax.set_title(f"Precision-Recall Curve (AUPRC={results.get('auprc', 0.0):.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    n_bins = 10
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (probs >= low) & (probs < high)
        else:
            mask = (probs >= low) & (probs <= high)
        if mask.any():
            bin_accs.append(labels[mask].mean())
            bin_confs.append(probs[mask].mean())
    if bin_accs:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, label="Model")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ax.set_title(f"Calibration (ECE={results.get('ece', 0.0):.4f})")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "calibration.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0.0, 1.0, 61)
    if len(pos_probs) > 0:
        ax.hist(
            pos_probs,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color="#27ae60",
            label=f"Positive (KN) n={len(pos_probs)}",
        )
    if len(neg_probs) > 0:
        ax.hist(
            neg_probs,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color="#c0392b",
            label=f"Negative (non-KN) n={len(neg_probs)}",
        )
    op_thr = float(results.get("op_threshold", 0.5))
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="thr=0.5")
    ax.axvline(op_thr, color="#34495e", linestyle="-.", linewidth=1.2, alpha=0.8, label=f"op_thr={op_thr:.3f}")
    ax.set_title("Optical-only Match Probability Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    fig.savefig(os.path.join(output_dir, "prob_distribution.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    pos_logits = logits[labels == 1]
    neg_logits = logits[labels == 0]
    finite_logits = logits[np.isfinite(logits)]
    if finite_logits.size > 0:
        lo, hi = np.percentile(finite_logits, [0.5, 99.5])
        span = max(2.0, float(hi - lo))
        pad = 0.1 * span
        bins = np.linspace(float(lo - pad), float(hi + pad), 81)

        fig, ax = plt.subplots(figsize=(10, 6))
        if len(pos_logits) > 0:
            ax.hist(
                pos_logits,
                bins=bins,
                histtype="step",
                linewidth=2.0,
                color="#27ae60",
                label=f"Positive (KN) n={len(pos_logits)}",
            )
        if len(neg_logits) > 0:
            ax.hist(
                neg_logits,
                bins=bins,
                histtype="step",
                linewidth=2.0,
                color="#c0392b",
                label=f"Negative (non-KN) n={len(neg_logits)}",
            )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="logit(0.5)=0")
        op_thr = float(results.get("op_threshold", 0.5))
        if 0.0 < op_thr < 1.0:
            op_logit = np.log(op_thr / (1.0 - op_thr))
            if np.isfinite(op_logit):
                ax.axvline(
                    float(op_logit),
                    color="#34495e",
                    linestyle="-.",
                    linewidth=1.2,
                    alpha=0.8,
                    label=f"logit(op_thr)={op_logit:.3f}",
                )
        ax.set_title("Optical-only Logits Distribution")
        ax.set_xlabel("Logit")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "logits_distribution.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"Plots saved to {output_dir}/")


def main():
    args = parse_args()
    config_dict = load_json(args.config)

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
    print(f"Using device: {device}")

    model, ckpt_args = load_model(args.checkpoint, device, config_dict)
    dataset, legacy_dataset, data_meta = build_eval_datasets(args, ckpt_args, config_dict)
    loader = build_eval_loader(dataset, args)

    n_ref = int(choose_value(None, config_dict, ckpt_args, "n_ref", default=64))
    ref_start = float(choose_value(None, config_dict, ckpt_args, "ref_start", default=-0.3))
    ref_end = float(choose_value(None, config_dict, ckpt_args, "ref_end", default=0.6))
    target_recall = float(
        choose_value(args.target_recall, config_dict, ckpt_args, "target_recall", default=0.98)
    )

    offset_enabled = bool(
        choose_value(args.time_offset_enable, config_dict, ckpt_args, "time_offset_enable", default=False)
    )
    offset_dist_npz = choose_value(args.offset_dist_npz, config_dict, ckpt_args, "offset_dist_npz", default=None)
    offset_dist_key = str(
        choose_value(args.offset_dist_key, config_dict, ckpt_args, "offset_dist_key", default="delta_days_combined")
    )
    offset_eval_mode = str(
        choose_value(args.offset_eval_mode, config_dict, ckpt_args, "offset_eval_mode", default="quantile_ensemble")
    )
    offset_eval_quantiles = str(
        choose_value(
            args.offset_eval_quantiles,
            config_dict,
            ckpt_args,
            "offset_eval_quantiles",
            default="0.1,0.3,0.5,0.7,0.9",
        )
    )
    offset_scale_days_divisor = float(
        choose_value(
            args.offset_scale_days_divisor,
            config_dict,
            ckpt_args,
            "offset_scale_days_divisor",
            default=100.0,
        )
    )

    offset_policy = EvalTimeOffsetPolicy(
        enabled=offset_enabled,
        dist_npz=offset_dist_npz,
        dist_key=offset_dist_key,
        eval_mode=offset_eval_mode,
        eval_quantiles=offset_eval_quantiles,
        scale_divisor=offset_scale_days_divisor,
    )
    print(f"Time offset policy: {json.dumps(offset_policy.info, indent=2)}")

    metrics, probs, labels, logits = run_evaluation(
        model=model,
        loader=loader,
        device=device,
        n_ref=n_ref,
        ref_start=ref_start,
        ref_end=ref_end,
        target_recall=target_recall,
        offset_policy=offset_policy,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if bool(data_meta.get("prefix_eval_enable", False)):
        manifest_out = args.prefix_manifest_out
        if manifest_out is None:
            manifest_out = os.path.join(args.output_dir, "prefix_eval_manifest.csv")
        if isinstance(dataset, OpticalPrefixEvalDataset):
            save_prefix_manifest_csv(manifest_out, dataset.manifest.to_dict(orient="records"))

    legacy_results = None
    if legacy_dataset is not None:
        legacy_loader = build_eval_loader(legacy_dataset, args)
        legacy_metrics, _, _, _ = run_evaluation(
            model=model,
            loader=legacy_loader,
            device=device,
            n_ref=n_ref,
            ref_start=ref_start,
            ref_end=ref_end,
            target_recall=target_recall,
            offset_policy=offset_policy,
        )
        legacy_results = legacy_metrics

    results = {
        "classification": metrics,
        "legacy_full_window": legacy_results,
        "meta": {
            "checkpoint": args.checkpoint,
            "device": str(device),
            "n_samples": int(len(labels)),
            "n_pos": int((labels == 1).sum()),
            "n_neg": int((labels == 0).sum()),
            "n_ref": n_ref,
            "ref_start": ref_start,
            "ref_end": ref_end,
            "batch_size": int(args.batch_size),
            "time_offset": offset_policy.info,
            **data_meta,
        },
    }

    out_json = os.path.join(args.output_dir, "eval_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Optical-only Evaluation Summary (fd_t0) ===")
    print(f"loss: {metrics['loss']:.4f}")
    print(f"AUROC: {metrics.get('auroc', 0.0):.4f}")
    print(f"AUPRC: {metrics.get('auprc', 0.0):.4f}")
    print(f"F1(opt): {metrics.get('f1_optimal', 0.0):.4f}")
    print(
        f"op@recall>={metrics['target_recall']:.3f}: "
        f"thr={metrics['op_threshold']:.3f}, "
        f"recall={metrics['op_recall']:.4f}, "
        f"precision={metrics['op_precision']:.4f}, "
        f"fpr={metrics['op_fpr']:.4f}"
    )
    if legacy_results is not None:
        print(
            "Legacy full-window secondary eval: "
            f"AUROC={legacy_results.get('auroc', 0.0):.4f}, "
            f"AUPRC={legacy_results.get('auprc', 0.0):.4f}, "
            f"thr={legacy_results.get('op_threshold', 0.0):.3f}"
        )
    print(f"Saved: {out_json}")

    if not args.no_plots:
        generate_plots(probs, labels, logits, metrics, args.output_dir)


if __name__ == "__main__":
    main()
