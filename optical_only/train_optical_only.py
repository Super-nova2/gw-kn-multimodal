#!/usr/bin/env python3
"""
Train optical-only KN classifier with first-detection time-zero and time-offset modeling.

Key additions versus baseline train script:
1) Optional empirical-CDF offset sampling in training.
2) Optional quantile-ensemble offset inference in validation (average logits).
"""

import argparse
import ctypes
import datetime
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - runtime environment dependent
    SummaryWriter = None

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "Model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from data_loader import create_optical_binary_dataloaders
from metrics import compute_classification_metrics
from model import OpticalKNClassifier
from optical_prefix import (
    apply_prefix_right_censoring_torch,
    load_prefix_ndet_distribution,
    parse_prefix_det_support,
    sample_prefix_target_k,
)


try:
    _LIBC = ctypes.CDLL("libc.so.6")
except Exception:  # pragma: no cover - libc availability depends on runtime
    _LIBC = None


def _trim_process_heap() -> bool:
    if _LIBC is None:
        return False
    try:
        return bool(_LIBC.malloc_trim(0))
    except Exception:
        return False


def _cleanup_cuda_caches(device: torch.device) -> None:
    if device.type != "cuda":
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def close_dataset_file_handles(dataset, seen: Optional[set] = None) -> int:
    if dataset is None:
        return 0
    if seen is None:
        seen = set()
    obj_id = id(dataset)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    n_closed = 0
    for attr in ("pos_file", "neg_file", "h5_file"):
        handle = getattr(dataset, attr, None)
        if handle is not None:
            try:
                handle.close()
                n_closed += 1
            except Exception:
                pass
            try:
                setattr(dataset, attr, None)
            except Exception:
                pass

    nested = getattr(dataset, "base_dataset", None)
    if nested is not None:
        n_closed += close_dataset_file_handles(nested, seen=seen)
    return n_closed


def recycle_dataloader_workers(loader) -> bool:
    if loader is None:
        return False
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return False
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass
    try:
        loader._iterator = None
    except Exception:
        pass
    return True


def run_memory_maintenance(
    device: torch.device,
    loaders: Optional[List[object]] = None,
    recycle_workers: bool = False,
    close_loader_files: bool = False,
    trim_heap: bool = False,
) -> Dict[str, int]:
    loader_list = [ldr for ldr in (loaders or []) if ldr is not None]
    recycled = 0
    closed = 0
    if recycle_workers:
        for loader in loader_list:
            recycled += int(recycle_dataloader_workers(loader))
    if close_loader_files:
        for loader in loader_list:
            closed += int(close_dataset_file_handles(getattr(loader, "dataset", None)))
    collected = int(gc.collect())
    _cleanup_cuda_caches(device)
    trimmed = int(_trim_process_heap()) if trim_heap else 0
    return {
        "gc_collected": collected,
        "workers_recycled": recycled,
        "file_handles_closed": closed,
        "heap_trimmed": trimmed,
    }


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


def parse_bin_edges(text: str, default: str) -> List[float]:
    raw = str(default if text is None else text)
    vals: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("Meta bin edges must not be empty.")
    arr = np.unique(np.asarray(vals, dtype=np.float64))
    if np.any(~np.isfinite(arr)):
        raise ValueError("Meta bin edges must be finite numeric values.")
    return [float(v) for v in arr.tolist()]


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


def compute_shortcut_audit(
    n_det: np.ndarray,
    n_bands: np.ndarray,
    t_span: np.ndarray,
    labels: np.ndarray,
    max_samples: int = 50000,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n_det = np.asarray(n_det, dtype=np.float64).reshape(-1)
    n_bands = np.asarray(n_bands, dtype=np.float64).reshape(-1)
    t_span = np.asarray(t_span, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n = int(labels.shape[0])
    if n == 0:
        return out
    if n > int(max_samples):
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=int(max_samples), replace=False)
        n_det = n_det[idx]
        n_bands = n_bands[idx]
        t_span = t_span[idx]
        labels = labels[idx]

    out["n_eval_samples"] = float(labels.shape[0])
    out["n_det_pos_median"] = float(np.median(n_det[labels == 1])) if np.any(labels == 1) else 0.0
    out["n_det_neg_median"] = float(np.median(n_det[labels == 0])) if np.any(labels == 0) else 0.0
    out["n_bands_pos_median"] = float(np.median(n_bands[labels == 1])) if np.any(labels == 1) else 0.0
    out["n_bands_neg_median"] = float(np.median(n_bands[labels == 0])) if np.any(labels == 0) else 0.0
    out["t_span_pos_median"] = float(np.median(t_span[labels == 1])) if np.any(labels == 1) else 0.0
    out["t_span_neg_median"] = float(np.median(t_span[labels == 0])) if np.any(labels == 0) else 0.0

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        X = np.column_stack([n_det, n_bands, t_span]).astype(np.float64)
        y = labels.astype(np.int64)
        if np.unique(y).size >= 2 and X.shape[0] >= 32:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
            out["shortcut_logreg_auc"] = float(roc_auc_score(y_te, probs))
    except Exception:
        pass

    return out


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
    if len(batch) == 6:
        opt_t, opt_v, opt_mask, opt_err, labels, slot_is_detection = batch
        return {
            "opt_t": opt_t,
            "opt_v": opt_v,
            "opt_mask": opt_mask,
            "opt_err": opt_err,
            "labels": labels,
            "slot_is_detection": slot_is_detection,
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
    raise ValueError(f"Unsupported optical batch length: {len(batch)}")


def compute_batch_sequence_meta(opt_t, opt_mask, slot_is_detection=None):
    valid_rows = opt_mask.sum(dim=-1) > 0
    if slot_is_detection is not None:
        n_det = ((slot_is_detection > 0) & valid_rows).sum(dim=1)
    else:
        n_det = valid_rows.sum(dim=1)
    n_bands = (opt_mask.sum(dim=1) > 0).sum(dim=1)
    row_idx = torch.arange(opt_t.size(0), device=opt_t.device)
    has_valid = valid_rows.any(dim=1)
    first_idx = torch.argmax(valid_rows.to(torch.int64), dim=1)
    rev_idx = torch.argmax(valid_rows.flip(dims=[1]).to(torch.int64), dim=1)
    last_idx = valid_rows.size(1) - 1 - rev_idx
    t_first = torch.where(has_valid, opt_t[row_idx, first_idx], torch.zeros(opt_t.size(0), device=opt_t.device, dtype=opt_t.dtype))
    t_last = torch.where(has_valid, opt_t[row_idx, last_idx], torch.zeros(opt_t.size(0), device=opt_t.device, dtype=opt_t.dtype))
    t_span = torch.where(has_valid, t_last - t_first, torch.zeros_like(t_last))
    return n_det, n_bands, t_span


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


class PrefixTrainPolicy:
    def __init__(self, args):
        self.enabled = bool(getattr(args, "prefix_train_enable", False))
        self.min_det = int(getattr(args, "prefix_min_det", 2))
        self.sampling = str(getattr(args, "prefix_train_sampling", "real_stream_ndet_empirical")).strip().lower()
        self.terminal_mix_prob = float(getattr(args, "prefix_terminal_mix_prob", 0.2))
        self.real_mix_weight = float(getattr(args, "prefix_real_mix_weight", 4.0))
        self.bucket_uniform_mix_weight = float(getattr(args, "prefix_bucket_uniform_mix_weight", 4.0))
        self.terminal_mix_weight = float(getattr(args, "prefix_terminal_mix_weight", 2.0))
        self.hist_path = getattr(args, "prefix_real_hist_path", None)
        self.rng = np.random.default_rng(int(args.seed) + 137)
        self.support = None
        self.probs = None
        self.uniform_probs = None
        self.hist_meta: Dict[str, object] = {}
        self.info: Dict[str, object] = {
            "enabled": self.enabled,
            "min_det": self.min_det,
            "sampling": self.sampling,
            "terminal_mix_prob": self.terminal_mix_prob,
            "real_mix_weight": self.real_mix_weight,
            "bucket_uniform_mix_weight": self.bucket_uniform_mix_weight,
            "terminal_mix_weight": self.terminal_mix_weight,
        }

        if not self.enabled:
            return
        if self.sampling not in {"real_stream_ndet_empirical", "real_stream_bucket_uniform_terminal_mixture"}:
            raise ValueError(f"Unsupported prefix_train_sampling: {self.sampling}")
        if self.hist_path is None:
            raise ValueError("prefix_train_enable=true requires --prefix_real_hist_path.")
        support, probs, hist_meta = load_prefix_ndet_distribution(self.hist_path, min_det=self.min_det)
        self.support = support
        self.probs = probs
        self.uniform_probs = np.full_like(probs, fill_value=(1.0 / float(probs.shape[0])), dtype=np.float64)
        self.hist_meta = hist_meta
        self.info.update(
            {
                "hist_path": str(self.hist_path),
                "hist_support": [int(v) for v in support.tolist()],
                "hist_probabilities": [float(v) for v in probs.tolist()],
            }
        )
        if self.sampling == "real_stream_bucket_uniform_terminal_mixture":
            mix_raw = np.asarray(
                [
                    self.real_mix_weight,
                    self.bucket_uniform_mix_weight,
                    self.terminal_mix_weight,
                ],
                dtype=np.float64,
            )
            if np.any(mix_raw < 0):
                raise ValueError("Prefix mixture weights must be >= 0.")
            if float(mix_raw.sum()) <= 0:
                raise ValueError("Prefix mixture weights must sum to > 0.")
            mix_probs = mix_raw / mix_raw.sum()
            self.info.update(
                {
                    "mixture_branch_names": ["real_stream", "bucket_uniform", "terminal"],
                    "mixture_branch_probabilities": [float(v) for v in mix_probs.tolist()],
                    "uniform_support": [int(v) for v in support.tolist()],
                }
            )

    def sample_target_k(self, slot_is_detection: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("PrefixTrainPolicy.sample_target_k called while disabled.")
        det_counts = (slot_is_detection > 0).sum(dim=1).detach().cpu().numpy().astype(np.int64, copy=False)
        if self.sampling == "real_stream_ndet_empirical":
            sampled = sample_prefix_target_k(
                det_counts=det_counts,
                support=self.support,
                probs=self.probs,
                rng=self.rng,
                min_det=self.min_det,
                terminal_mix_prob=self.terminal_mix_prob,
            )
        else:
            sampled = np.empty_like(det_counts)
            mix_raw = np.asarray(
                [
                    self.real_mix_weight,
                    self.bucket_uniform_mix_weight,
                    self.terminal_mix_weight,
                ],
                dtype=np.float64,
            )
            mix_probs = mix_raw / mix_raw.sum()
            branch_ids = self.rng.choice(3, size=det_counts.shape[0], replace=True, p=mix_probs)

            real_mask = branch_ids == 0
            if np.any(real_mask):
                sampled[real_mask] = sample_prefix_target_k(
                    det_counts=det_counts[real_mask],
                    support=self.support,
                    probs=self.probs,
                    rng=self.rng,
                    min_det=self.min_det,
                    terminal_mix_prob=0.0,
                )

            uniform_mask = branch_ids == 1
            if np.any(uniform_mask):
                sampled[uniform_mask] = sample_prefix_target_k(
                    det_counts=det_counts[uniform_mask],
                    support=self.support,
                    probs=self.uniform_probs,
                    rng=self.rng,
                    min_det=self.min_det,
                    terminal_mix_prob=0.0,
                )

            terminal_mask = branch_ids == 2
            if np.any(terminal_mask):
                sampled[terminal_mask] = det_counts[terminal_mask]

        return torch.from_numpy(sampled).to(device=slot_is_detection.device, dtype=torch.long)

    def describe(self) -> Dict[str, object]:
        return dict(self.info)


class TimeOffsetPolicy:
    def __init__(self, args):
        self.enabled = bool(args.time_offset_enable)
        self.scale_divisor = float(args.offset_scale_days_divisor)
        if self.scale_divisor <= 0:
            raise ValueError("offset_scale_days_divisor must be > 0.")

        self.train_sampling = str(args.offset_train_sampling).strip().lower()
        self.eval_mode = str(args.offset_eval_mode).strip().lower()
        self.eval_offsets_days: List[float] = [0.0]

        seed = int(args.seed if args.offset_seed is None else args.offset_seed)
        self.rng = np.random.default_rng(seed)
        self.samples: Optional[np.ndarray] = None
        self.sample_bank: Optional[np.ndarray] = None
        self.info: Dict[str, object] = {
            "enabled": bool(self.enabled),
            "scale_divisor": float(self.scale_divisor),
            "train_sampling": self.train_sampling,
            "eval_mode": self.eval_mode,
            "seed": int(seed),
        }

        if not self.enabled:
            self.info["eval_offsets_days"] = [0.0]
            return

        dist_path = args.offset_dist_npz
        if dist_path is None:
            raise ValueError("time_offset_enable=true requires --offset_dist_npz.")

        dist_path = Path(dist_path)
        if not dist_path.exists():
            raise FileNotFoundError(f"Offset distribution file not found: {dist_path}")

        dist_key = str(args.offset_dist_key)
        with np.load(dist_path, allow_pickle=False) as npz:
            if dist_key not in npz:
                raise KeyError(
                    f"offset_dist_key '{dist_key}' not found in {dist_path}. "
                    f"Available keys: {list(npz.keys())}"
                )
            raw = np.asarray(npz[dist_key], dtype=np.float64).reshape(-1)

        raw = raw[np.isfinite(raw)]
        if raw.size == 0:
            raise ValueError(f"Offset distribution is empty after filtering NaN/Inf: {dist_path}:{dist_key}")

        self.samples = raw.astype(np.float32, copy=False)

        bank_size = max(1, min(int(args.offset_bank_size), int(self.samples.shape[0])))
        if bank_size < int(self.samples.shape[0]):
            bank_idx = self.rng.integers(0, int(self.samples.shape[0]), size=bank_size, endpoint=False)
            self.sample_bank = self.samples[bank_idx].astype(np.float32, copy=False)
        else:
            self.sample_bank = self.samples

        if self.train_sampling != "empirical_cdf":
            raise ValueError(f"Unsupported offset_train_sampling: {self.train_sampling}")

        if self.eval_mode == "quantile_ensemble":
            quantiles = parse_quantiles(args.offset_eval_quantiles)
            q_vals = np.quantile(self.samples.astype(np.float64), np.asarray(quantiles, dtype=np.float64))
            self.eval_offsets_days = [float(v) for v in q_vals.tolist()]
        elif self.eval_mode == "median":
            self.eval_offsets_days = [float(np.quantile(self.samples.astype(np.float64), 0.5))]
        elif self.eval_mode == "zero":
            self.eval_offsets_days = [0.0]
        else:
            raise ValueError(f"Unsupported offset_eval_mode: {self.eval_mode}")

        self.info.update(
            {
                "dist_path": str(dist_path),
                "dist_key": dist_key,
                "dist_count": int(self.samples.shape[0]),
                "dist_min_days": float(np.min(self.samples)),
                "dist_max_days": float(np.max(self.samples)),
                "dist_mean_days": float(np.mean(self.samples)),
                "dist_std_days": float(np.std(self.samples)),
                "bank_size": int(self.sample_bank.shape[0]),
                "eval_offsets_days": [float(v) for v in self.eval_offsets_days],
            }
        )

    def sample_train_offsets(self, batch_size: int) -> np.ndarray:
        if not self.enabled:
            return np.zeros((int(batch_size),), dtype=np.float32)
        if self.sample_bank is None:
            raise RuntimeError("time offset sample bank is not initialized")
        idx = self.rng.integers(0, int(self.sample_bank.shape[0]), size=int(batch_size), endpoint=False)
        return self.sample_bank[idx].astype(np.float32, copy=False)

    def describe(self) -> Dict[str, object]:
        return dict(self.info)


def build_ref_time(batch_size, n_ref, ref_start, ref_end, device, dtype):
    ref = torch.linspace(ref_start, ref_end, n_ref, dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)


def apply_time_offsets(opt_t, opt_mask, delta_days, scale_divisor):
    valid = (opt_mask.sum(dim=-1) > 0).to(dtype=opt_t.dtype)
    shift = (delta_days.to(device=opt_t.device, dtype=opt_t.dtype) / float(scale_divisor)).unsqueeze(1)
    return opt_t - shift * valid


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
    single_band_keep_prob=1.0,
    target_ndet_jitter=0.0,
):
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
        band_mask = torch.rand(opt_v.size(0), opt_v.size(2), device=opt_v.device) < band_dropout
        if band_mask.any():
            band_mask = band_mask[:, None, :]
            opt_mask = opt_mask.masked_fill(band_mask, 0)
            opt_v = opt_v.masked_fill(band_mask, 0.0)
            if opt_err is not None:
                opt_err = opt_err.masked_fill(band_mask, 0.0)

    if target_ndet_jitter > 0:
        jitter = float(target_ndet_jitter)
        valid = opt_mask > 0
        for i in range(opt_mask.size(0)):
            idx = torch.nonzero(valid[i], as_tuple=False)
            n_obs = int(idx.size(0))
            if n_obs <= 1:
                continue
            frac = 1.0 + float(torch.empty((1,), device=opt_mask.device).uniform_(-jitter, jitter).item())
            target = int(round(n_obs * frac))
            target = max(1, min(n_obs, target))
            if target >= n_obs:
                continue
            perm = torch.randperm(n_obs, device=opt_mask.device)
            keep = idx[perm[:target]]
            keep_mask = torch.zeros_like(valid[i], dtype=torch.bool)
            keep_mask[keep[:, 0], keep[:, 1]] = True
            drop = valid[i] & (~keep_mask)
            if drop.any():
                opt_mask[i] = opt_mask[i].masked_fill(drop, 0)
                opt_v[i] = opt_v[i].masked_fill(drop, 0.0)
                if opt_err is not None:
                    opt_err[i] = opt_err[i].masked_fill(drop, 0.0)

    if single_band_keep_prob < 1.0:
        keep_prob = float(single_band_keep_prob)
        band_presence = opt_mask.sum(dim=1) > 0
        n_bands = band_presence.sum(dim=1)
        single_ids = torch.where(n_bands == 1)[0]
        for i in single_ids.tolist():
            if float(torch.rand((1,), device=opt_mask.device).item()) <= keep_prob:
                continue
            valid = opt_mask[i] > 0
            n_valid = int(valid.sum().item())
            if n_valid <= 1:
                continue
            drop = (torch.rand_like(opt_mask[i]) < 0.5) & valid
            if int((valid & (~drop)).sum().item()) <= 0:
                idx = torch.nonzero(valid, as_tuple=False)
                keep_one = idx[torch.randint(0, idx.size(0), (1,), device=opt_mask.device)[0]]
                drop[keep_one[0], keep_one[1]] = False
            if drop.any():
                opt_mask[i] = opt_mask[i].masked_fill(drop, 0)
                opt_v[i] = opt_v[i].masked_fill(drop, 0.0)
                if opt_err is not None:
                    opt_err[i] = opt_err[i].masked_fill(drop, 0.0)

    return opt_t, opt_v, opt_mask, opt_err


def sample_universal_target_k(
    slot_is_detection: torch.Tensor,
    opt_mask: torch.Tensor,
    min_det: int,
    max_det: int = 12,
    terminal_prob: float = 0.2,
) -> torch.Tensor:
    valid_rows = opt_mask.sum(dim=-1) > 0
    det_counts = ((slot_is_detection > 0) & valid_rows).sum(dim=1).to(dtype=torch.long)
    if bool((det_counts < int(min_det)).any().item()):
        raise ValueError("Universal prefix sampling received samples below prefix_min_det.")
    capped_upper = torch.minimum(det_counts, torch.full_like(det_counts, int(max_det)))
    rand_u = torch.rand(det_counts.shape[0], device=slot_is_detection.device)
    terminal_mask = rand_u < float(terminal_prob)
    range_size = torch.clamp(capped_upper - int(min_det) + 1, min=1)
    draw = (torch.rand(det_counts.shape[0], device=slot_is_detection.device) * range_size.to(torch.float32)).floor().to(torch.long)
    sampled = draw + int(min_det)
    sampled = torch.minimum(sampled, capped_upper)
    sampled = torch.where(terminal_mask, det_counts, sampled)
    sampled = torch.clamp(sampled, min=int(min_det))
    return sampled


def bucketize_n_det(n_det: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(n_det, dtype=torch.long)
    out = torch.where(n_det == 4, torch.ones_like(out), out)
    out = torch.where((n_det >= 5) & (n_det <= 6), torch.full_like(out, 2), out)
    out = torch.where((n_det >= 7) & (n_det <= 12), torch.full_like(out, 3), out)
    out = torch.where(n_det >= 13, torch.full_like(out, 4), out)
    return out


def bucketize_n_bands(n_bands: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(n_bands, dtype=torch.long)
    out = torch.where(n_bands == 2, torch.ones_like(out), out)
    out = torch.where(n_bands == 3, torch.full_like(out, 2), out)
    out = torch.where(n_bands >= 4, torch.full_like(out, 3), out)
    return out


def bucketize_t_span(t_span: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(t_span, dtype=torch.long)
    out = torch.where(t_span > 0.01, torch.ones_like(out), out)
    out = torch.where(t_span > 0.05, torch.full_like(out, 2), out)
    out = torch.where(t_span > 0.2, torch.full_like(out, 3), out)
    out = torch.where(t_span > 0.6, torch.full_like(out, 4), out)
    return out


def build_universal_view(
    opt_t: torch.Tensor,
    opt_v: torch.Tensor,
    opt_mask: torch.Tensor,
    opt_err: torch.Tensor,
    slot_is_detection: torch.Tensor,
    args,
) -> Dict[str, torch.Tensor]:
    target_k = sample_universal_target_k(
        slot_is_detection=slot_is_detection,
        opt_mask=opt_mask,
        min_det=int(args.prefix_min_det),
        max_det=12,
        terminal_prob=float(args.prefix_terminal_mix_prob),
    )
    total_det = ((slot_is_detection > 0) & (opt_mask.sum(dim=-1) > 0)).sum(dim=1).to(dtype=torch.long)
    opt_t, opt_v, opt_mask, opt_err, slot_is_detection, stats = apply_prefix_right_censoring_torch(
        opt_t=opt_t,
        opt_v=opt_v,
        opt_mask=opt_mask,
        opt_err=opt_err,
        slot_is_detection=slot_is_detection,
        target_k=target_k,
        min_det=int(args.prefix_min_det),
    )

    valid_rows = opt_mask.sum(dim=-1) > 0
    keep_prob = torch.empty((opt_mask.size(0), 1), device=opt_mask.device).uniform_(
        float(args.view_keep_prob_min),
        float(args.view_keep_prob_max),
    )
    row_keep = (~valid_rows) | (torch.rand(valid_rows.shape, device=opt_mask.device) < keep_prob)
    det_rows = (slot_is_detection > 0) & valid_rows
    det_after_row = (det_rows & row_keep).sum(dim=1)
    failed_row = det_after_row < int(args.prefix_min_det)
    if bool(failed_row.any().item()):
        row_keep[failed_row] = row_keep[failed_row] | det_rows[failed_row]

    row_keep_3d = row_keep.unsqueeze(-1)
    opt_t = torch.where(row_keep, opt_t, torch.zeros_like(opt_t))
    opt_v = torch.where(row_keep_3d, opt_v, torch.zeros_like(opt_v))
    opt_mask = torch.where(row_keep_3d, opt_mask, torch.zeros_like(opt_mask))
    opt_err = torch.where(row_keep_3d, opt_err, torch.zeros_like(opt_err))
    slot_is_detection = torch.where(row_keep, slot_is_detection, torch.zeros_like(slot_is_detection))

    pre_band_opt_v = opt_v.clone()
    pre_band_opt_mask = opt_mask.clone()
    pre_band_opt_err = opt_err.clone()
    pre_band_slot = slot_is_detection.clone()

    observed_bands = opt_mask.sum(dim=1) > 0
    band_drop = torch.empty((opt_mask.size(0), 1), device=opt_mask.device).uniform_(
        0.0,
        float(args.view_band_dropout_max),
    )
    band_keep = torch.rand((opt_mask.size(0), opt_mask.size(2)), device=opt_mask.device) >= band_drop
    band_keep = band_keep & observed_bands
    missing_any_band = ~band_keep.any(dim=1)
    if bool(missing_any_band.any().item()):
        band_counts = pre_band_opt_mask.sum(dim=1)
        best_band = torch.argmax(band_counts, dim=1)
        band_keep[missing_any_band] = False
        band_keep[missing_any_band, best_band[missing_any_band]] = True
    band_keep_3d = band_keep.unsqueeze(1)
    opt_v = torch.where(band_keep_3d, opt_v, torch.zeros_like(opt_v))
    opt_mask = torch.where(band_keep_3d, opt_mask, torch.zeros_like(opt_mask))
    opt_err = torch.where(band_keep_3d, opt_err, torch.zeros_like(opt_err))
    slot_is_detection = torch.where(opt_mask.sum(dim=-1) > 0, slot_is_detection, torch.zeros_like(slot_is_detection))

    det_after_band = ((slot_is_detection > 0) & (opt_mask.sum(dim=-1) > 0)).sum(dim=1)
    failed_band = det_after_band < int(args.prefix_min_det)
    if bool(failed_band.any().item()):
        opt_v[failed_band] = pre_band_opt_v[failed_band]
        opt_mask[failed_band] = pre_band_opt_mask[failed_band]
        opt_err[failed_band] = pre_band_opt_err[failed_band]
        slot_is_detection[failed_band] = pre_band_slot[failed_band]

    n_det, n_bands, t_span = compute_batch_sequence_meta(opt_t, opt_mask, slot_is_detection)
    return {
        "opt_t": opt_t,
        "opt_v": opt_v,
        "opt_mask": opt_mask,
        "opt_err": opt_err,
        "slot_is_detection": slot_is_detection,
        "target_k": target_k,
        "is_terminal_prefix": (target_k >= total_det).to(dtype=torch.long),
        "n_det": n_det.to(dtype=torch.long),
        "n_bands": n_bands.to(dtype=torch.long),
        "t_span": t_span,
        "bucket_n_det": bucketize_n_det(n_det.to(dtype=torch.long)),
        "bucket_n_bands": bucketize_n_bands(n_bands.to(dtype=torch.long)),
        "bucket_t_span": bucketize_t_span(t_span),
    }


def symmetric_bernoulli_kl_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    pa = torch.sigmoid(logits_a).clamp(min=eps, max=1.0 - eps)
    pb = torch.sigmoid(logits_b).clamp(min=eps, max=1.0 - eps)
    kl_ab = pa * torch.log(pa / pb) + (1.0 - pa) * torch.log((1.0 - pa) / (1.0 - pb))
    kl_ba = pb * torch.log(pb / pa) + (1.0 - pb) * torch.log((1.0 - pb) / (1.0 - pa))
    return 0.5 * (kl_ab.mean() + kl_ba.mean())


def compute_universal_loss(
    view_a: Dict[str, torch.Tensor],
    view_b: Optional[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    criterion,
    args,
    stage_mode: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits_a = view_a["logits"]
    cls_a = criterion(logits_a, labels)
    metrics = {
        "cls_a": float(cls_a.detach().item()),
    }
    if view_b is None:
        return cls_a, metrics

    logits_b = view_b["logits"]
    cls_b = criterion(logits_b, labels)
    loss = 0.5 * (cls_a + cls_b)
    metrics["cls_b"] = float(cls_b.detach().item())

    if stage_mode in {"dual", "dual_adv"}:
        cons_embed = 1.0 - F.cosine_similarity(view_a["proj_feat"], view_b["proj_feat"], dim=-1).mean()
        cons_prob = symmetric_bernoulli_kl_from_logits(logits_a, logits_b)
        loss = (
            loss
            + float(args.consistency_embed_weight) * cons_embed
            + float(args.consistency_prob_weight) * cons_prob
        )
        metrics["cons_embed"] = float(cons_embed.detach().item())
        metrics["cons_prob"] = float(cons_prob.detach().item())

    if stage_mode == "dual_adv":
        adv_criterion = nn.CrossEntropyLoss()
        adv_det = 0.5 * (
            adv_criterion(view_a["adv_logits_n_det"], view_a["bucket_n_det"])
            + adv_criterion(view_b["adv_logits_n_det"], view_b["bucket_n_det"])
        )
        adv_band = 0.5 * (
            adv_criterion(view_a["adv_logits_n_bands"], view_a["bucket_n_bands"])
            + adv_criterion(view_b["adv_logits_n_bands"], view_b["bucket_n_bands"])
        )
        adv_span = 0.5 * (
            adv_criterion(view_a["adv_logits_t_span"], view_a["bucket_t_span"])
            + adv_criterion(view_b["adv_logits_t_span"], view_b["bucket_t_span"])
        )
        loss = (
            loss
            + float(args.adv_det_weight) * adv_det
            + float(args.adv_band_weight) * adv_band
            + float(args.adv_span_weight) * adv_span
        )
        metrics["adv_det"] = float(adv_det.detach().item())
        metrics["adv_band"] = float(adv_band.detach().item())
        metrics["adv_span"] = float(adv_span.detach().item())

    return loss, metrics


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


def forward_logits_with_offsets(
    model,
    opt_t,
    opt_v,
    ref_time,
    opt_mask,
    opt_err,
    device,
    amp_dtype,
    args,
    offset_policy,
    training,
):
    out = forward_outputs_with_offsets(
        model=model,
        opt_t=opt_t,
        opt_v=opt_v,
        ref_time=ref_time,
        opt_mask=opt_mask,
        opt_err=opt_err,
        device=device,
        amp_dtype=amp_dtype,
        args=args,
        offset_policy=offset_policy,
        training=training,
        return_aux=False,
    )
    return out["logits"]


def forward_outputs_with_offsets(
    model,
    opt_t,
    opt_v,
    ref_time,
    opt_mask,
    opt_err,
    device,
    amp_dtype,
    args,
    offset_policy,
    training,
    return_aux,
):
    if not offset_policy.enabled:
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            out = model.forward_with_aux(opt_t, opt_v, ref_time, opt_mask, opt_err, return_aux=return_aux)
        out["logits"] = out["logits"].squeeze(-1).float()
        return out

    if training:
        delta_np = offset_policy.sample_train_offsets(opt_t.size(0))
        delta_days = torch.from_numpy(delta_np).to(device=device, dtype=torch.float32)
        shifted_opt_t = apply_time_offsets(opt_t, opt_mask, delta_days, args.offset_scale_days_divisor)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            out = model.forward_with_aux(
                shifted_opt_t,
                opt_v,
                ref_time,
                opt_mask,
                opt_err,
                return_aux=return_aux,
            )
        out["logits"] = out["logits"].squeeze(-1).float()
        return out

    logits_sum = None
    for off_days in offset_policy.eval_offsets_days:
        delta_days = torch.full((opt_t.size(0),), float(off_days), device=device, dtype=torch.float32)
        shifted_opt_t = apply_time_offsets(opt_t, opt_mask, delta_days, args.offset_scale_days_divisor)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            out = model.forward_with_aux(
                shifted_opt_t,
                opt_v,
                ref_time,
                opt_mask,
                opt_err,
                return_aux=return_aux,
            )
        logits = out["logits"].squeeze(-1).float()
        logits_sum = logits if logits_sum is None else logits_sum + logits

    return {
        "logits": logits_sum / float(max(1, len(offset_policy.eval_offsets_days))),
    }


def run_eval(model, loader, device, args, criterion, amp_dtype, offset_policy):
    model.eval()
    losses = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []
    all_n_det = []
    all_n_bands = []
    all_t_span = []
    all_actual_target_k = []
    all_is_terminal_prefix = []
    ref_time_cache = None

    with torch.no_grad():
        for batch in loader:
            batch_dict = unpack_optical_batch(batch)
            opt_t = batch_dict["opt_t"]
            opt_v = batch_dict["opt_v"]
            opt_mask = batch_dict["opt_mask"]
            opt_err = batch_dict["opt_err"]
            labels = batch_dict["labels"]
            slot_is_detection = batch_dict["slot_is_detection"]
            actual_target_k = batch_dict["actual_target_k"]
            is_terminal_prefix = batch_dict["is_terminal_prefix"]
            opt_t = opt_t.to(device, non_blocking=True)
            opt_v = opt_v.to(device, non_blocking=True)
            opt_mask = opt_mask.to(device, non_blocking=True)
            opt_err = opt_err.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            if slot_is_detection is not None:
                slot_is_detection = slot_is_detection.to(device, non_blocking=True)
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
                ref_time_cache = build_ref_time(
                    batch_size, args.n_ref, args.ref_start, args.ref_end, device, opt_t.dtype
                )

            logits = forward_logits_with_offsets(
                model=model,
                opt_t=opt_t,
                opt_v=opt_v,
                ref_time=ref_time_cache,
                opt_mask=opt_mask,
                opt_err=opt_err,
                device=device,
                amp_dtype=amp_dtype,
                args=args,
                offset_policy=offset_policy,
                training=False,
            )
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            losses += loss.item()
            n_batches += 1
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.detach().cpu().long())
            n_det, n_bands, t_span = compute_batch_sequence_meta(opt_t, opt_mask, slot_is_detection)
            all_n_det.append(n_det.detach().cpu())
            all_n_bands.append(n_bands.detach().cpu())
            all_t_span.append(t_span.detach().cpu())
            if actual_target_k is not None:
                all_actual_target_k.append(actual_target_k.detach().cpu())
            if is_terminal_prefix is not None:
                all_is_terminal_prefix.append(is_terminal_prefix.detach().cpu())

    if n_batches == 0:
        model.train()
        return None

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    cls_metrics = compute_classification_metrics(probs, labels)
    op = select_threshold_for_target_recall(probs, labels, target_recall=args.target_recall)
    shortcut_metrics: Dict[str, float] = {}
    if bool(getattr(args, "shortcut_audit_enable", False)):
        n_det_np = torch.cat(all_n_det, dim=0).numpy()
        n_bands_np = torch.cat(all_n_bands, dim=0).numpy()
        t_span_np = torch.cat(all_t_span, dim=0).numpy()
        labels_np = labels.numpy()
        shortcut_metrics = compute_shortcut_audit(
            n_det=n_det_np,
            n_bands=n_bands_np,
            t_span=t_span_np,
            labels=labels_np,
            max_samples=int(getattr(args, "shortcut_audit_val_samples", 50000)),
        )
    prefix_bucket_metrics = build_prefix_bucket_metrics(
        probs=probs,
        labels=labels,
        actual_target_k=(torch.cat(all_actual_target_k, dim=0) if all_actual_target_k else None),
        is_terminal_prefix=(torch.cat(all_is_terminal_prefix, dim=0) if all_is_terminal_prefix else None),
    )

    model.train()
    out = {
        "loss": losses / n_batches,
        "auroc": float(cls_metrics.get("auroc", 0.0)),
        "auprc": float(cls_metrics.get("auprc", 0.0)),
        "f1_optimal": float(cls_metrics.get("f1_optimal", 0.0)),
        "op_threshold": op["threshold"],
        "op_recall": op["recall"],
        "op_precision": op["precision"],
        "op_fpr": op["fpr"],
        "op_meets_target_recall": bool(op["meets_target_recall"]),
        "offset_eval_count": int(len(offset_policy.eval_offsets_days)),
        "task_mode": "prefix_right_censored"
        if (bool(getattr(args, "prefix_train_enable", False)) or bool(getattr(args, "universal_train_enable", False)))
        else "full_window",
    }
    for k, v in shortcut_metrics.items():
        out[f"shortcut_{k}"] = float(v)
    if prefix_bucket_metrics:
        out["prefix_bucket_metrics"] = prefix_bucket_metrics
    del all_probs, all_labels, all_n_det, all_n_bands, all_t_span
    del all_actual_target_k, all_is_terminal_prefix
    del probs, labels, cls_metrics, op, shortcut_metrics, prefix_bucket_metrics
    del ref_time_cache
    run_memory_maintenance(
        device,
        loaders=None,
        recycle_workers=False,
        close_loader_files=False,
        trim_heap=bool(getattr(args, "memory_trim_enable", True)),
    )
    return out


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    args,
    criterion,
    scaler,
    amp_dtype,
    offset_policy,
    prefix_policy,
    universal_stage_mode: Optional[str] = None,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    ref_time_cache = None
    step_cleanup_interval = max(0, int(getattr(args, "step_memory_cleanup_interval", 0)))

    pbar = tqdm(loader, desc="Train", mininterval=0.0, miniters=100)
    for batch in pbar:
        batch_dict = unpack_optical_batch(batch)
        opt_t = batch_dict["opt_t"]
        opt_v = batch_dict["opt_v"]
        opt_mask = batch_dict["opt_mask"]
        opt_err = batch_dict["opt_err"]
        labels = batch_dict["labels"]
        slot_is_detection = batch_dict["slot_is_detection"]
        opt_t = opt_t.to(device, non_blocking=True)
        opt_v = opt_v.to(device, non_blocking=True)
        opt_mask = opt_mask.to(device, non_blocking=True)
        opt_err = opt_err.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()
        if slot_is_detection is not None:
            slot_is_detection = slot_is_detection.to(device, non_blocking=True)
        target_k = None

        if bool(getattr(args, "universal_train_enable", False)):
            if slot_is_detection is None:
                raise ValueError("universal_train_enable=true requires slot_is_detection in training batches.")
            if not getattr(model, "universal_aux_enable", False):
                raise ValueError("universal_train_enable=true requires model.universal_aux_enable=True.")
        elif bool(getattr(args, "prefix_train_enable", False)):
            if slot_is_detection is None:
                raise ValueError("prefix_train_enable=true requires slot_is_detection in training batches.")
            target_k = prefix_policy.sample_target_k(slot_is_detection)
            opt_t, opt_v, opt_mask, opt_err, slot_is_detection, _ = apply_prefix_right_censoring_torch(
                opt_t=opt_t,
                opt_v=opt_v,
                opt_mask=opt_mask,
                opt_err=opt_err,
                slot_is_detection=slot_is_detection,
                target_k=target_k,
                min_det=int(args.prefix_min_det),
            )

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
                single_band_keep_prob=args.single_band_keep_prob,
                target_ndet_jitter=0.0,
            )
        else:
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
                single_band_keep_prob=args.single_band_keep_prob,
                target_ndet_jitter=args.target_ndet_jitter,
            )

        batch_size = opt_t.size(0)
        if (
            ref_time_cache is None
            or ref_time_cache.shape[0] != batch_size
            or ref_time_cache.dtype != opt_t.dtype
        ):
            ref_time_cache = build_ref_time(
                batch_size, args.n_ref, args.ref_start, args.ref_end, device, opt_t.dtype
            )

        optimizer.zero_grad(set_to_none=True)
        if bool(getattr(args, "universal_train_enable", False)):
            view_a = build_universal_view(opt_t, opt_v, opt_mask, opt_err, slot_is_detection, args)
            out_a = forward_outputs_with_offsets(
                model=model,
                opt_t=view_a["opt_t"],
                opt_v=view_a["opt_v"],
                ref_time=ref_time_cache,
                opt_mask=view_a["opt_mask"],
                opt_err=view_a["opt_err"],
                device=device,
                amp_dtype=amp_dtype,
                args=args,
                offset_policy=offset_policy,
                training=True,
                return_aux=(universal_stage_mode in {"dual", "dual_adv"}),
            )
            out_a.update(
                {
                    "bucket_n_det": view_a["bucket_n_det"],
                    "bucket_n_bands": view_a["bucket_n_bands"],
                    "bucket_t_span": view_a["bucket_t_span"],
                }
            )
            out_b = None
            if universal_stage_mode in {"dual", "dual_adv"}:
                view_b = build_universal_view(opt_t, opt_v, opt_mask, opt_err, slot_is_detection, args)
                out_b = forward_outputs_with_offsets(
                    model=model,
                    opt_t=view_b["opt_t"],
                    opt_v=view_b["opt_v"],
                    ref_time=ref_time_cache,
                    opt_mask=view_b["opt_mask"],
                    opt_err=view_b["opt_err"],
                    device=device,
                    amp_dtype=amp_dtype,
                    args=args,
                    offset_policy=offset_policy,
                    training=True,
                    return_aux=True,
                )
                out_b.update(
                    {
                        "bucket_n_det": view_b["bucket_n_det"],
                        "bucket_n_bands": view_b["bucket_n_bands"],
                        "bucket_t_span": view_b["bucket_t_span"],
                    }
                )
            loss, loss_metrics = compute_universal_loss(
                view_a=out_a,
                view_b=out_b,
                labels=labels,
                criterion=criterion,
                args=args,
                stage_mode=("single" if universal_stage_mode is None else universal_stage_mode),
            )
        else:
            logits = forward_logits_with_offsets(
                model=model,
                opt_t=opt_t,
                opt_v=opt_v,
                ref_time=ref_time_cache,
                opt_mask=opt_mask,
                opt_err=opt_err,
                device=device,
                amp_dtype=amp_dtype,
                args=args,
                offset_policy=offset_policy,
                training=True,
            )
            loss = criterion(logits, labels)
            loss_metrics = None

        scaler.scale(loss).backward()
        if args.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        if n_batches % 50 == 0:
            postfix = {"loss": f"{(total_loss / n_batches):.4f}"}
            if loss_metrics and "cons_embed" in loss_metrics:
                postfix["cons"] = f"{loss_metrics['cons_embed']:.3f}"
            if loss_metrics and "adv_det" in loss_metrics:
                postfix["adv"] = f"{loss_metrics['adv_det']:.3f}"
            pbar.set_postfix(postfix)

        del batch, batch_dict, opt_t, opt_v, opt_mask, opt_err, labels, loss
        if slot_is_detection is not None:
            del slot_is_detection
        if target_k is not None:
            del target_k
        if bool(getattr(args, "universal_train_enable", False)):
            del view_a, out_a
            if out_b is not None:
                del out_b, view_b
        else:
            del logits
        if step_cleanup_interval > 0 and (n_batches % step_cleanup_interval) == 0:
            run_memory_maintenance(
                device,
                loaders=None,
                recycle_workers=False,
                close_loader_files=False,
                trim_heap=False,
            )

    pbar.close()
    del ref_time_cache
    return total_loss / max(1, n_batches)


def save_checkpoint(path, model, optimizer, epoch, args, val_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "val_metrics": val_metrics,
            "args": vars(args),
        },
        path,
    )


def build_train_summary_payload(
    *,
    run_name: str,
    save_root: Path,
    args,
    offset_policy,
    prefix_policy,
    best_score: float,
    best_epoch: int,
    best_metrics: Dict[str, object],
    current_epoch: int,
    current_stage: str,
    last_epoch_metrics: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    payload = {
        "run_name": run_name,
        "save_root": str(save_root),
        "arch_version": str(args.arch_version),
        "best_epoch": int(best_epoch),
        "best_auroc_plus_auprc": float(best_score),
        "best_precision_at_target_recall": float(best_metrics.get("op_precision", 0.0)) if best_metrics else 0.0,
        "time_offset": offset_policy.describe(),
        "prefix_task": prefix_policy.describe(),
        "meta_filter": {
            "n_det_min": (
                None if getattr(args, "meta_filter_n_det_min", None) is None else int(args.meta_filter_n_det_min)
            ),
            "n_det_max": (
                None if getattr(args, "meta_filter_n_det_max", None) is None else int(args.meta_filter_n_det_max)
            ),
            "n_bands_max": (
                None
                if getattr(args, "meta_filter_n_bands_max", None) is None
                else int(args.meta_filter_n_bands_max)
            ),
            "t_span_max": (
                None
                if getattr(args, "meta_filter_t_span_max", None) is None
                else float(args.meta_filter_t_span_max)
            ),
            "relax_t_span_if_below_rows": parse_relax_t_span_thresholds(
                getattr(args, "meta_filter_relax_t_span_if_below_rows", None)
            ),
        },
        "real_stream_profile_path": getattr(args, "real_stream_profile_path", None),
        "task_mode": "prefix_right_censored"
        if (bool(args.prefix_train_enable) or bool(getattr(args, "universal_train_enable", False)))
        else "full_window",
        "current_epoch": int(current_epoch),
        "current_stage": str(current_stage),
    }
    if last_epoch_metrics:
        payload["last_epoch_metrics"] = dict(last_epoch_metrics)
    if best_metrics:
        payload.update(best_metrics)
    return payload


def write_train_summary_files(save_root: Path, payload: Dict[str, object]) -> None:
    for name in ("train_summary.json", "best_checkpoint_summary.json"):
        with (save_root / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def init_tensorboard_writer(args, save_root):
    if bool(args.disable_tensorboard):
        print("TensorBoard logging disabled by --disable_tensorboard.")
        return None

    if SummaryWriter is None:
        print("TensorBoard is unavailable (missing tensorboard package). Skipping TB logging.")
        return None

    log_dir = Path(args.tb_log_dir) if args.tb_log_dir else (save_root / "tb_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=int(args.tb_flush_secs))
    writer.add_text("run/config", json.dumps(vars(args), indent=2), global_step=0)
    print(f"TensorBoard log dir: {log_dir}")
    return writer


def resolve_run_name(args):
    raw_name = None
    if args.run_name is not None and str(args.run_name).strip() != "":
        raw_name = str(args.run_name).strip()
    else:
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "").strip()
        if slurm_job_id != "":
            raw_name = f"job{slurm_job_id}"
        else:
            raw_name = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")

    run_name = raw_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    if run_name in {"", ".", ".."}:
        raise ValueError(f"Invalid run_name resolved from '{raw_name}'.")
    return run_name


def build_stage_optimizer(model, args, freeze_encoder):
    head_params = list(model.classifier.parameters())
    if getattr(model, "universal_aux_enable", False):
        head_params += list(model.projection_head.parameters())
        head_params += list(model.adv_head_n_det.parameters())
        head_params += list(model.adv_head_n_bands.parameters())
        head_params += list(model.adv_head_t_span.parameters())
    if freeze_encoder:
        model.set_encoder_trainable(False)
        return torch.optim.AdamW(
            head_params,
            lr=args.lr_head_stage1,
            weight_decay=args.weight_decay,
        )

    model.set_encoder_trainable(True)
    return torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_head_stage2},
            {"params": model.optical_encoder.parameters(), "lr": args.lr_encoder_stage2},
        ],
        weight_decay=args.weight_decay,
    )


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    offset_policy = TimeOffsetPolicy(args)
    print(f"Time offset policy: {json.dumps(offset_policy.describe(), indent=2)}")
    prefix_policy = PrefixTrainPolicy(args)
    print(f"Prefix train policy: {json.dumps(prefix_policy.describe(), indent=2)}")
    if bool(args.prefix_train_enable) and float(args.target_ndet_jitter) > 0:
        print("prefix_train_enable=true: target_ndet_jitter will be ignored in favor of causal right-censoring.")
    meta_bins_n_det = parse_bin_edges(
        getattr(args, "meta_bins_n_det", None),
        default="3,5,8,12,20,40,80,200",
    )
    meta_bins_n_bands = parse_bin_edges(
        getattr(args, "meta_bins_n_bands", None),
        default="1,2,3,4,5,6",
    )
    meta_bins_t_span = parse_bin_edges(
        getattr(args, "meta_bins_t_span", None),
        default="0,0.01,0.05,0.1,0.2,0.5,1.0",
    )
    print(
        "Meta matched sampling: "
        f"enable={bool(args.meta_matched_sampling)} "
        f"fallback={args.meta_match_fallback} "
        f"bins_det={meta_bins_n_det} bins_band={meta_bins_n_bands} bins_tspan={meta_bins_t_span}"
    )
    relax_thresholds = parse_relax_t_span_thresholds(getattr(args, "meta_filter_relax_t_span_if_below_rows", None))
    print(
        "Meta filter policy: "
        f"n_det=[{getattr(args, 'meta_filter_n_det_min', None)},{getattr(args, 'meta_filter_n_det_max', None)}] "
        f"| n_bands<={getattr(args, 'meta_filter_n_bands_max', None)} "
        f"| t_span<={getattr(args, 'meta_filter_t_span_max', None)} "
        f"| relax_if_below_rows={json.dumps(relax_thresholds)} "
        f"| real_stream_profile={getattr(args, 'real_stream_profile_path', None)}"
    )

    train_loader, val_loader, steps_per_epoch, val_steps = create_optical_binary_dataloaders(
        pos_h5_path=args.pos_data_path,
        neg_h5_path=args.neg_data_path,
        neg_group=args.neg_group,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        steps_per_epoch=args.steps_per_epoch,
        val_steps_per_epoch=args.val_steps_per_epoch,
        val_split=args.val_split,
        split_seed=args.split_seed,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=args.prefetch_factor,
        cache_in_memory=bool(args.cache_in_memory),
        meta_matched_sampling=bool(args.meta_matched_sampling),
        meta_match_fallback=str(args.meta_match_fallback),
        meta_bins_n_det=meta_bins_n_det,
        meta_bins_n_bands=meta_bins_n_bands,
        meta_bins_t_span=meta_bins_t_span,
        prefix_train_enable=(bool(args.prefix_train_enable) or bool(getattr(args, "universal_train_enable", False))),
        prefix_min_det=int(args.prefix_min_det),
        prefix_eval_det_support=str(args.prefix_eval_det_support),
        prefix_eval_include_terminal=True,
        meta_filter_n_det_min=getattr(args, "meta_filter_n_det_min", None),
        meta_filter_n_det_max=getattr(args, "meta_filter_n_det_max", None),
        meta_filter_n_bands_max=getattr(args, "meta_filter_n_bands_max", None),
        meta_filter_t_span_max=getattr(args, "meta_filter_t_span_max", None),
        meta_filter_relax_t_span_if_below_rows=int(relax_thresholds["train"]),
    )
    print(f"Train Steps/Epoch: {steps_per_epoch} | Val Steps/Epoch: {val_steps}")

    model = OpticalKNClassifier(
        optical_input_dim=6,
        ref_time_dim=args.ref_dim,
        enc_dim=args.enc_dim,
        opt_dropout=args.opt_dropout,
        feature_dropout=args.feature_dropout,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        universal_aux_enable=bool(getattr(args, "universal_train_enable", False)),
        proj_dim=64,
        adv_hidden_dim=(args.head_hidden_dim if args.head_hidden_dim is not None else args.enc_dim),
        n_det_bucket_classes=5,
        n_bands_bucket_classes=4,
        t_span_bucket_classes=5,
        grl_lambda=float(getattr(args, "grl_lambda", 1.0)),
    ).to(device)

    if args.pretrained_albef_ckpt:
        if not os.path.exists(args.pretrained_albef_ckpt):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_albef_ckpt}")
        ckpt = torch.load(args.pretrained_albef_ckpt, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_optical_encoder_from_albef_state_dict(state_dict, strict=False)
        print(
            "Loaded optical encoder from ALBEF checkpoint "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )

    criterion = nn.BCEWithLogitsLoss()

    if bool(getattr(args, "universal_train_enable", False)):
        stage_plan = [
            ("stage1_head_only", int(getattr(args, "universal_stage1_epochs", 2)), True, "single"),
            ("stage2_finetune", int(getattr(args, "universal_stage2_epochs", 6)), False, "dual"),
            ("stage3_finetune_adv", int(getattr(args, "universal_stage3_epochs", 6)), False, "dual_adv"),
        ]
    else:
        stage_plan = [
            ("stage1_head_only", int(args.epochs_stage1), True, None),
            ("stage2_finetune", int(args.epochs_stage2), False, None),
        ]

    best_score = float("-inf")
    best_metrics = {}
    best_epoch = -1
    global_epoch = 0
    no_improve = 0
    last_epoch_metrics: Dict[str, object] = {}

    run_name = resolve_run_name(args)
    args.run_name = run_name
    print(f"Architecture version: {args.arch_version} (requires nocoord-compatible checkpoint)")

    save_root = Path(args.ckpt_path) / "optical_only" / run_name
    save_root.mkdir(parents=True, exist_ok=True)
    epoch_ckpt_root = save_root / "epoch_ckpts"
    epoch_ckpt_root.mkdir(parents=True, exist_ok=True)
    latest_run_file = save_root.parent / "latest_run.txt"
    latest_run_file.write_text(f"{run_name}\n", encoding="utf-8")

    print(f"Run name: {run_name}")
    print(f"Checkpoint directory: {save_root}")
    tb_writer = init_tensorboard_writer(args, save_root)
    recycle_every = max(0, int(getattr(args, "recycle_dataloader_workers_every_n_epochs", 1)))
    cleanup_enable = bool(getattr(args, "memory_cleanup_enable", True))
    trim_heap_enable = bool(getattr(args, "memory_trim_enable", True))
    cleanup_close_files = bool(getattr(args, "memory_cleanup_close_loader_files", True))
    if cleanup_enable:
        print(
            "Memory cleanup policy: "
            f"recycle_workers_every_n_epochs={recycle_every}, "
            f"close_loader_files={cleanup_close_files}, trim_heap={trim_heap_enable}, "
            f"step_cleanup_interval={int(getattr(args, 'step_memory_cleanup_interval', 0))}"
        )

    optimizer = None
    for stage_name, stage_epochs, freeze_encoder, universal_stage_mode in stage_plan:
        if stage_epochs <= 0:
            continue

        optimizer = build_stage_optimizer(model, args, freeze_encoder=freeze_encoder)
        if stage_name in {"stage2_finetune", "stage3_finetune_adv"}:
            no_improve = 0

        print(
            f"\n[{stage_name}] epochs={stage_epochs} freeze_encoder={freeze_encoder} "
            f"lr_head={optimizer.param_groups[0]['lr']}"
        )

        for _ in range(stage_epochs):
            global_epoch += 1
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args,
                criterion,
                scaler,
                amp_dtype,
                offset_policy,
                prefix_policy,
                universal_stage_mode=universal_stage_mode,
            )
            val_metrics = run_eval(model, val_loader, device, args, criterion, amp_dtype, offset_policy)
            if val_metrics is None:
                epoch_metrics = {
                    "stage": stage_name,
                    "epoch": global_epoch,
                    "train_loss": train_loss,
                }
                save_checkpoint(
                    str(epoch_ckpt_root / f"optical_only_epoch_{global_epoch:04d}.pth"),
                    model,
                    optimizer,
                    global_epoch,
                    args,
                    epoch_metrics,
                )
                print(
                    f"Saved epoch checkpoint: {epoch_ckpt_root / f'optical_only_epoch_{global_epoch:04d}.pth'} "
                    "(val_metrics unavailable)"
                )
                continue

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", float(train_loss), global_epoch)
                tb_writer.add_scalar("train/lr_head", float(optimizer.param_groups[0]["lr"]), global_epoch)
                if len(optimizer.param_groups) > 1:
                    tb_writer.add_scalar("train/lr_encoder", float(optimizer.param_groups[1]["lr"]), global_epoch)
                tb_writer.add_scalar(
                    "train/stage_id",
                    1.0 if stage_name == "stage1_head_only" else 2.0,
                    global_epoch,
                )

                tb_writer.add_scalar("val/loss", float(val_metrics["loss"]), global_epoch)
                tb_writer.add_scalar("val/auroc", float(val_metrics["auroc"]), global_epoch)
                tb_writer.add_scalar("val/auprc", float(val_metrics["auprc"]), global_epoch)
                tb_writer.add_scalar("val/f1_optimal", float(val_metrics["f1_optimal"]), global_epoch)
                tb_writer.add_scalar("val/op_threshold", float(val_metrics["op_threshold"]), global_epoch)
                tb_writer.add_scalar("val/op_recall", float(val_metrics["op_recall"]), global_epoch)
                tb_writer.add_scalar("val/op_precision", float(val_metrics["op_precision"]), global_epoch)
                tb_writer.add_scalar("val/op_fpr", float(val_metrics["op_fpr"]), global_epoch)
                tb_writer.add_scalar(
                    "val/op_meets_target_recall",
                    1.0 if val_metrics["op_meets_target_recall"] else 0.0,
                    global_epoch,
                )

            score = float(val_metrics["auroc"] + val_metrics["auprc"])
            if tb_writer is not None:
                tb_writer.add_scalar("val/score_auroc_plus_auprc", score, global_epoch)
            improved = score > (best_score + args.early_stop_min_delta)

            print(
                f"Epoch {global_epoch} | stage={stage_name} "
                f"| train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} "
                f"| AUROC={val_metrics['auroc']:.4f} | AUPRC={val_metrics['auprc']:.4f} "
                f"| thr={val_metrics['op_threshold']:.3f} "
                f"| recall={val_metrics['op_recall']:.4f} "
                f"| precision={val_metrics['op_precision']:.4f} "
                f"| fpr={val_metrics['op_fpr']:.4f} "
                f"| eval_k={val_metrics['offset_eval_count']}"
            )

            epoch_metrics = {
                "stage": stage_name,
                "epoch": global_epoch,
                "train_loss": train_loss,
                "score_auroc_plus_auprc": score,
                **val_metrics,
            }
            last_epoch_metrics = dict(epoch_metrics)
            save_checkpoint(
                str(epoch_ckpt_root / f"optical_only_epoch_{global_epoch:04d}.pth"),
                model,
                optimizer,
                global_epoch,
                args,
                epoch_metrics,
            )

            if improved:
                best_score = score
                best_epoch = global_epoch
                best_metrics = {
                    "stage": stage_name,
                    "epoch": global_epoch,
                    "train_loss": train_loss,
                    **val_metrics,
                }
                save_checkpoint(
                    str(save_root / "optical_only_best.pth"),
                    model,
                    optimizer,
                    global_epoch,
                    args,
                    best_metrics,
                )
                no_improve = 0
                print(
                    f"Saved best checkpoint: {save_root / 'optical_only_best.pth'} "
                    f"(auroc+auprc={best_score:.4f})"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("best/score_auroc_plus_auprc", float(best_score), global_epoch)
                    tb_writer.add_scalar("best/op_precision", float(val_metrics["op_precision"]), global_epoch)
                    tb_writer.add_scalar("best/auroc", float(val_metrics["auroc"]), global_epoch)
                    tb_writer.add_scalar("best/auprc", float(val_metrics["auprc"]), global_epoch)
                    tb_writer.add_scalar("best/epoch", float(best_epoch), global_epoch)
            else:
                if stage_name in {"stage2_finetune", "stage3_finetune_adv"}:
                    no_improve += 1
                    if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                        print(f"Early stopping triggered in {stage_name}.")
                        break

            write_train_summary_files(
                save_root,
                build_train_summary_payload(
                    run_name=run_name,
                    save_root=save_root,
                    args=args,
                    offset_policy=offset_policy,
                    prefix_policy=prefix_policy,
                    best_score=best_score,
                    best_epoch=best_epoch,
                    best_metrics=best_metrics,
                    current_epoch=global_epoch,
                    current_stage=stage_name,
                    last_epoch_metrics=last_epoch_metrics,
                ),
            )

            if cleanup_enable:
                maintenance = run_memory_maintenance(
                    device,
                    loaders=[train_loader, val_loader],
                    recycle_workers=(recycle_every > 0 and (global_epoch % recycle_every) == 0),
                    close_loader_files=cleanup_close_files,
                    trim_heap=trim_heap_enable,
                )
                print(
                    f"Epoch {global_epoch} memory cleanup | "
                    f"gc={maintenance['gc_collected']} "
                    f"| workers_recycled={maintenance['workers_recycled']} "
                    f"| file_handles_closed={maintenance['file_handles_closed']} "
                    f"| heap_trimmed={maintenance['heap_trimmed']}"
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        if (
            stage_name in {"stage2_finetune", "stage3_finetune_adv"}
            and args.early_stop_patience > 0
            and no_improve >= args.early_stop_patience
        ):
            break

    final_metrics = {
        "run_name": run_name,
        "save_root": str(save_root),
        "arch_version": str(args.arch_version),
        "best_epoch": best_epoch,
        "best_auroc_plus_auprc": best_score,
        "best_precision_at_target_recall": float(best_metrics.get("op_precision", 0.0)) if best_metrics else 0.0,
        "time_offset": offset_policy.describe(),
        "prefix_task": prefix_policy.describe(),
        "task_mode": "prefix_right_censored"
        if (bool(args.prefix_train_enable) or bool(getattr(args, "universal_train_enable", False)))
        else "full_window",
        **best_metrics,
    }
    write_train_summary_files(
        save_root,
        build_train_summary_payload(
            run_name=run_name,
            save_root=save_root,
            args=args,
            offset_policy=offset_policy,
            prefix_policy=prefix_policy,
            best_score=best_score,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            current_epoch=global_epoch,
            current_stage="complete",
            last_epoch_metrics=last_epoch_metrics,
        ),
    )

    if optimizer is not None:
        save_checkpoint(
            str(save_root / "optical_only_last.pth"),
            model,
            optimizer,
            global_epoch,
            args,
            final_metrics,
        )

    run_memory_maintenance(
        device,
        loaders=[train_loader, val_loader],
        recycle_workers=True,
        close_loader_files=True,
        trim_heap=trim_heap_enable,
    )

    if tb_writer is not None:
        try:
            tb_writer.add_text("run/final_metrics", json.dumps(final_metrics, indent=2), global_step=global_epoch)
            tb_writer.flush()
            tb_writer.close()
        except Exception:
            pass

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Summary: {save_root / 'train_summary.json'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--pos_data_path", type=str, default=None)
    parser.add_argument("--neg_data_path", type=str, default=None)
    parser.add_argument("--neg_group", type=str, default="ELASTICC2/optical_data")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--pretrained_albef_ckpt", type=str, default=None)

    parser.add_argument("--epochs_stage1", type=int, default=5)
    parser.add_argument("--epochs_stage2", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--val_steps_per_epoch", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)

    parser.add_argument("--lr_head_stage1", type=float, default=1e-3)
    parser.add_argument("--lr_head_stage2", type=float, default=3e-4)
    parser.add_argument("--lr_encoder_stage2", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)

    parser.add_argument("--n_ref", type=int, default=64)
    parser.add_argument("--ref_start", type=float, default=-0.3)
    parser.add_argument("--ref_end", type=float, default=0.6)
    parser.add_argument("--ref_dim", type=int, default=64)
    parser.add_argument("--enc_dim", type=int, default=64)

    parser.add_argument("--opt_dropout", type=float, default=0.1)
    parser.add_argument("--feature_dropout", type=float, default=0.0)
    parser.add_argument("--head_hidden_dim", type=int, default=None)
    parser.add_argument("--head_dropout", type=float, default=0.2)
    parser.add_argument("--arch_version", type=str, default="optical_only_nocoord_v1")

    parser.add_argument("--opt_aug_noise", type=float, default=0.0)
    parser.add_argument("--opt_aug_time_jitter", type=float, default=0.0)
    parser.add_argument("--opt_aug_dropout", type=float, default=0.35)
    parser.add_argument("--opt_aug_band_dropout", type=float, default=0.30)
    parser.add_argument("--single_band_keep_prob", type=float, default=0.35)
    parser.add_argument("--target_ndet_jitter", type=float, default=0.0)
    parser.add_argument("--prefix_train_enable", action="store_true")
    parser.add_argument("--prefix_min_det", type=int, default=2)
    parser.add_argument("--prefix_train_sampling", type=str, default="real_stream_ndet_empirical")
    parser.add_argument("--prefix_terminal_mix_prob", type=float, default=0.2)
    parser.add_argument("--prefix_real_mix_weight", type=float, default=4.0)
    parser.add_argument("--prefix_bucket_uniform_mix_weight", type=float, default=4.0)
    parser.add_argument("--prefix_terminal_mix_weight", type=float, default=2.0)
    parser.add_argument("--prefix_eval_det_support", type=str, default="2,3,4,5,6,8,10,12")
    parser.add_argument("--prefix_real_hist_path", type=str, default=None)
    parser.add_argument("--universal_train_enable", action="store_true")
    parser.add_argument("--universal_stage1_epochs", type=int, default=2)
    parser.add_argument("--universal_stage2_epochs", type=int, default=6)
    parser.add_argument("--universal_stage3_epochs", type=int, default=6)
    parser.add_argument("--view_keep_prob_min", type=float, default=0.55)
    parser.add_argument("--view_keep_prob_max", type=float, default=1.0)
    parser.add_argument("--view_band_dropout_max", type=float, default=0.5)
    parser.add_argument("--consistency_embed_weight", type=float, default=0.10)
    parser.add_argument("--consistency_prob_weight", type=float, default=0.05)
    parser.add_argument("--adv_det_weight", type=float, default=0.05)
    parser.add_argument("--adv_band_weight", type=float, default=0.03)
    parser.add_argument("--adv_span_weight", type=float, default=0.05)
    parser.add_argument("--grl_lambda", type=float, default=1.0)

    parser.add_argument("--meta_matched_sampling", type=int, default=1)
    parser.add_argument("--meta_match_fallback", type=str, default="nearest")
    parser.add_argument("--meta_bins_n_det", type=str, default="3,5,8,12,20,40,80,200")
    parser.add_argument("--meta_bins_n_bands", type=str, default="1,2,3,4,5,6")
    parser.add_argument("--meta_bins_t_span", type=str, default="0,0.01,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--meta_filter_n_det_min", type=int, default=None)
    parser.add_argument("--meta_filter_n_det_max", type=int, default=None)
    parser.add_argument("--meta_filter_n_bands_max", type=int, default=None)
    parser.add_argument("--meta_filter_t_span_max", type=float, default=None)
    parser.add_argument("--meta_filter_relax_t_span_if_below_rows", type=str, default=None)
    parser.add_argument("--real_stream_profile_path", type=str, default=None)

    parser.add_argument("--shortcut_audit_enable", type=int, default=1)
    parser.add_argument("--shortcut_audit_val_samples", type=int, default=50000)

    parser.add_argument("--target_recall", type=float, default=0.98)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--persistent_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--memory_cleanup_enable", type=int, default=1)
    parser.add_argument("--memory_trim_enable", type=int, default=1)
    parser.add_argument("--memory_cleanup_close_loader_files", type=int, default=1)
    parser.add_argument("--recycle_dataloader_workers_every_n_epochs", type=int, default=1)
    parser.add_argument("--step_memory_cleanup_interval", type=int, default=0)
    parser.add_argument("--cache_in_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--tb_log_dir", type=str, default=None)
    parser.add_argument("--tb_flush_secs", type=int, default=30)
    parser.add_argument("--disable_tensorboard", action="store_true")

    parser.add_argument("--time_offset_enable", action="store_true")
    parser.add_argument("--offset_dist_npz", type=str, default=None)
    parser.add_argument("--offset_dist_key", type=str, default="delta_days_combined")
    parser.add_argument("--offset_train_sampling", type=str, default="empirical_cdf")
    parser.add_argument("--offset_eval_mode", type=str, default="quantile_ensemble")
    parser.add_argument("--offset_eval_quantiles", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--offset_scale_days_divisor", type=float, default=100.0)
    parser.add_argument("--offset_seed", type=int, default=None)
    parser.add_argument("--offset_bank_size", type=int, default=1000000)
    parser.add_argument("--ood_reject_enable", action="store_true")
    parser.add_argument("--ood_uncertainty_metric", type=str, default="logit_std")
    parser.add_argument("--ood_uncertainty_threshold", type=float, default=0.75)
    parser.add_argument("--regime_eval_enable", action="store_true")

    pre_args, _ = parser.parse_known_args()
    if pre_args.config is not None:
        config_path = Path(pre_args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError("Config JSON must be an object.")
        parser.set_defaults(**config)

    args = parser.parse_args()
    if args.offset_seed is None:
        args.offset_seed = int(args.seed)
    if not (0.0 <= float(args.single_band_keep_prob) <= 1.0):
        raise ValueError("--single_band_keep_prob must be in [0, 1].")
    if float(args.target_ndet_jitter) < 0:
        raise ValueError("--target_ndet_jitter must be >= 0.")
    if int(args.recycle_dataloader_workers_every_n_epochs) < 0:
        raise ValueError("--recycle_dataloader_workers_every_n_epochs must be >= 0.")
    if int(args.step_memory_cleanup_interval) < 0:
        raise ValueError("--step_memory_cleanup_interval must be >= 0.")
    if int(args.prefix_min_det) < 1:
        raise ValueError("--prefix_min_det must be >= 1.")
    if not (0.0 <= float(args.prefix_terminal_mix_prob) <= 1.0):
        raise ValueError("--prefix_terminal_mix_prob must be in [0, 1].")
    if float(args.prefix_real_mix_weight) < 0:
        raise ValueError("--prefix_real_mix_weight must be >= 0.")
    if float(args.prefix_bucket_uniform_mix_weight) < 0:
        raise ValueError("--prefix_bucket_uniform_mix_weight must be >= 0.")
    if float(args.prefix_terminal_mix_weight) < 0:
        raise ValueError("--prefix_terminal_mix_weight must be >= 0.")
    sampling_mode = str(args.prefix_train_sampling).strip().lower()
    if sampling_mode not in {"real_stream_ndet_empirical", "real_stream_bucket_uniform_terminal_mixture"}:
        raise ValueError(
            "--prefix_train_sampling must be 'real_stream_ndet_empirical' or "
            "'real_stream_bucket_uniform_terminal_mixture'."
        )
    if (
        sampling_mode == "real_stream_bucket_uniform_terminal_mixture"
        and float(args.prefix_real_mix_weight)
        + float(args.prefix_bucket_uniform_mix_weight)
        + float(args.prefix_terminal_mix_weight)
        <= 0
    ):
        raise ValueError("Prefix mixture weights must sum to > 0.")
    parse_prefix_det_support(args.prefix_eval_det_support)
    if str(args.meta_match_fallback).strip().lower() not in {"nearest", "random"}:
        raise ValueError("--meta_match_fallback must be 'nearest' or 'random'.")
    if args.meta_filter_n_det_min is not None and args.meta_filter_n_det_max is not None:
        if int(args.meta_filter_n_det_min) > int(args.meta_filter_n_det_max):
            raise ValueError("--meta_filter_n_det_min must be <= --meta_filter_n_det_max.")
    if args.meta_filter_n_bands_max is not None and int(args.meta_filter_n_bands_max) < 1:
        raise ValueError("--meta_filter_n_bands_max must be >= 1.")
    if args.meta_filter_t_span_max is not None and float(args.meta_filter_t_span_max) <= 0:
        raise ValueError("--meta_filter_t_span_max must be > 0.")
    parse_relax_t_span_thresholds(args.meta_filter_relax_t_span_if_below_rows)
    if not (0.0 < float(args.view_keep_prob_min) <= float(args.view_keep_prob_max) <= 1.0):
        raise ValueError("view_keep_prob_min/view_keep_prob_max must satisfy 0 < min <= max <= 1.")
    if not (0.0 <= float(args.view_band_dropout_max) <= 1.0):
        raise ValueError("--view_band_dropout_max must be in [0, 1].")
    if float(args.consistency_embed_weight) < 0 or float(args.consistency_prob_weight) < 0:
        raise ValueError("Consistency weights must be >= 0.")
    if float(args.adv_det_weight) < 0 or float(args.adv_band_weight) < 0 or float(args.adv_span_weight) < 0:
        raise ValueError("Adversarial weights must be >= 0.")
    if int(args.universal_stage1_epochs) < 0 or int(args.universal_stage2_epochs) < 0 or int(args.universal_stage3_epochs) < 0:
        raise ValueError("Universal stage epochs must be >= 0.")
    if float(args.grl_lambda) < 0:
        raise ValueError("--grl_lambda must be >= 0.")
    return args


if __name__ == "__main__":
    args = parse_args()
    for key in ("pos_data_path", "neg_data_path", "ckpt_path"):
        if getattr(args, key) is None:
            raise ValueError(f"{key} must be provided via CLI or --config.")
    if not os.path.exists(args.pos_data_path):
        raise FileNotFoundError(f"Positive data file not found: {args.pos_data_path}")
    if not os.path.exists(args.neg_data_path):
        raise FileNotFoundError(f"Negative data file not found: {args.neg_data_path}")
    if bool(args.time_offset_enable):
        if args.offset_dist_npz is None:
            raise ValueError("time_offset_enable=true requires --offset_dist_npz.")
        if not os.path.exists(args.offset_dist_npz):
            raise FileNotFoundError(f"Offset distribution file not found: {args.offset_dist_npz}")
    if bool(args.prefix_train_enable):
        if args.prefix_real_hist_path is None:
            raise ValueError("prefix_train_enable=true requires --prefix_real_hist_path.")
        if not os.path.exists(args.prefix_real_hist_path):
            raise FileNotFoundError(f"Prefix real-stream histogram file not found: {args.prefix_real_hist_path}")
    if args.real_stream_profile_path is not None and str(args.real_stream_profile_path).strip() != "":
        if not os.path.exists(args.real_stream_profile_path):
            raise FileNotFoundError(f"Real-stream profile file not found: {args.real_stream_profile_path}")

    os.makedirs(args.ckpt_path, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] Starting optical-only KN training (fd_t0)")
    print(json.dumps(vars(args), indent=2))
    train(args)
