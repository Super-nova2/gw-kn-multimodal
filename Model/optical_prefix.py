#!/usr/bin/env python3
"""
Utilities for right-censored optical prefix classification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


DEFAULT_PREFIX_DET_SUPPORT = (2, 3, 4, 5, 6, 8, 10, 12)
PREFIX_TIME_EPS = 1.0e-8


def parse_prefix_det_support(text: str | Sequence[int] | None) -> List[int]:
    if text is None:
        return [int(v) for v in DEFAULT_PREFIX_DET_SUPPORT]
    if isinstance(text, (list, tuple)):
        vals = [int(v) for v in text]
    else:
        vals = []
        for part in str(text).split(","):
            part = part.strip()
            if not part:
                continue
            vals.append(int(part))
    vals = sorted(set(int(v) for v in vals if int(v) >= 1))
    if not vals:
        raise ValueError("prefix_det_support must contain at least one positive integer.")
    return vals


def compute_sample_prefix_stats_numpy(
    opt_t: np.ndarray,
    opt_mask: np.ndarray,
    slot_is_detection: np.ndarray,
    target_k: int,
    min_det: int = 2,
    time_eps: float = PREFIX_TIME_EPS,
) -> Dict[str, object]:
    time_vec = np.asarray(opt_t, dtype=np.float64).reshape(-1)
    mask_mat = np.asarray(opt_mask, dtype=np.float32)
    det_vec = np.asarray(slot_is_detection, dtype=np.float32).reshape(-1)
    if mask_mat.ndim != 2:
        raise ValueError(f"opt_mask must be [T,B], got shape={mask_mat.shape}")
    if mask_mat.shape[0] != time_vec.shape[0]:
        raise ValueError("opt_t and opt_mask first dimensions must match.")
    if det_vec.shape[0] != time_vec.shape[0]:
        raise ValueError("slot_is_detection length must match opt_t length.")

    valid_rows = np.asarray(mask_mat.sum(axis=1) > 0, dtype=bool)
    det_rows = np.asarray((det_vec > 0) & valid_rows, dtype=bool)
    total_det = int(det_rows.sum())
    if total_det < int(min_det):
        raise ValueError(
            f"Sample has only {total_det} detections, below required min_det={int(min_det)}."
        )

    actual_target_k = int(max(int(min_det), min(int(target_k), total_det)))
    det_indices = np.flatnonzero(det_rows)
    cut_row = int(det_indices[actual_target_k - 1])
    cut_time = float(time_vec[cut_row])
    keep_rows = np.asarray(valid_rows & (time_vec <= (cut_time + float(time_eps))), dtype=bool)

    keep_mask = keep_rows[:, None]
    kept_det = np.asarray(det_rows & keep_rows, dtype=bool)
    kept_mask = np.where(keep_mask, mask_mat, 0.0).astype(np.float32, copy=False)
    band_hits = kept_mask.sum(axis=0) > 0
    kept_times = time_vec[keep_rows]
    t_last = float(np.max(kept_times)) if kept_times.size > 0 else 0.0
    t_span = float(np.max(kept_times) - np.min(kept_times)) if kept_times.size > 0 else 0.0

    return {
        "keep_rows": keep_rows,
        "cut_row": int(cut_row),
        "cut_time": float(cut_time),
        "target_k": int(target_k),
        "actual_target_k": int(actual_target_k),
        "total_det": int(total_det),
        "actual_n_det_snr5": int(kept_det.sum()),
        "actual_n_obs": int(keep_rows.sum()),
        "actual_n_bands": int(band_hits.sum()),
        "t_last": float(t_last),
        "t_span": float(t_span),
        "is_terminal_prefix": bool(actual_target_k >= total_det),
    }


def apply_prefix_right_censoring_numpy(
    opt_t: np.ndarray,
    opt_v: np.ndarray,
    opt_mask: np.ndarray,
    opt_err: np.ndarray | None,
    slot_is_detection: np.ndarray,
    target_k: int,
    min_det: int = 2,
    time_eps: float = PREFIX_TIME_EPS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, Dict[str, object]]:
    stats = compute_sample_prefix_stats_numpy(
        opt_t=opt_t,
        opt_mask=opt_mask,
        slot_is_detection=slot_is_detection,
        target_k=target_k,
        min_det=min_det,
        time_eps=time_eps,
    )
    keep_rows = np.asarray(stats["keep_rows"], dtype=bool)
    row_keep = keep_rows[:, None]

    out_t = np.asarray(opt_t, dtype=np.float32).copy()
    out_v = np.asarray(opt_v, dtype=np.float32).copy()
    out_mask = np.asarray(opt_mask, dtype=np.float32).copy()
    out_det = np.asarray(slot_is_detection, dtype=np.float32).copy()
    out_err = None if opt_err is None else np.asarray(opt_err, dtype=np.float32).copy()

    out_t[~keep_rows] = 0.0
    out_v[~keep_rows, :] = 0.0
    out_mask[~keep_rows, :] = 0.0
    out_det[~keep_rows] = 0.0
    if out_err is not None:
        out_err[~keep_rows, :] = 0.0
    return out_t, out_v, out_mask, out_err, out_det, stats


def apply_prefix_right_censoring_torch(
    opt_t: torch.Tensor,
    opt_v: torch.Tensor,
    opt_mask: torch.Tensor,
    opt_err: torch.Tensor | None,
    slot_is_detection: torch.Tensor,
    target_k: torch.Tensor,
    min_det: int = 2,
    time_eps: float = PREFIX_TIME_EPS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, Dict[str, torch.Tensor]]:
    if opt_t.ndim != 2:
        raise ValueError(f"opt_t must be [B,T], got shape={tuple(opt_t.shape)}")
    if opt_v.ndim != 3 or opt_mask.ndim != 3:
        raise ValueError("opt_v and opt_mask must be [B,T,C].")
    if slot_is_detection.ndim != 2:
        raise ValueError("slot_is_detection must be [B,T].")

    row_valid = opt_mask.sum(dim=-1) > 0
    det_rows = (slot_is_detection > 0) & row_valid
    det_counts = det_rows.sum(dim=1)
    if bool((det_counts < int(min_det)).any().item()):
        raise ValueError("apply_prefix_right_censoring_torch received samples below prefix_min_det.")

    target_k = target_k.to(device=opt_t.device, dtype=torch.long).reshape(-1)
    actual_k = torch.minimum(target_k, det_counts.to(dtype=torch.long))
    actual_k = torch.clamp(actual_k, min=int(min_det))

    det_cumsum = det_rows.to(dtype=torch.long).cumsum(dim=1)
    cutoff_hits = det_rows & (det_cumsum >= actual_k[:, None])
    cutoff_idx = torch.argmax(cutoff_hits.to(dtype=torch.int64), dim=1)
    row_idx = torch.arange(opt_t.size(0), device=opt_t.device)
    cutoff_time = opt_t[row_idx, cutoff_idx]

    keep_rows = row_valid & (opt_t <= (cutoff_time[:, None] + float(time_eps)))
    keep_rows_3d = keep_rows.unsqueeze(-1)

    out_t = torch.where(keep_rows, opt_t, torch.zeros_like(opt_t))
    out_v = torch.where(keep_rows_3d, opt_v, torch.zeros_like(opt_v))
    out_mask = torch.where(keep_rows_3d, opt_mask, torch.zeros_like(opt_mask))
    out_det = torch.where(keep_rows, slot_is_detection, torch.zeros_like(slot_is_detection))
    out_err = None if opt_err is None else torch.where(keep_rows_3d, opt_err, torch.zeros_like(opt_err))

    actual_n_det = (out_det > 0).sum(dim=1)
    actual_n_obs = keep_rows.sum(dim=1)
    actual_n_bands = (out_mask.sum(dim=1) > 0).sum(dim=1)
    valid_after = keep_rows
    first_idx = torch.argmax(valid_after.to(torch.int64), dim=1)
    last_idx = valid_after.size(1) - 1 - torch.argmax(valid_after.flip(dims=[1]).to(torch.int64), dim=1)
    t_first = out_t[row_idx, first_idx]
    t_last = out_t[row_idx, last_idx]
    t_span = torch.where(valid_after.any(dim=1), t_last - t_first, torch.zeros_like(t_last))

    stats = {
        "target_k": target_k,
        "actual_target_k": actual_k,
        "total_det": det_counts.to(dtype=torch.long),
        "cutoff_idx": cutoff_idx.to(dtype=torch.long),
        "cutoff_time": cutoff_time.to(dtype=opt_t.dtype),
        "actual_n_det_snr5": actual_n_det.to(dtype=torch.long),
        "actual_n_obs": actual_n_obs.to(dtype=torch.long),
        "actual_n_bands": actual_n_bands.to(dtype=torch.long),
        "t_last": t_last.to(dtype=opt_t.dtype),
        "t_span": t_span.to(dtype=opt_t.dtype),
        "is_terminal_prefix": (actual_k >= det_counts.to(dtype=torch.long)),
    }
    return out_t, out_v, out_mask, out_err, out_det, stats


def load_prefix_ndet_distribution(
    json_path: str | Path,
    min_det: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    hist = None
    if isinstance(raw, dict):
        for key in ("counts", "hist", "n_det_counts", "distribution"):
            if key in raw and isinstance(raw[key], dict):
                hist = raw[key]
                break
        if hist is None:
            hist = raw
    else:
        raise ValueError(f"Unsupported prefix ndet histogram JSON format: {path}")

    keys: List[int] = []
    vals: List[float] = []
    for k, v in hist.items():
        kk = int(k)
        vv = float(v)
        if kk < int(min_det) or vv <= 0:
            continue
        keys.append(int(kk))
        vals.append(float(vv))

    if not keys:
        raise ValueError(f"No valid n_det counts >= {int(min_det)} found in {path}")

    support = np.asarray(keys, dtype=np.int64)
    probs = np.asarray(vals, dtype=np.float64)
    probs = probs / np.sum(probs)
    meta = dict(raw) if isinstance(raw, dict) else {}
    meta["support"] = [int(v) for v in support.tolist()]
    meta["probabilities"] = [float(v) for v in probs.tolist()]
    return support, probs, meta


def sample_prefix_target_k(
    det_counts: np.ndarray,
    support: Sequence[int],
    probs: Sequence[float],
    rng: np.random.Generator,
    min_det: int = 2,
    terminal_mix_prob: float = 0.0,
) -> np.ndarray:
    det_counts = np.asarray(det_counts, dtype=np.int64).reshape(-1)
    if det_counts.size == 0:
        return np.asarray([], dtype=np.int64)
    support_arr = np.asarray(support, dtype=np.int64).reshape(-1)
    prob_arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    if support_arr.size == 0 or prob_arr.size != support_arr.size:
        raise ValueError("support/probs must be non-empty and have the same length.")
    if np.any(det_counts < int(min_det)):
        raise ValueError("sample_prefix_target_k received det_counts below prefix_min_det.")

    sampled = np.empty(det_counts.shape[0], dtype=np.int64)
    for i, det_count in enumerate(det_counts.tolist()):
        feasible = support_arr <= int(det_count)
        if not np.any(feasible):
            sampled[i] = int(det_count)
            continue
        local_support = support_arr[feasible]
        local_probs = prob_arr[feasible]
        local_probs = local_probs / np.sum(local_probs)
        sampled[i] = int(rng.choice(local_support, replace=True, p=local_probs))

    mix_prob = float(terminal_mix_prob)
    if mix_prob > 0:
        terminal_mask = rng.random(det_counts.shape[0]) < mix_prob
        sampled[terminal_mask] = det_counts[terminal_mask]
    return sampled.astype(np.int64, copy=False)


def build_prefix_manifest_entries(
    base_idx: int,
    label: int,
    opt_t: np.ndarray,
    opt_mask: np.ndarray,
    slot_is_detection: np.ndarray,
    det_support: Sequence[int],
    min_det: int = 2,
    include_terminal: bool = True,
) -> List[Dict[str, object]]:
    det_rows = (np.asarray(slot_is_detection, dtype=np.float32) > 0) & (
        np.asarray(opt_mask, dtype=np.float32).sum(axis=1) > 0
    )
    total_det = int(det_rows.sum())
    if total_det < int(min_det):
        return []

    entries: List[Dict[str, object]] = []
    support = [int(k) for k in det_support if int(min_det) <= int(k) <= total_det]
    terminal_k = int(total_det)
    if include_terminal and terminal_k not in support:
        support = support + [terminal_k]

    seen = set()
    for k in support:
        if k in seen:
            continue
        seen.add(int(k))
        stats = compute_sample_prefix_stats_numpy(
            opt_t=opt_t,
            opt_mask=opt_mask,
            slot_is_detection=slot_is_detection,
            target_k=int(k),
            min_det=min_det,
        )
        entries.append(
            {
                "base_idx": int(base_idx),
                "label": int(label),
                "prefix_det_target_k": int(k),
                "prefix_cut_time": float(stats["cut_time"]),
                "actual_n_det_snr5": int(stats["actual_n_det_snr5"]),
                "actual_n_obs": int(stats["actual_n_obs"]),
                "actual_n_bands": int(stats["actual_n_bands"]),
                "is_terminal_prefix": bool(int(k) >= total_det),
            }
        )
    return entries
