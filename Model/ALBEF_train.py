from data_loader import (
    create_training_dataloader,
    create_train_val_dataloaders,
    create_supcon_dataloaders,
    build_effective_input_window_metadata,
    build_gw_to_lc_mapping,
)
from model import GWOpticalALBEFModel, normalize_fusion_mode, migrate_time_embed_state_dict
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
import math
import gc
import json
import time
import warnings
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
        k: v
        for k, v in cfg.items()
        if k not in BASH_ONLY_CONFIG_KEYS and v is not None
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


DEFAULT_MTAN_LUPT_M5 = np.asarray([23.9, 25.0, 24.7, 24.0, 23.3, 22.1], dtype=np.float64)


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
            raise ValueError(f"Invalid hard-negative time window value: {v}. Must be > 0.")
        vals.append(v)
    if not vals:
        raise ValueError("hardneg_time_window_days produced empty list.")
    vals = sorted(set(vals))
    return vals


def parse_lupt_m5_mag_text(text: str) -> np.ndarray:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 6:
        raise ValueError("mtan_lupt_m5_mag must provide exactly 6 comma-separated values in order u,g,r,i,z,Y.")
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


def _collect_dataset_window_metadata(args) -> Dict[str, object]:
    return {
        "train_positive": _read_optical_h5_window_metadata(getattr(args, "data_path", None)),
        "train_negative": _read_optical_h5_window_metadata(getattr(args, "neg_data_path", None)),
        "eval_positive": _read_optical_h5_window_metadata(getattr(args, "test_data_path", None)),
        "eval_negative": _read_optical_h5_window_metadata(getattr(args, "neg_data_path", None)),
    }


def _build_effective_input_window_metadata(args) -> Dict[str, object]:
    pos_meta = getattr(args, "_dataset_window_metadata", {}).get("train_positive", {})
    dataset_window_start = pos_meta.get("time_window_start") if isinstance(pos_meta, dict) else None
    dataset_window_end = pos_meta.get("time_window_end") if isinstance(pos_meta, dict) else None
    return build_effective_input_window_metadata(
        float(args.ref_start),
        float(args.ref_end),
        runtime_input_window_start=float(args.ref_start),
        runtime_input_window_end=float(args.ref_end),
        dataset_window_start=(None if dataset_window_start is None else float(dataset_window_start)),
        dataset_window_end=(None if dataset_window_end is None else float(dataset_window_end)),
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
                    m5 = np.asarray(f.attrs["lupt_m5_mag"], dtype=np.float64).reshape(-1)
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
    shift = (delta_days.to(device=opt_t.device, dtype=opt_t.dtype) / float(scale_divisor)).unsqueeze(1)
    return opt_t + shift * valid


def apply_hard_negative_time_shift(opt_t, opt_mask, delta_days, scale_divisor):
    """
    Shift optical timeline for hard negatives with absolute mismatch preserved.

    t_shift = t + (delta_days / scale_divisor), where:
      delta_days = t_candidate_gw - t_anchor_gw
    """
    valid = (opt_mask.sum(dim=-1) > 0).to(dtype=opt_t.dtype)
    shift = (delta_days.to(device=opt_t.device, dtype=opt_t.dtype) / float(scale_divisor)).unsqueeze(1)
    return opt_t + shift * valid


# NOTE: apply_time_offsets() and apply_hard_negative_time_shift() are kept as reusable
# primitives for potential future time-shift experiments, but are no longer called in
# the main training / eval pipeline after the offset infra removal.


def compute_time_delta_days(opt_zero_time_mjd, gw_anchor_time_mjd):
    dt = opt_zero_time_mjd.to(torch.float32) - gw_anchor_time_mjd.to(torch.float32)
    dt = torch.where(torch.isfinite(dt), dt, torch.zeros_like(dt))
    return dt


def resolve_optical_candidate_time_mjd(opt_first_detection_mjd, fallback_event_time_mjd):
    """Return the absolute time anchor for optical candidates."""
    if opt_first_detection_mjd is not None:
        return opt_first_detection_mjd
    return fallback_event_time_mjd


def gather_candidate_optical_time_mjd(opt_first_detection_mjd, fallback_event_time_mjd, candidate_idx):
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
        delta_days = torch.zeros((opt_t.size(0),), device=opt_t.device, dtype=torch.float32)
        shifted_opt_t = opt_t
    else:
        delta_days = compute_time_delta_days(candidate_zero_time_mjd, anchor_gw_time_mjd)
        shifted_opt_t = apply_hard_negative_time_shift(opt_t, opt_mask, delta_days, scale_divisor)
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
        out[f"{key}_std"] = float(var ** 0.5)
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


def _sample_semi_hard_from_candidates(
    sim_row,
    same_event_row,
    cand_idx,
    margin,
):
    if cand_idx.numel() == 0:
        return None
    pos_vals = sim_row[same_event_row]
    if pos_vals.numel() == 0:
        return cand_idx[sim_row[cand_idx].argmax()]
    pos_med = torch.median(pos_vals)
    cand_sims = sim_row[cand_idx]
    lower = (1.0 - margin) * pos_med
    band_mask = (cand_sims >= lower) & (cand_sims <= pos_med)
    band_idx = cand_idx[band_mask]
    if band_idx.numel() > 0:
        return band_idx[cand_sims[band_mask].argmax()]
    return cand_idx[cand_sims.argmax()]


def sample_inbatch_hard_negatives_with_time(
    sim_g2o,
    gw_indices,
    batch_event_time_mjd,
    window_days,
    min_candidates,
    semi_hard,
    semi_hard_margin,
    fallback_mode,
    candidate_time_mjd=None,
    active_rows=None,
):
    sim = sim_g2o.detach()
    batch_size = sim.size(0)
    device = sim.device
    if active_rows is None:
        active_rows = torch.arange(batch_size, dtype=torch.long, device=device)
    else:
        active_rows = active_rows.to(device=device, dtype=torch.long)
    n_active = int(active_rows.numel())
    if n_active <= 0:
        return torch.empty((0,), dtype=torch.long, device=device), [0 for _ in window_days], 0

    same_event = gw_indices.unsqueeze(0) == gw_indices.unsqueeze(1)  # [B, B]
    candidate_mask = ~same_event[active_rows]  # [A, B]
    neg_idx = torch.empty((n_active,), dtype=torch.long, device=device)
    unresolved = torch.ones((n_active,), dtype=torch.bool, device=device)
    level_hits = [0 for _ in window_days]
    fallback_count = 0

    finite_anchor = None
    if batch_event_time_mjd is not None:
        anchor_times = batch_event_time_mjd[active_rows]  # [A]
        candidate_times = (
            batch_event_time_mjd
            if candidate_time_mjd is None
            else candidate_time_mjd.to(device=device, dtype=torch.float32)
        )
        finite_anchor = torch.isfinite(anchor_times)
        if finite_anchor.any():
            dt_matrix = torch.abs(candidate_times.unsqueeze(0) - anchor_times.unsqueeze(1))  # [A, B]
            for lvl, wnd in enumerate(window_days):
                mask_lvl = candidate_mask & (dt_matrix <= float(wnd))
                cand_count = mask_lvl.sum(dim=1)
                is_last = (lvl == len(window_days) - 1)
                accepted = unresolved & finite_anchor & (
                    (cand_count >= int(min_candidates)) | (is_last & (cand_count > 0))
                )
                if not accepted.any():
                    continue
                acc_rows = torch.nonzero(accepted, as_tuple=False).squeeze(-1)
                masked_sim = sim[active_rows[acc_rows]].masked_fill(~mask_lvl[acc_rows], -1e9)
                neg_idx[acc_rows] = masked_sim.argmax(dim=1)
                unresolved[acc_rows] = False
                level_hits[lvl] += int(acc_rows.numel())

    if unresolved.any():
        unresolved_rows = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
        fallback_count = int(unresolved_rows.numel())
        fallback_mode_l = str(fallback_mode).strip().lower()
        for ridx in unresolved_rows.tolist():
            global_row = int(active_rows[ridx].item())
            candidate_all = torch.nonzero(candidate_mask[ridx], as_tuple=False).squeeze(-1)
            if candidate_all.numel() == 0:
                neg_idx[ridx] = global_row
                continue
            if fallback_mode_l == "inbatch_semihard" or semi_hard:
                chosen = _sample_semi_hard_from_candidates(
                    sim[global_row], same_event[global_row], candidate_all, semi_hard_margin
                )
            else:
                chosen = candidate_all[sim[global_row, candidate_all].argmax()]
            if chosen is None:
                chosen = candidate_all[0]
            neg_idx[ridx] = chosen

    return neg_idx, level_hits, int(fallback_count)


class HardNegativeMemoryBank:
    """
    Memory bank for hard-negative optical candidates.

    Stores projected optical embeddings and full optical features on GPU for
    fast similarity search and direct fusion feature retrieval.
    """

    def __init__(self, args, device):
        self.enabled = bool(args.hardneg_memory_bank_enable)
        self.capacity = max(1, int(args.hardneg_memory_bank_size))
        self.topk = max(1, int(args.hardneg_memory_topk))
        self.warmup_steps = max(0, int(args.hardneg_memory_warmup_steps))
        self.device = device

        self.ptr = 0
        self.size = 0
        self.initialized = False

        self.feat_o_bank = None
        self.gw_time_bank = None
        self.gw_idx_bank = None
        self.h_l_bank = None
        self.z_l_bank = None
        self.coords_bank = None
        self.zero_time_bank = None
        self.opt_t_bank = None
        self.opt_v_bank = None
        self.opt_mask_bank = None
        self.opt_err_bank = None

    def _ensure_initialized(self, feat_o, h_l, z_l, opt_coords, opt_t, opt_v, opt_mask, opt_err):
        if self.initialized:
            return
        d_proj = int(feat_o.size(1))
        seq_len = int(h_l.size(1))
        d_enc = int(h_l.size(2))
        d_z = int(z_l.size(1))
        n_time = int(opt_t.size(1))
        n_band = int(opt_v.size(2))
        self.feat_o_bank = torch.empty(
            (self.capacity, d_proj), device=self.device, dtype=torch.float16
        )
        self.gw_time_bank = torch.empty(
            (self.capacity,), device=self.device, dtype=torch.float32
        )
        self.gw_idx_bank = torch.empty(
            (self.capacity,), device=self.device, dtype=torch.long
        )
        self.h_l_bank = torch.empty(
            (self.capacity, seq_len, d_enc), device=self.device, dtype=torch.float16
        )
        self.z_l_bank = torch.empty(
            (self.capacity, d_z), device=self.device, dtype=torch.float16
        )
        self.coords_bank = torch.empty(
            (self.capacity, int(opt_coords.size(1))), device=self.device, dtype=torch.float32
        )
        self.zero_time_bank = torch.empty(
            (self.capacity,), device=self.device, dtype=torch.float32
        )
        self.opt_t_bank = torch.empty(
            (self.capacity, n_time), device=self.device, dtype=torch.float32
        )
        self.opt_v_bank = torch.empty(
            (self.capacity, n_time, n_band), device=self.device, dtype=torch.float16
        )
        self.opt_mask_bank = torch.empty(
            (self.capacity, n_time, n_band), device=self.device, dtype=torch.float16
        )
        self.opt_err_bank = torch.empty(
            (self.capacity, n_time, n_band), device=self.device, dtype=torch.float16
        )
        self.initialized = True

    def update(
        self,
        feat_o,
        h_l,
        z_l,
        opt_coords,
        gw_indices,
        opt_t,
        opt_v,
        opt_mask,
        opt_err,
        candidate_time_mjd=None,
        candidate_gw_time_mjd=None,
    ):
        if not self.enabled:
            return
        if (
            feat_o is None
            or h_l is None
            or z_l is None
            or (candidate_time_mjd is None and candidate_gw_time_mjd is None)
            or opt_t is None
            or opt_v is None
            or opt_mask is None
            or opt_err is None
        ):
            return
        self._ensure_initialized(feat_o, h_l, z_l, opt_coords, opt_t, opt_v, opt_mask, opt_err)

        n = int(feat_o.size(0))
        if n <= 0:
            return

        feat_o_w = feat_o.detach().to(device=self.device, dtype=torch.float16)
        candidate_time = candidate_time_mjd if candidate_time_mjd is not None else candidate_gw_time_mjd
        candidate_time_w = candidate_time.detach().to(device=self.device, dtype=torch.float32)
        gw_idx_w = gw_indices.detach().to(device=self.device, dtype=torch.long)
        h_l_w = h_l.detach().to(device=self.device, dtype=torch.float16)
        z_l_w = z_l.detach().to(device=self.device, dtype=torch.float16)
        coords_w = opt_coords.detach().to(device=self.device, dtype=torch.float32)
        zero_w = candidate_time_w
        opt_t_w = opt_t.detach().to(device=self.device, dtype=torch.float32)
        opt_v_w = opt_v.detach().to(device=self.device, dtype=torch.float16)
        opt_mask_w = opt_mask.detach().to(device=self.device, dtype=torch.float16)
        opt_err_w = opt_err.detach().to(device=self.device, dtype=torch.float16)

        first = min(n, self.capacity - self.ptr)
        sl1 = slice(self.ptr, self.ptr + first)
        self.feat_o_bank[sl1] = feat_o_w[:first]
        self.gw_time_bank[sl1] = candidate_time_w[:first]
        self.gw_idx_bank[sl1] = gw_idx_w[:first]
        self.h_l_bank[sl1] = h_l_w[:first]
        self.z_l_bank[sl1] = z_l_w[:first]
        self.coords_bank[sl1] = coords_w[:first]
        self.zero_time_bank[sl1] = zero_w[:first]
        self.opt_t_bank[sl1] = opt_t_w[:first]
        self.opt_v_bank[sl1] = opt_v_w[:first]
        self.opt_mask_bank[sl1] = opt_mask_w[:first]
        self.opt_err_bank[sl1] = opt_err_w[:first]

        if n > first:
            rem = n - first
            sl2 = slice(0, rem)
            self.feat_o_bank[sl2] = feat_o_w[first:]
            self.gw_time_bank[sl2] = candidate_time_w[first:]
            self.gw_idx_bank[sl2] = gw_idx_w[first:]
            self.h_l_bank[sl2] = h_l_w[first:]
            self.z_l_bank[sl2] = z_l_w[first:]
            self.coords_bank[sl2] = coords_w[first:]
            self.zero_time_bank[sl2] = zero_w[first:]
            self.opt_t_bank[sl2] = opt_t_w[first:]
            self.opt_v_bank[sl2] = opt_v_w[first:]
            self.opt_mask_bank[sl2] = opt_mask_w[first:]
            self.opt_err_bank[sl2] = opt_err_w[first:]

        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.capacity, self.size + n)

    def mine(
        self,
        feat_g,
        gw_indices,
        batch_event_time_mjd,
        temperature,
        window_days,
        min_candidates,
        global_step,
        active_rows=None,
    ):
        batch_size = int(feat_g.size(0))
        if active_rows is None:
            active_rows = torch.arange(batch_size, dtype=torch.long, device=feat_g.device)
        else:
            active_rows = active_rows.to(device=feat_g.device, dtype=torch.long)
        n_active = int(active_rows.numel())
        sel = torch.full((n_active,), -1, dtype=torch.long, device=feat_g.device)
        level_hits = [0 for _ in window_days]
        fallback_count = 0
        attempted = 0

        if (
            (not self.enabled)
            or (not self.initialized)
            or self.size <= 0
            or int(global_step) < self.warmup_steps
            or batch_event_time_mjd is None
            or n_active <= 0
        ):
            return sel, level_hits, fallback_count, attempted

        n_bank = int(self.size)
        bank_feat = self.feat_o_bank[:n_bank]
        bank_time = self.gw_time_bank[:n_bank]
        bank_gw_idx = self.gw_idx_bank[:n_bank]
        k = min(int(self.topk), n_bank)
        if k <= 0:
            return sel, level_hits, fallback_count, attempted

        feat_g_active = feat_g[active_rows]
        gw_indices_active = gw_indices[active_rows]
        anchor_times = batch_event_time_mjd[active_rows]
        finite_anchor = torch.isfinite(anchor_times)
        attempted = int(finite_anchor.sum().item())
        if attempted <= 0:
            return sel, level_hits, fallback_count, attempted

        sim_mem = torch.matmul(
            feat_g_active.to(torch.float16), bank_feat.T
        ).to(torch.float32) / temperature
        same_mem = gw_indices_active.unsqueeze(1) == bank_gw_idx.unsqueeze(0)
        sim_mem = sim_mem.masked_fill(same_mem, -1e9)
        topk_vals, topk_idx = torch.topk(sim_mem, k=k, dim=1)  # [A, K]
        valid_topk = bank_gw_idx[topk_idx] != gw_indices_active.unsqueeze(1)
        topk_time = bank_time[topk_idx]
        dt_topk = torch.abs(topk_time - anchor_times.unsqueeze(1))

        unresolved = finite_anchor.clone()
        for lvl, wnd in enumerate(window_days):
            cand_mask = valid_topk & (dt_topk <= float(wnd))
            cand_count = cand_mask.sum(dim=1)
            is_last = (lvl == len(window_days) - 1)
            accepted = unresolved & (
                (cand_count >= int(min_candidates)) | (is_last & (cand_count > 0))
            )
            if not accepted.any():
                continue
            rows = torch.nonzero(accepted, as_tuple=False).squeeze(-1)
            masked_vals = topk_vals[rows].masked_fill(~cand_mask[rows], -1e9)
            best_pos = masked_vals.argmax(dim=1)
            sel[rows] = topk_idx[rows, best_pos]
            unresolved[rows] = False
            level_hits[lvl] += int(rows.numel())

        fallback_count = int(unresolved.sum().item())

        return sel, level_hits, fallback_count, attempted

    def fetch(self, bank_indices, device, h_l_dtype, z_l_dtype, opt_dtype):
        if bank_indices.numel() == 0:
            return None
        idx = bank_indices.detach().to(device=self.device, dtype=torch.long)
        h_sel = self.h_l_bank[idx].to(device=device, dtype=h_l_dtype)
        z_sel = self.z_l_bank[idx].to(device=device, dtype=z_l_dtype)
        c_sel = self.coords_bank[idx].to(device=device, dtype=torch.float32)
        t_sel = self.zero_time_bank[idx].to(device=device, dtype=torch.float32)
        gw_idx_sel = self.gw_idx_bank[idx].to(device=device, dtype=torch.long)
        opt_t_sel = self.opt_t_bank[idx].to(device=device, dtype=opt_dtype)
        opt_v_sel = self.opt_v_bank[idx].to(device=device, dtype=opt_dtype)
        opt_mask_sel = self.opt_mask_bank[idx].to(device=device, dtype=opt_dtype)
        opt_err_sel = self.opt_err_bank[idx].to(device=device, dtype=opt_dtype)
        return (
            h_sel,
            z_sel,
            c_sel,
            t_sel,
            gw_idx_sel,
            opt_t_sel,
            opt_v_sel,
            opt_mask_sel,
            opt_err_sel,
        )

def load_gw_event_time_mjd_table(h5_path, device):
    """
    Load GW event times from HDF5 as a device tensor.

    Returns:
        Tensor [n_gw] on `device`, or None if field is missing/invalid.
    """
    ds_path = "events/gw_data/event_time_mjd"
    with h5py.File(h5_path, "r") as f:
        n_gw = int(f["events/gw_data/scalars"].shape[0])
        if ds_path not in f:
            print(
                f"WARNING: '{ds_path}' not found in {h5_path}. "
                "Time compatibility is disabled."
            )
            return None
        event_time_mjd = np.asarray(f[ds_path][:], dtype=np.float32)

    if event_time_mjd.shape[0] != n_gw:
        print(
            "WARNING: event_time_mjd length mismatch "
            f"(got {event_time_mjd.shape[0]}, expected {n_gw}). "
            "Time compatibility is disabled."
        )
        return None

    n_invalid = int((~np.isfinite(event_time_mjd)).sum())
    print(
        "Loaded GW event times: "
        f"{event_time_mjd.shape[0]} entries, invalid={n_invalid}"
    )
    return torch.from_numpy(event_time_mjd).to(device=device)

def sample_easy_negatives(batch_size, device):
    if batch_size < 2:
        return None
    shift = int(torch.randint(1, batch_size, (1,), device=device).item())
    return (torch.arange(batch_size, device=device) + shift) % batch_size

def augment_gw_data(gw_s, gw_m, training=True,
                    noise_std=0.05, scalar_jitter=0.02, channel_dropout_prob=0.1):
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
    opt_xyz = torch.stack([
        torch.cos(dec) * torch.cos(ra),
        torch.cos(dec) * torch.sin(ra),
        torch.sin(dec)
    ], dim=-1)  # [B, 3]

    pix_xyz = gw_m[:, :3, :]  # [B, 3, 19200]
    dot = torch.bmm(opt_xyz.unsqueeze(1), pix_xyz).squeeze(1)  # [B, 19200]
    nearest_idx = dot.argmax(dim=-1)  # [B]

    dA = gw_m[:, 3, :]  # [B, 19200]
    dP = gw_m[:, 4, :]  # [B, 19200]
    dA = torch.nan_to_num(dA.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    dP = torch.nan_to_num(dP.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    density = dP / dA.clamp_min(torch.finfo(dP.dtype).eps)
    density_at_opt = density[torch.arange(density.size(0), device=density.device), nearest_idx]  # [B]

    # Credible level: posterior mass with density >= density at the optical position.
    total_probability = dP.sum(dim=-1).clamp_min(torch.finfo(dP.dtype).eps)
    cred_level = torch.where(
        density >= density_at_opt.unsqueeze(-1),
        dP,
        torch.zeros_like(dP),
    ).sum(dim=-1) / total_probability  # [B]
    return cred_level.unsqueeze(-1)  # [B, 1]


def _unwrap_compiled_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _model_requires_cred_level(model) -> bool:
    base = _unwrap_compiled_model(model)
    if hasattr(base, "uses_cred_level_input"):
        return bool(base.uses_cred_level_input())
    return bool(getattr(base, "dual_fusion", False))


def augment_optical_data(opt_t, opt_v, opt_mask, opt_err, training=True,
                         time_jitter=0.0, flux_noise=0.0,
                         obs_dropout=0.0, band_dropout=0.0):
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
        band_mask = torch.rand(opt_v.size(0), opt_v.size(2), device=opt_v.device) < band_dropout
        if band_mask.any():
            band_mask = band_mask[:, None, :]
            opt_mask = opt_mask.masked_fill(band_mask, 0)
            opt_v = opt_v.masked_fill(band_mask, 0.0)
            if opt_err is not None:
                opt_err = opt_err.masked_fill(band_mask, 0.0)

    return opt_t, opt_v, opt_mask, opt_err

def build_lr_scheduler(optimizer, args, steps_per_epoch, start_step):
    if args.lr_scheduler == "none":
        return None
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)
    min_lr = args.min_lr if args.min_lr is not None else 0.0
    min_lr_ratio = min(min_lr / args.lr, 1.0)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_step - 1)

def compute_cls_weight(args, epoch):
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
    start = int(getattr(args, "retrieval_start_epoch", 0))
    return int(epoch) >= start


def is_cls_active(epoch, args):
    """Return True when classification loss is active (after ramp)."""
    return compute_cls_weight(args, epoch) > 0.0


def compute_itc_weight(args, epoch):
    if args.itc_decay_epochs <= 0 or args.itc_decay_ratio <= 0:
        return args.itc_weight
    if epoch < args.itc_decay_start_epoch:
        return args.itc_weight
    progress = (epoch - args.itc_decay_start_epoch + 1) / float(args.itc_decay_epochs)
    progress = min(1.0, progress)
    return args.itc_weight * (1.0 - args.itc_decay_ratio * progress)

def compute_hard_neg_ratio(args, epoch):
    if epoch < args.hard_neg_start_epoch:
        return 0.0
    if args.hard_neg_ramp_epochs <= 0:
        return 1.0
    progress = (epoch - args.hard_neg_start_epoch + 1) / float(args.hard_neg_ramp_epochs)
    return min(1.0, progress)


def compute_curriculum_full_epoch(start_epoch, ramp_epochs):
    start_epoch = max(0, int(start_epoch))
    ramp_epochs = int(ramp_epochs)
    if ramp_epochs <= 0:
        return start_epoch
    return start_epoch + ramp_epochs - 1


def hard_neg_full_activation_reachable(args):
    total_epochs = max(1, int(getattr(args, "epochs", 1)))
    hard_neg_full_epoch = compute_curriculum_full_epoch(
        getattr(args, "hard_neg_start_epoch", 0),
        getattr(args, "hard_neg_ramp_epochs", 0),
    )
    return hard_neg_full_epoch < total_epochs


def is_best_ckpt_selection_eligible(args, epoch, hard_neg_ratio=None):
    """Gate best-checkpoint selection until the requested metric is meaningful."""
    metric_name = str(getattr(args, "best_ckpt_metric", "fusion_gallery_mrr"))

    if metric_name in CLS_BEST_CKPT_METRICS:
        if not should_compute_cls_metrics(args):
            return False
        if is_cls_branch_enabled(args) and not is_cls_active(epoch, args):
            return False
        if hard_neg_full_activation_reachable(args):
            ratio = compute_hard_neg_ratio(args, epoch) if hard_neg_ratio is None else float(hard_neg_ratio)
            return ratio >= (1.0 - 1e-8)
        return True

    if metric_name in FUSION_GALLERY_BEST_CKPT_METRICS:
        if is_gallery_enabled(args):
            metric_ready = is_retrieval_active(args, epoch)
        else:
            metric_ready = should_compute_fusion_gallery_metrics(args, epoch)
        if not metric_ready:
            return False
        if is_cls_branch_enabled(args) and hard_neg_full_activation_reachable(args):
            ratio = compute_hard_neg_ratio(args, epoch) if hard_neg_ratio is None else float(hard_neg_ratio)
            return ratio >= (1.0 - 1e-8)
        if gallery_hard_neg_full_activation_reachable(args):
            hard_weight = compute_gallery_hard_neg_weight(args, epoch)
            full_weight = float(getattr(args, "gallery_hard_neg_weight", 0.5))
            if full_weight > 0.0 and hard_weight < (full_weight - 1e-8):
                return False
        return True

    if metric_name in G2O_BEST_CKPT_METRICS:
        return True

    return True


def describe_best_ckpt_selection_pending(args, epoch, hard_neg_ratio=None):
    metric_name = str(getattr(args, "best_ckpt_metric", "fusion_gallery_mrr"))
    ratio = compute_hard_neg_ratio(args, epoch) if hard_neg_ratio is None else float(hard_neg_ratio)

    if metric_name in CLS_BEST_CKPT_METRICS:
        if not should_compute_cls_metrics(args):
            return "waiting for classification metrics to be enabled."
        if is_cls_branch_enabled(args) and not is_cls_active(epoch, args):
            return "waiting for CLS ramp/start before tracking classification metric."
        if hard_neg_full_activation_reachable(args) and ratio < (1.0 - 1e-8):
            return f"waiting for full hard-negative ratio (current={ratio:.3f}, required=1.000)."
        return "waiting for classification metric to become available."

    if metric_name in FUSION_GALLERY_BEST_CKPT_METRICS:
        if is_gallery_enabled(args) and not is_retrieval_active(args, epoch):
            start = int(getattr(args, "retrieval_start_epoch", 0))
            return f"waiting for fusion-gallery training start (epoch {start + 1})."
        if (not is_gallery_enabled(args)) and not should_compute_fusion_gallery_metrics(args, epoch):
            return "waiting for fusion-gallery metrics to be enabled."
        if is_cls_branch_enabled(args) and hard_neg_full_activation_reachable(args) and ratio < (1.0 - 1e-8):
            return f"waiting for full hard-negative ratio (current={ratio:.3f}, required=1.000)."
        if gallery_hard_neg_full_activation_reachable(args):
            hard_weight = compute_gallery_hard_neg_weight(args, epoch)
            full_weight = float(getattr(args, "gallery_hard_neg_weight", 0.5))
            if full_weight > 0.0 and hard_weight < (full_weight - 1e-8):
                return f"waiting for retrieval hard-mining ramp (current={hard_weight:.3f}, required={full_weight:.3f})."
        return "waiting for fusion-gallery metric to become available."

    return "waiting for requested best-checkpoint metric to become available."


def compute_weighted_cls_loss(pos_loss, hard_loss, neg_loss, has_negatives, args):
    pos_weight = args.cls_pos_weight
    neg_weight = args.cls_neg_weight
    extra_neg_weight = args.cls_extra_neg_weight

    if has_negatives:
        denom = max(1e-8, pos_weight + neg_weight + extra_neg_weight)
        return (
            pos_weight * pos_loss + neg_weight * hard_loss + extra_neg_weight * neg_loss
        ) / denom

    denom = max(1e-8, pos_weight + neg_weight)
    return (pos_weight * pos_loss + neg_weight * hard_loss) / denom


def compute_fusion_gallery_nce_loss(scores, positive_mask):
    """Multi-positive retrieval NCE over fusion-head candidate scores."""
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must have matching shape [n_query, n_candidate].")
    positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
    valid_rows = positive_mask.any(dim=1)
    if not valid_rows.any():
        return scores.sum() * 0.0

    scores_valid = scores[valid_rows]
    pos_valid = positive_mask[valid_rows]
    neg_large = torch.finfo(scores_valid.dtype).min
    pos_scores = scores_valid.masked_fill(~pos_valid, neg_large)
    return (torch.logsumexp(scores_valid, dim=1) - torch.logsumexp(pos_scores, dim=1)).mean()


def is_gallery_hard_neg_enabled(args):
    """Return True when retrieval hard-negative mining is enabled."""
    return bool(getattr(args, "gallery_hard_neg_enable", False))


def compute_gallery_hard_neg_weight(args, epoch):
    """Compute ramp-ed weight for the retrieval hard-negative auxiliary loss.

    Returns 0.0 when the hard-mining start epoch (``retrieval_start_epoch +
    gallery_hard_neg_start_after_retrieval_epochs``) has not been reached, then
    ramps linearly to ``gallery_hard_neg_weight`` over
    ``gallery_hard_neg_ramp_epochs``.
    """
    if not is_gallery_hard_neg_enabled(args):
        return 0.0
    if float(getattr(args, "gallery_loss_weight", 0.0)) <= 0.0:
        return 0.0
    retrieval_start = int(getattr(args, "retrieval_start_epoch", 0))
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
    full_epoch = (
        int(getattr(args, "retrieval_start_epoch", 0))
        + int(getattr(args, "gallery_hard_neg_start_after_retrieval_epochs", 2))
        + max(0, int(getattr(args, "gallery_hard_neg_ramp_epochs", 2)))
        - 1
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


def compute_fusion_gallery_metrics(scores, positive_mask, ks=(1, 5)):
    """Compute retrieval metrics for fusion-head mini-gallery scores."""
    if scores.ndim != 2 or positive_mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must have matching shape [n_query, n_candidate].")
    positive_mask = positive_mask.to(device=scores.device, dtype=torch.bool)
    valid_rows = positive_mask.any(dim=1)
    out = {f"fusion_gallery_recall_at_{int(k)}": 0.0 for k in ks}
    out["fusion_gallery_mrr"] = 0.0
    if not valid_rows.any():
        return out

    scores_valid = scores[valid_rows]
    pos_valid = positive_mask[valid_rows]
    order = scores_valid.argsort(dim=1, descending=True)
    sorted_pos = pos_valid.gather(1, order)
    n_candidate = int(scores.size(1))
    for k in ks:
        actual_k = min(int(k), n_candidate)
        out[f"fusion_gallery_recall_at_{int(k)}"] = float(sorted_pos[:, :actual_k].any(dim=1).float().mean().item())

    ranks = torch.arange(1, n_candidate + 1, device=scores.device, dtype=torch.float32).unsqueeze(0)
    first_pos_rank = torch.where(
        sorted_pos,
        ranks.expand_as(sorted_pos),
        torch.full_like(ranks.expand_as(sorted_pos), float("inf")),
    ).min(dim=1).values
    out["fusion_gallery_mrr"] = float((1.0 / first_pos_rank).mean().item())
    return out


def build_gallery_time_delta_matrix(
    query_event_time_mjd,
    candidate_event_time_mjd,
    *,
    positive_mask=None,
    distractor_time_mode="actual",
    distractor_time_window_days=30.0,
):
    """Build per-query/per-candidate dt for fusion mini-gallery scoring.

    In ``actual`` mode this is the raw candidate-minus-query MJD difference.
    In ``synthetic_after_gw`` mode, matching positives keep their real dt while
    every non-matching distractor is assigned a synthetic first-detection offset
    sampled uniformly from [0, distractor_time_window_days].
    """
    if query_event_time_mjd is None or candidate_event_time_mjd is None:
        return None

    query = query_event_time_mjd.to(torch.float32).reshape(-1, 1)
    candidate = candidate_event_time_mjd.to(device=query.device, dtype=torch.float32).reshape(1, -1)
    dt = candidate - query
    dt = torch.where(torch.isfinite(dt), dt, torch.zeros_like(dt))

    mode = str(distractor_time_mode).strip().lower()
    if mode in {"actual", "real", "none"}:
        return dt
    if mode not in {"synthetic_after_gw", "synthetic", "after_gw"}:
        raise ValueError(
            f"Unsupported gallery_distractor_time_mode='{distractor_time_mode}'. "
            "Expected 'actual' or 'synthetic_after_gw'."
        )

    window_days = float(distractor_time_window_days)
    if window_days <= 0:
        raise ValueError("gallery_distractor_time_window_days must be > 0 for synthetic_after_gw mode.")

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
    n_candidate = int(h_candidates.size(0))
    if n_candidate == 0:
        return torch.empty((n_query, 0), dtype=g.dtype, device=g.device)

    max_pairs = max(1, int(chunk_size))
    # Balance query and candidate chunk sizes proportionally
    q_chunk = min(n_query, max(1, int(math.sqrt(max_pairs * max(1, n_query / max(1, n_candidate))))))
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
            scores[q_start:q_end, c_start:c_end] = (logits[:, 1] - logits[:, 0]).reshape(n_q, n_c)

    return scores


def build_pairwise_time_delta_matrix(query_event_time_mjd, candidate_event_time_mjd):
    if query_event_time_mjd is None or candidate_event_time_mjd is None:
        return None
    query = query_event_time_mjd.to(torch.float32).reshape(-1, 1)
    candidate = candidate_event_time_mjd.to(device=query.device, dtype=torch.float32).reshape(1, -1)
    dt = candidate - query
    return torch.where(torch.isfinite(dt), dt, torch.zeros_like(dt))

def apply_temperature_schedule(model, args, epoch):
    if args.temp_schedule == "learned":
        return None
    if args.temp_schedule == "fixed":
        target_temp = args.temp_init
    else:
        progress = epoch / float(max(1, args.epochs - 1))
        target_temp = args.temp_final + 0.5 * (args.temp_init - args.temp_final) * (1.0 + math.cos(math.pi * progress))
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

    raise ValueError(f"Unsupported best_ckpt_metric: {metric_name}")

def evaluate(
    model,
    val_loader,
    device,
    args,
    epoch,
    amp_dtype=torch.float32,
    gw_event_time_mjd_table=None,
):
    from metrics import (compute_retrieval_metrics,
                         compute_classification_metrics,
                         compute_embedding_metrics)

    model.eval()
    has_negatives = args.neg_data_path is not None
    cls_branch_enabled = is_cls_branch_enabled(args)
    ref_time_cache = None
    cls_weight = compute_cls_weight(args, epoch)
    cls_active = cls_weight > 0.0
    itc_weight = compute_itc_weight(args, epoch)
    hard_neg_ratio = compute_hard_neg_ratio(args, epoch)
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
    val_hard_acc = 0.0
    val_neg_acc = 0.0
    val_total_acc = 0.0
    val_batches = 0

    # Accumulators for new metrics (computed globally after all batches)
    all_cls_probs = []    # predicted P(match) for classification metrics
    all_cls_labels = []   # ground-truth labels for classification metrics
    all_cls_sources = []  # source tag per sample ('pos', 'hard', 'neg')
    all_feat_g = []       # L2-normalized GW embeddings for embedding metrics
    all_feat_o = []       # L2-normalized optical embeddings
    all_gw_indices = []   # GW event indices
    retrieval_metrics_accum = []  # per-batch retrieval metric dicts
    fusion_gallery_metrics_accum = []
    dt_stats = {
        "pos": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "hard": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
        "extra": {"sum": 0.0, "sum_sq": 0.0, "count": 0},
    }
    hard_window_days = getattr(args, "_hardneg_window_days", parse_day_windows(args.hardneg_time_window_days))
    hard_window_hits = [0 for _ in hard_window_days]
    hard_fallback_total = 0
    hard_total = 0

    with torch.no_grad():
        for batch_data in val_loader:
            _opt_zero_time_mjd_base = None
            _neg_zero_time_mjd_base = None
            _neg_zero_time_mjd_cls_base = None
            _opt_first_detection_mjd = None
            if has_negatives:
                if len(batch_data) >= 17:
                    (
                        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                        neg_t, neg_v, neg_mask, neg_err, neg_coords,
                        _opt_zero_time_mjd_base, _neg_zero_time_mjd_base, _neg_zero_time_mjd_cls_base,
                        _opt_first_detection_mjd,
                    ) = batch_data[:17]
                elif len(batch_data) >= 16:
                    (
                        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                        neg_t, neg_v, neg_mask, neg_err, neg_coords,
                        _opt_zero_time_mjd_base, _neg_zero_time_mjd_base, _neg_zero_time_mjd_cls_base,
                    ) = batch_data[:16]
                elif len(batch_data) >= 15:
                    (
                        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                        neg_t, neg_v, neg_mask, neg_err, neg_coords,
                        _opt_zero_time_mjd_base, _neg_zero_time_mjd_base,
                    ) = batch_data[:15]
                else:
                    (
                        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                        neg_t, neg_v, neg_mask, neg_err, neg_coords,
                    ) = batch_data[:13]
            else:
                if len(batch_data) >= 10:
                    gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices, _opt_zero_time_mjd_base, _opt_first_detection_mjd = batch_data[:10]
                elif len(batch_data) >= 9:
                    gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices, _opt_zero_time_mjd_base = batch_data[:9]
                else:
                    gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data

            gw_s = gw_s.to(device, non_blocking=True)
            gw_m = gw_m.to(device, non_blocking=True)
            opt_t = opt_t.to(device, non_blocking=True)
            opt_v = opt_v.to(device, non_blocking=True)
            opt_mask = opt_mask.to(device, non_blocking=True)
            opt_err = opt_err.to(device, non_blocking=True)
            opt_coords = opt_coords.to(device, non_blocking=True)
            gw_indices = gw_indices.to(device, non_blocking=True).long()
            batch_event_time_mjd = None
            if gw_event_time_mjd_table is not None:
                batch_event_time_mjd = gw_event_time_mjd_table[gw_indices]

            if has_negatives:
                neg_t = neg_t.to(device, non_blocking=True)
                neg_v = neg_v.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)
                if _neg_zero_time_mjd_base is not None:
                    _neg_zero_time_mjd_base = _neg_zero_time_mjd_base.to(device, non_blocking=True)
                if _neg_zero_time_mjd_cls_base is not None:
                    _neg_zero_time_mjd_cls_base = _neg_zero_time_mjd_cls_base.to(device, non_blocking=True)
            if _opt_first_detection_mjd is not None:
                _opt_first_detection_mjd = _opt_first_detection_mjd.to(device, non_blocking=True)
            elif _opt_zero_time_mjd_base is not None:
                _opt_first_detection_mjd = _opt_zero_time_mjd_base.to(device, non_blocking=True)

            batch_size = gw_s.size(0)
            if (
                ref_time_cache is None
                or ref_time_cache.shape[0] != batch_size
                or ref_time_cache.dtype != opt_t.dtype
            ):
                ref_time_cache = build_ref_time(
                    batch_size, args.n_ref, args.ref_start, args.ref_end, device, opt_t.dtype
                )
            opt_ref_t = ref_time_cache
            need_cred_level = _model_requires_cred_level(model)

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
                g, z_l, h_l, H_gw = model.encode(
                    gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
                )
                extra_neg_z_itc = None
                extra_neg_event_time_mjd = None
                if has_negatives and not cls_branch_enabled:
                    neg_zero_time_mjd_for_itc = _neg_zero_time_mjd_cls_base
                    if neg_zero_time_mjd_for_itc is None:
                        neg_zero_time_mjd_for_itc = _neg_zero_time_mjd_base
                    with torch.no_grad():
                        extra_neg_z_itc, _ = model.encode_optical(
                            neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err,
                        )
                    extra_neg_event_time_mjd = neg_zero_time_mjd_for_itc
                if args.itc_loss_type == "supcon":
                    itc_loss, sim_g2o, sim_g2o_for_loss = model.compute_supcon_loss(
                        g, z_l, gw_indices, margin=args.supcon_margin,
                        gw_event_time_mjd=batch_event_time_mjd,
                        opt_event_time_mjd=resolve_optical_candidate_time_mjd(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        ),
                        extra_neg_z=extra_neg_z_itc,
                        extra_neg_opt_event_time_mjd=extra_neg_event_time_mjd,
                    )
                else:
                    itc_loss, sim_g2o, sim_g2o_for_loss = model.compute_itc_loss(
                        g, z_l, gw_indices, mask=args.mask_itc,
                        gw_event_time_mjd=batch_event_time_mjd,
                        opt_event_time_mjd=resolve_optical_candidate_time_mjd(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        ),
                        extra_neg_z=extra_neg_z_itc,
                        extra_neg_opt_event_time_mjd=extra_neg_event_time_mjd,
                    )

                # Extract L2-normalized projected features for embedding metrics
                feat_g, feat_o = model.get_contrastive_embeddings(g, z_l)
                labels_pos = torch.ones(batch_size, device=device, dtype=torch.long)
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
                    cred_level = compute_credible_level(gw_m, opt_coords) if need_cred_level else None
                    if _opt_first_detection_mjd is not None and batch_event_time_mjd is not None:
                        dt_pos = compute_time_delta_days(_opt_first_detection_mjd, batch_event_time_mjd)
                    else:
                        dt_pos = torch.zeros((batch_size,), device=device, dtype=torch.float32)
                    logits_pos = model.fusion_logits(
                        g, h_l, z_l=z_l, H_gw=H_gw, cred_level=cred_level,
                        gw_s=gw_s, gw_m=gw_m, opt_coords=opt_coords,
                        dt_days=dt_pos,
                    )
                    pos_loss = model.cls_criterion(logits_pos, labels_pos)

                    easy_idx = sample_easy_negatives(batch_size, device)
                    choose_hard_mask = None
                    if easy_idx is None:
                        neg_idx = None
                    elif hard_neg_ratio <= 0:
                        neg_idx = easy_idx
                        choose_hard_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
                    else:
                        if batch_event_time_mjd is not None:
                            hard_idx, lvl_hits, fallback_cnt = sample_inbatch_hard_negatives_with_time(
                                sim_g2o=sim_g2o,
                                gw_indices=gw_indices,
                                batch_event_time_mjd=batch_event_time_mjd,
                                candidate_time_mjd=resolve_optical_candidate_time_mjd(
                                    _opt_first_detection_mjd, batch_event_time_mjd
                                ),
                                window_days=hard_window_days,
                                min_candidates=args.hardneg_min_candidates,
                                semi_hard=args.semi_hard,
                                semi_hard_margin=args.semi_hard_margin,
                                fallback_mode=args.hardneg_fallback_mode,
                            )
                            hard_total += batch_size
                            hard_fallback_total += int(fallback_cnt)
                            for li, hv in enumerate(lvl_hits):
                                hard_window_hits[li] += int(hv)
                        elif args.semi_hard:
                            hard_idx = model.sample_semi_hard_negatives(sim_g2o, gw_indices, margin=args.semi_hard_margin)
                        else:
                            hard_idx = model.sample_hard_negatives(sim_g2o, gw_indices)
                        if hard_neg_ratio >= 1:
                            neg_idx = hard_idx
                            choose_hard_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
                        else:
                            choose_hard = torch.rand(batch_size, device=device) < hard_neg_ratio
                            neg_idx = torch.where(choose_hard, hard_idx, easy_idx)
                            choose_hard_mask = choose_hard

                    if neg_idx is None:
                        hard_loss = torch.zeros((), device=device)
                        logits_hard = torch.zeros((batch_size, 2), device=device, dtype=g.dtype)
                    else:
                        h_l_hard = h_l[neg_idx].clone()
                        z_l_hard = z_l[neg_idx].clone()
                        coords_hard = opt_coords[neg_idx].clone()
                        if batch_event_time_mjd is not None:
                            dt_hard = compute_time_delta_days(
                                gather_candidate_optical_time_mjd(
                                    _opt_first_detection_mjd, batch_event_time_mjd, neg_idx
                                ),
                                batch_event_time_mjd,
                            )
                        else:
                            dt_hard = torch.zeros((batch_size,), device=device, dtype=torch.float32)
                        if choose_hard_mask is None:
                            choose_hard_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
                        # NOTE: encode_hard_negative_with_time_shift removed.
                        # Features already correctly encoded at h_l_hard/z_l_hard above.
                        cred_level_hard = compute_credible_level(gw_m, coords_hard) if need_cred_level else None
                        if batch_event_time_mjd is not None and choose_hard_mask is not None and choose_hard_mask.any():
                            _accumulate_time_delta_stats(dt_stats, "hard", dt_hard[choose_hard_mask])
                        logits_hard = model.fusion_logits(
                            g, h_l_hard, z_l=z_l_hard, H_gw=H_gw, cred_level=cred_level_hard,
                            gw_s=gw_s, gw_m=gw_m, opt_coords=coords_hard,
                            dt_days=dt_hard,
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
                        cred_level_neg = compute_credible_level(gw_m, neg_coords) if need_cred_level else None
                        neg_event_time = (
                            neg_zero_time_mjd_for_cls.to(device=device, dtype=torch.float32)
                            if neg_zero_time_mjd_for_cls is not None
                            else None
                        )
                        dt_neg = (
                            compute_time_delta_days(neg_event_time, batch_event_time_mjd)
                            if (neg_event_time is not None and batch_event_time_mjd is not None)
                            else torch.zeros((batch_size,), device=device, dtype=torch.float32)
                        )
                        logits_neg = model.fusion_logits(
                            g, h_l_neg, z_l=z_l_neg, H_gw=H_gw, cred_level=cred_level_neg,
                            gw_s=gw_s, gw_m=gw_m, opt_coords=neg_coords,
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
                        pos_loss, hard_loss, neg_loss, has_negatives, args
                    )
                    gallery_loss = torch.zeros((), device=device)
                    gallery_hard_loss = torch.zeros((), device=device)
                    gallery_hard_weight = 0.0
                    total_loss = (
                        itc_weight * itc_loss
                        + cls_weight * cls_loss
                    )
                else:
                    logits_pos = torch.zeros((batch_size, 2), device=device, dtype=g.dtype)
                    logits_hard = torch.zeros((batch_size, 2), device=device, dtype=g.dtype)
                    logits_neg = torch.zeros((batch_size, 2), device=device, dtype=g.dtype) if has_negatives else None
                    pos_loss = torch.zeros((), device=device)
                    hard_loss = torch.zeros((), device=device)
                    neg_loss = torch.zeros((), device=device) if has_negatives else None
                    cls_loss = torch.zeros((), device=device)
                    gallery_loss = torch.zeros((), device=device)
                    gallery_hard_loss = torch.zeros((), device=device)
                    gallery_hard_weight = 0.0
                    need_extra_optical_features = (
                        has_negatives
                        and (
                            gallery_loss_active
                            or fusion_gallery_metrics_enabled
                            or cls_metrics_enabled
                        )
                    )
                    if need_extra_optical_features:
                        z_l_neg, h_l_neg = model.encode_optical(
                            neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err
                        )
                        gallery_extra_h_l = h_l_neg
                        gallery_extra_z_l = z_l_neg
                        gallery_extra_coords = neg_coords
                        neg_event_time = (
                            _neg_zero_time_mjd_cls_base.to(device=device, dtype=torch.float32)
                            if _neg_zero_time_mjd_cls_base is not None
                            else (
                                _neg_zero_time_mjd_base.to(device=device, dtype=torch.float32)
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
                if compute_gallery_this_epoch and gallery_extra_h_l is not None:
                    h_candidates = [h_l]
                    z_candidates = [z_l]
                    coords_candidates = [opt_coords]
                    candidate_gw_indices = [gw_indices]
                    candidate_time_blocks = []
                    if batch_event_time_mjd is not None:
                        pos_gallery_time = resolve_optical_candidate_time_mjd(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        ).to(device=device, dtype=torch.float32)
                        candidate_time_blocks.append(pos_gallery_time)
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
                            torch.full((batch_size,), -1, device=device, dtype=gw_indices.dtype)
                        )
                        if batch_event_time_mjd is not None:
                            if gallery_extra_event_time is None:
                                candidate_time_blocks.append(
                                    torch.full((batch_size,), float("nan"), device=device, dtype=torch.float32)
                                )
                            else:
                                candidate_time_blocks.append(
                                    gallery_extra_event_time.to(device=device, dtype=torch.float32)
                                )
                    h_gallery = torch.cat(h_candidates, dim=0)
                    z_gallery = torch.cat(z_candidates, dim=0)
                    coords_gallery = torch.cat(coords_candidates, dim=0)
                    gallery_gw_indices = torch.cat(candidate_gw_indices, dim=0)
                    gallery_positive_mask = gw_indices.unsqueeze(1) == gallery_gw_indices.unsqueeze(0)
                    gallery_candidate_time = torch.cat(candidate_time_blocks, dim=0) if candidate_time_blocks else None
                    dt_gallery = build_gallery_time_delta_matrix(
                        batch_event_time_mjd,
                        gallery_candidate_time,
                        positive_mask=gallery_positive_mask,
                        distractor_time_mode=args.gallery_distractor_time_mode,
                        distractor_time_window_days=args.gallery_distractor_time_window_days,
                    )
                    gallery_scores = compute_fusion_gallery_score_matrix(
                        model,
                        g=g,
                        H_gw=H_gw,
                        gw_s=gw_s,
                        gw_m=gw_m,
                        h_candidates=h_gallery,
                        z_candidates=z_gallery,
                        opt_coords_candidates=coords_gallery,
                        dt_days_matrix=dt_gallery,
                        need_cred_level=need_cred_level,
                        chunk_size=args.gallery_score_chunk_size,
                    )
                    gallery_loss = compute_fusion_gallery_nce_loss(gallery_scores, gallery_positive_mask)
                    if fusion_gallery_metrics_enabled:
                        fusion_gallery_metrics_accum.append(
                            compute_fusion_gallery_metrics(gallery_scores.detach(), gallery_positive_mask, ks=(1, 5))
                        )
                    if gallery_loss_active:
                        total_loss = total_loss + float(args.gallery_loss_weight) * gallery_loss
                    gallery_hard_weight = compute_gallery_hard_neg_weight(args, epoch)
                    if gallery_hard_weight > 0.0:
                        gallery_hard_loss = compute_fusion_gallery_hard_nce_loss(
                            gallery_scores, gallery_positive_mask,
                            topk=int(args.gallery_hard_neg_topk),
                        )
                        total_loss = total_loss + (
                            float(args.gallery_loss_weight)
                            * gallery_hard_weight
                            * gallery_hard_loss
                        )

                if cls_metrics_enabled and not cls_branch_enabled:
                    cred_level = compute_credible_level(gw_m, opt_coords) if need_cred_level else None
                    if _opt_first_detection_mjd is not None and batch_event_time_mjd is not None:
                        dt_pos = compute_time_delta_days(_opt_first_detection_mjd, batch_event_time_mjd)
                    else:
                        dt_pos = torch.zeros((batch_size,), device=device, dtype=torch.float32)
                    logits_pos = model.fusion_logits(
                        g, h_l, z_l=z_l, H_gw=H_gw, cred_level=cred_level,
                        gw_s=gw_s, gw_m=gw_m, opt_coords=opt_coords,
                        dt_days=dt_pos,
                    )
                    cls_logits_available = True

                    metric_neg_idx = sample_easy_negatives(batch_size, device)
                    if metric_neg_idx is not None:
                        h_l_metric_hard = h_l[metric_neg_idx].clone()
                        z_l_metric_hard = z_l[metric_neg_idx].clone()
                        coords_metric_hard = opt_coords[metric_neg_idx].clone()
                        if batch_event_time_mjd is not None:
                            dt_metric_hard = compute_time_delta_days(
                                gather_candidate_optical_time_mjd(
                                    _opt_first_detection_mjd, batch_event_time_mjd, metric_neg_idx
                                ),
                                batch_event_time_mjd,
                            )
                        else:
                            dt_metric_hard = torch.zeros((batch_size,), device=device, dtype=torch.float32)
                        cred_level_hard = compute_credible_level(gw_m, coords_metric_hard) if need_cred_level else None
                        logits_hard = model.fusion_logits(
                            g, h_l_metric_hard, z_l=z_l_metric_hard, H_gw=H_gw, cred_level=cred_level_hard,
                            gw_s=gw_s, gw_m=gw_m, opt_coords=coords_metric_hard,
                            dt_days=dt_metric_hard,
                        )
                        cls_hard_logits_available = True

                    if has_negatives and gallery_extra_h_l is not None and gallery_extra_z_l is not None:
                        cred_level_neg = compute_credible_level(gw_m, neg_coords) if need_cred_level else None
                        dt_neg_metric = (
                            compute_time_delta_days(gallery_extra_event_time, batch_event_time_mjd)
                            if (gallery_extra_event_time is not None and batch_event_time_mjd is not None)
                            else torch.zeros((batch_size,), device=device, dtype=torch.float32)
                        )
                        logits_neg = model.fusion_logits(
                            g, gallery_extra_h_l, z_l=gallery_extra_z_l, H_gw=H_gw, cred_level=cred_level_neg,
                            gw_s=gw_s, gw_m=gw_m, opt_coords=gallery_extra_coords,
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
                    (ext_count,), -1,
                    device=gw_indices.device, dtype=gw_indices.dtype,
                )
                batch_ret_gw = torch.cat([gw_indices, ext_gw_indices], dim=0)
                batch_ret = compute_retrieval_metrics(
                    sim_ext, batch_ret_gw, ks=(1, 5, 10)
                )
            else:
                batch_ret = compute_retrieval_metrics(
                    sim_ext, gw_indices, ks=(1, 5, 10)
                )
            retrieval_metrics_accum.append(batch_ret)

            # --- ITC accuracy (in-batch only) ---
            itc_preds = sim_g2o_d.argmax(dim=1)
            anchor_gw = gw_indices
            pred_gw = gw_indices[itc_preds]
            itc_acc = (anchor_gw == pred_gw).float().mean().item()

            if cls_logits_available:
                pos_acc = (logits_pos.argmax(dim=1) == labels_pos).float().mean().item()
                acc_parts = [pos_acc]
                hard_acc = 0.0
                if cls_hard_logits_available:
                    hard_acc = (logits_hard.argmax(dim=1) == labels_neg).float().mean().item()
                    acc_parts.append(hard_acc)
                neg_acc = 0.0
                if has_negatives and cls_extra_neg_logits_available:
                    neg_acc = (logits_neg.argmax(dim=1) == labels_neg).float().mean().item()
                    acc_parts.append(neg_acc)
                total_acc = sum(acc_parts) / float(max(1, len(acc_parts)))
            else:
                pos_acc = 0.0
                hard_acc = 0.0
                neg_acc = 0.0
                total_acc = 0.0

            val_itc_acc += itc_acc
            val_pos_acc += pos_acc
            val_hard_acc += hard_acc
            val_neg_acc += neg_acc
            val_total_acc += total_acc
            val_batches += 1

            # --- Accumulate for global classification + embedding metrics ---
            if cls_metrics_enabled and cls_logits_available:
                pos_probs = torch.softmax(logits_pos.detach().float(), dim=1)[:, 1]
                all_cls_probs.append(pos_probs.cpu())
                all_cls_labels.append(labels_pos.cpu())
                all_cls_sources.extend(['pos'] * batch_size)

                if cls_hard_logits_available:
                    hard_probs = torch.softmax(logits_hard.detach().float(), dim=1)[:, 1]
                    all_cls_probs.append(hard_probs.cpu())
                    all_cls_labels.append(labels_neg[:batch_size].cpu())
                    all_cls_sources.extend(['hard'] * batch_size)

                if has_negatives and cls_extra_neg_logits_available:
                    neg_probs_val = torch.softmax(logits_neg.detach().float(), dim=1)[:, 1]
                    all_cls_probs.append(neg_probs_val.cpu())
                    all_cls_labels.append(labels_neg[:batch_size].cpu())
                    all_cls_sources.extend(['neg'] * batch_size)

            all_feat_g.append(feat_g.detach().cpu())
            all_feat_o.append(feat_o.detach().cpu())
            all_gw_indices.append(gw_indices.cpu())

            # Memory cleanup in validation loop
            del g, z_l, h_l, sim_g2o, sim_g2o_d, itc_loss, cls_loss, total_loss
            del logits_pos, logits_hard, pos_loss, hard_loss, feat_g, feat_o
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
            retrieval[key] = sum(d[key] for d in retrieval_metrics_accum) / len(retrieval_metrics_accum)
    if fusion_gallery_metrics_accum:
        for key in fusion_gallery_metrics_accum[0]:
            retrieval[key] = sum(d[key] for d in fusion_gallery_metrics_accum) / len(fusion_gallery_metrics_accum)

    # Global classification metrics
    if all_cls_probs:
        cls_metrics = compute_classification_metrics(
            torch.cat(all_cls_probs),
            torch.cat(all_cls_labels),
            all_sources=all_cls_sources
        )
    else:
        cls_metrics = {}

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
        "total_acc": val_total_acc / val_batches,
        "retrieval": retrieval,
        "classification": cls_metrics,
        "embedding": emb_metrics,
    }
    if any(int(v.get("count", 0)) > 0 for v in dt_stats.values()):
        metrics["time_delta"] = _finalize_time_delta_stats(dt_stats)
    if hard_total > 0:
        mining = {
            "hardneg_fallback_rate": float(hard_fallback_total) / float(max(1, hard_total)),
        }
        for i, wnd in enumerate(hard_window_days):
            mining[f"hardneg_window_level_{i}_hit_rate"] = float(hard_window_hits[i]) / float(max(1, hard_total))
            mining[f"hardneg_window_level_{i}_days"] = float(wnd)
        metrics["hardneg_mining"] = mining

    model.train()
    # Memory cleanup after validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return metrics


def train(args):
    if args.temp_final is None:
        args.temp_final = args.temp_init
    if args.temp_min <= 0 or args.temp_max <= 0:
        raise ValueError("temp_min and temp_max must be > 0.")
    if args.temp_min >= args.temp_max:
        raise ValueError("temp_min must be < temp_max.")

    if args.hardneg_min_candidates < 1:
        raise ValueError("hardneg_min_candidates must be >= 1.")
    if args.hardneg_memory_bank_size < 1:
        raise ValueError("hardneg_memory_bank_size must be >= 1.")
    if args.hardneg_memory_topk < 1:
        raise ValueError("hardneg_memory_topk must be >= 1.")
    if args.hardneg_memory_interval < 1:
        raise ValueError("hardneg_memory_interval must be >= 1.")
    if args.hardneg_memory_max_rows < 1:
        raise ValueError("hardneg_memory_max_rows must be >= 1.")
    if args.gallery_hard_neg_topk < 1:
        raise ValueError("gallery_hard_neg_topk must be >= 1.")
    if args.gallery_hard_neg_weight < 0:
        raise ValueError("gallery_hard_neg_weight must be >= 0.")
    if args.gallery_hard_neg_start_after_retrieval_epochs < 0:
        raise ValueError("gallery_hard_neg_start_after_retrieval_epochs must be >= 0.")
    if args.gallery_hard_neg_ramp_epochs < 0:
        raise ValueError("gallery_hard_neg_ramp_epochs must be >= 0.")
    if args.gallery_loss_weight < 0:
        raise ValueError("gallery_loss_weight must be >= 0.")
    if args.gallery_score_chunk_size < 1:
        raise ValueError("gallery_score_chunk_size must be >= 1.")
    args.gallery_distractor_time_mode = str(args.gallery_distractor_time_mode).strip().lower()
    if args.gallery_distractor_time_mode in {"synthetic", "after_gw"}:
        args.gallery_distractor_time_mode = "synthetic_after_gw"
    if args.gallery_distractor_time_mode in {"real", "none"}:
        args.gallery_distractor_time_mode = "actual"
    if args.gallery_distractor_time_mode not in {"actual", "synthetic_after_gw"}:
        raise ValueError("gallery_distractor_time_mode must be 'actual' or 'synthetic_after_gw'.")
    if args.gallery_distractor_time_window_days <= 0:
        raise ValueError("gallery_distractor_time_window_days must be > 0.")
    if args.time_delta_cls_scale_days is None:
        args.time_delta_cls_scale_days = float(args.time_compat_tau_days)
    if args.time_delta_cls_scale_days <= 0:
        raise ValueError("time_delta_cls_scale_days must be > 0.")
    if args.time_delta_cls_clip < 0:
        raise ValueError("time_delta_cls_clip must be >= 0.")
    args.fusion_mode = normalize_fusion_mode(getattr(args, "fusion_mode", None), dual_fusion=args.dual_fusion)
    args.dual_fusion = args.fusion_mode != "legacy_g2o"
    args._hardneg_window_days = parse_day_windows(args.hardneg_time_window_days)


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args._dataset_window_metadata = _collect_dataset_window_metadata(args)
    args._effective_input_window_metadata = _build_effective_input_window_metadata(args)
    print(f"Dataset window metadata: {json.dumps(args._dataset_window_metadata, indent=2)}")
    print(f"Effective input window metadata: {json.dumps(args._effective_input_window_metadata, indent=2)}")
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
    need_zero_time_mjd_for_cls = bool(args.use_time_delta_cls_feature or args.gallery_loss_weight > 0)
    print(f"CLS/gallery absolute-time metadata requested: {int(need_zero_time_mjd_for_cls)}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(os.path.dirname(args.ckpt_path), "tb_logs", f"run_albef_{timestamp}_pid{os.getpid()}")
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
    if args.val_split is not None and 0 < args.val_split < 1:
        if args.itc_loss_type == "supcon":
            train_loader, val_loader, steps_per_epoch, val_steps = create_supcon_dataloaders(
                h5_path=args.data_path,
                batch_size=args.batch_size,
                samples_per_gw=args.samples_per_gw,
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
            )
            print(
                f"SupCon mode: {args.samples_per_gw} samples/GW, "
                f"{args.batch_size // args.samples_per_gw} GW/batch"
            )
        else:
            train_loader, val_loader, steps_per_epoch, val_steps = create_train_val_dataloaders(
                h5_path=args.data_path,
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
            )
        print(f"Train Steps/Epoch: {steps_per_epoch} | Val Steps/Epoch: {val_steps}")
    else:
        if args.steps_per_epoch is not None:
            steps_per_epoch = args.steps_per_epoch
        else:
            with h5py.File(args.data_path, 'r') as f:
                total_optical = f['events/optical_data/values'].shape[0]
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
        )

    mtan_cfg = resolve_mtan_runtime_config(args)
    print("mTAN runtime config:")
    print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in mtan_cfg.items()}, indent=2))

    model = GWOpticalALBEFModel(
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
        print("Legacy dual cross-attention fusion enabled with per-pair credible level.")
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
        print("Retrieval-only mode enabled: fusion/classification branch disabled; extra negatives go into contrastive loss.")

    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("Model compiled with torch.compile() for optimized execution.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
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
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt.get("epoch", -1) + 1
        global_step = start_epoch * steps_per_epoch
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
    hardneg_memory = HardNegativeMemoryBank(args, device)
    print(
        "Hard-negative mining config: "
        f"windows_days={args._hardneg_window_days}, "
        f"min_candidates={args.hardneg_min_candidates}, "
        f"memory_enable={args.hardneg_memory_bank_enable}, "
        f"memory_size={args.hardneg_memory_bank_size}, topk={args.hardneg_memory_topk}, "
        f"warmup_steps={args.hardneg_memory_warmup_steps}, "
        f"memory_interval={args.hardneg_memory_interval}, "
        f"memory_max_rows={args.hardneg_memory_max_rows}"
    )

    # Prevent accidental test leakage during HPO by forcing OOD monitor off.
    if args.hpo_trial_number is not None and getattr(args, "enable_ood_monitoring", False):
        print("WARNING: Disabling OOD monitoring during HPO trial to avoid test leakage.")
        args.enable_ood_monitoring = False

    # Build OOD monitoring loader from independent dataset (opt-in only).
    ood_val_loader = None
    ood_event_time_mjd_table = None
    if getattr(args, 'test_data_path', None):
        if not os.path.exists(args.test_data_path):
            print(f"WARNING: test_data_path not found, skip OOD monitoring: {args.test_data_path}")
        elif not getattr(args, "enable_ood_monitoring", False):
            print(
                "OOD monitoring is disabled by default to prevent test leakage. "
                "Pass --enable_ood_monitoring ONLY when test_data_path is a development set (not final test set)."
            )
        else:
            ood_test_steps = args.test_steps
            ood_batch_size = args.val_batch_size or args.batch_size
            # Cap batch size to number of unique GW events in the OOD dataset
            ood_gw_map = build_gw_to_lc_mapping(args.test_data_path)
            n_ood_gw = len(ood_gw_map)
            if ood_batch_size > n_ood_gw:
                print(f"Reducing OOD test batch_size from {ood_batch_size} to {n_ood_gw} (available GW events)")
                ood_batch_size = n_ood_gw
            ood_val_loader = create_training_dataloader(
                h5_path=args.test_data_path,
                batch_size=ood_batch_size,
                steps_per_epoch=ood_test_steps,
                num_workers=args.num_workers,
                pin_memory=bool(args.pin_memory),
                persistent_workers=bool(args.persistent_workers),
                prefetch_factor=args.prefetch_factor,
                cache_in_memory=False,
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
                loader_usage="ood",
                loader_label="OOD DataLoader",
            )
            ood_event_time_mjd_table = load_gw_event_time_mjd_table(args.test_data_path, device)
            if ood_event_time_mjd_table is None:
                raise ValueError(
                    "Hard-negative time-shift re-encoding requires 'events/gw_data/event_time_mjd' "
                    "in OOD dataset."
                )
            print(
                f"OOD test loader: {args.test_data_path} "
                f"({ood_test_steps} steps, batch_size={ood_batch_size})"
            )

    pbar_update_every = 500
    tb_log_interval = 50
    best_val_score = None
    best_epoch_idx = None
    best_val_metrics = {}
    epochs_no_improve = 0
    best_tracking_started = False
    best_tracking_start_epoch = None
    last_selection_score = None
    lr_scheduler = build_lr_scheduler(optimizer, args, steps_per_epoch, global_step)

    for epoch in range(start_epoch, args.epochs):
        epoch_total = 0.0
        epoch_itc = 0.0
        epoch_cls = 0.0
        hardneg_level_hits_epoch = [0 for _ in args._hardneg_window_days]
        hardneg_fallback_epoch = 0
        hardneg_total_epoch = 0
        hardneg_mem_attempted_epoch = 0
        hardneg_mem_selected_epoch = 0
        hardneg_mem_fallback_epoch = 0

        ref_time_cache = None
        apply_temperature_schedule(model, args, epoch)
        cls_weight = compute_cls_weight(args, epoch)
        itc_weight = compute_itc_weight(args, epoch)
        hard_neg_ratio = compute_hard_neg_ratio(args, epoch)
        best_selection_eligible = is_best_ckpt_selection_eligible(
            args, epoch, hard_neg_ratio=hard_neg_ratio
        )
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
            elif hard_neg_full_activation_reachable(args):
                print(
                    "Best-checkpoint selection activated "
                    f"at epoch {epoch+1} (hard_neg_ratio={hard_neg_ratio:.3f})."
                )
            else:
                print(
                    "Best-checkpoint selection activated "
                    f"at epoch {epoch+1} (hard negatives disabled or unreachable in this run)."
                )

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            mininterval=0,
            miniters=pbar_update_every
        )
        
        for batch_idx, batch_data in enumerate(pbar):
            _opt_zero_time_mjd_base = None
            _neg_zero_time_mjd_base = None
            _neg_zero_time_mjd_cls_base = None
            _opt_first_detection_mjd = None
            if has_negatives:
                if len(batch_data) >= 17:
                    (
                        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                        neg_t, neg_v, neg_mask, neg_err, neg_coords,
                        _opt_zero_time_mjd_base, _neg_zero_time_mjd_base, _neg_zero_time_mjd_cls_base,
                        _opt_first_detection_mjd,
                    ) = batch_data[:17]
                elif len(batch_data) >= 16:
                    (
                        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                        neg_t, neg_v, neg_mask, neg_err, neg_coords,
                        _opt_zero_time_mjd_base, _neg_zero_time_mjd_base, _neg_zero_time_mjd_cls_base,
                    ) = batch_data[:16]
                elif len(batch_data) >= 15:
                    (
                        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                        neg_t, neg_v, neg_mask, neg_err, neg_coords,
                        _opt_zero_time_mjd_base, _neg_zero_time_mjd_base,
                    ) = batch_data[:15]
                else:
                    (
                        gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                        neg_t, neg_v, neg_mask, neg_err, neg_coords,
                    ) = batch_data[:13]
            else:
                if len(batch_data) >= 10:
                    gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices, _opt_zero_time_mjd_base, _opt_first_detection_mjd = batch_data[:10]
                elif len(batch_data) >= 9:
                    gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices, _opt_zero_time_mjd_base = batch_data[:9]
                else:
                    gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data

            gw_s = gw_s.to(device, non_blocking=True)
            gw_m = gw_m.to(device, non_blocking=True)
            gw_s, gw_m = augment_gw_data(
                gw_s, gw_m, training=True,
                noise_std=args.gw_aug_noise,
                scalar_jitter=args.gw_aug_jitter,
                channel_dropout_prob=args.gw_aug_dropout
            )
            opt_t = opt_t.to(device, non_blocking=True)
            opt_v = opt_v.to(device, non_blocking=True)
            opt_mask = opt_mask.to(device, non_blocking=True)
            opt_err = opt_err.to(device, non_blocking=True)
            opt_coords = opt_coords.to(device, non_blocking=True)
            gw_indices = gw_indices.to(device, non_blocking=True).long()
            batch_event_time_mjd = None
            if gw_event_time_mjd_table is not None:
                batch_event_time_mjd = gw_event_time_mjd_table[gw_indices]

            if has_negatives:
                neg_t = neg_t.to(device, non_blocking=True)
                neg_v = neg_v.to(device, non_blocking=True)
                neg_mask = neg_mask.to(device, non_blocking=True)
                neg_err = neg_err.to(device, non_blocking=True)
                neg_coords = neg_coords.to(device, non_blocking=True)
                if _neg_zero_time_mjd_base is not None:
                    _neg_zero_time_mjd_base = _neg_zero_time_mjd_base.to(device, non_blocking=True)
                if _neg_zero_time_mjd_cls_base is not None:
                    _neg_zero_time_mjd_cls_base = _neg_zero_time_mjd_cls_base.to(device, non_blocking=True)
            if _opt_first_detection_mjd is not None:
                _opt_first_detection_mjd = _opt_first_detection_mjd.to(device, non_blocking=True)
            elif _opt_zero_time_mjd_base is not None:
                _opt_first_detection_mjd = _opt_zero_time_mjd_base.to(device, non_blocking=True)
            opt_t_raw = opt_t.clone()
            opt_v_raw = opt_v.clone()
            opt_mask_raw = opt_mask.clone()
            opt_err_raw = opt_err.clone()

            opt_t, opt_v, opt_mask, opt_err = augment_optical_data(
                opt_t, opt_v, opt_mask, opt_err, training=True,
                time_jitter=args.opt_aug_time_jitter,
                flux_noise=args.opt_aug_noise,
                obs_dropout=args.opt_aug_dropout,
                band_dropout=args.opt_aug_band_dropout
            )
            if has_negatives:
                neg_t, neg_v, neg_mask, neg_err = augment_optical_data(
                    neg_t, neg_v, neg_mask, neg_err, training=True,
                    time_jitter=args.opt_aug_time_jitter,
                    flux_noise=args.opt_aug_noise,
                    obs_dropout=args.opt_aug_dropout,
                    band_dropout=args.opt_aug_band_dropout
                )

            batch_size = gw_s.size(0)

            if (
                ref_time_cache is None
                or ref_time_cache.shape[0] != batch_size
                or ref_time_cache.dtype != opt_t.dtype
            ):
                ref_time_cache = build_ref_time(
                    batch_size, args.n_ref, args.ref_start, args.ref_end, device, opt_t.dtype
                )
            opt_ref_t = ref_time_cache

            should_tb_log = (global_step % tb_log_interval == 0)
            should_perf_log = should_tb_log
            step_cpu_start = time.perf_counter() if should_perf_log else None
            inbatch_hard_mine_ms = 0.0
            memory_hard_mine_ms = 0.0
            memory_fetch_ms = 0.0
            hard_branch_total_ms = 0.0

            optimizer.zero_grad(set_to_none=True)
            need_cred_level = _model_requires_cred_level(model)

            with autocast(device_type='cuda', dtype=amp_dtype, enabled=(device.type == 'cuda')):
                g, z_l, h_l, H_gw = model.encode(
                    gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
                )
                extra_neg_z_itc = None
                extra_neg_event_time_mjd = None
                if has_negatives and not cls_branch_enabled:
                    neg_zero_time_mjd_for_itc = _neg_zero_time_mjd_cls_base
                    if neg_zero_time_mjd_for_itc is None:
                        neg_zero_time_mjd_for_itc = _neg_zero_time_mjd_base
                    with torch.no_grad():
                        extra_neg_z_itc, _ = model.encode_optical(
                            neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err,
                        )
                    extra_neg_event_time_mjd = neg_zero_time_mjd_for_itc
                if args.itc_loss_type == "supcon":
                    itc_loss, sim_g2o, sim_g2o_for_loss = model.compute_supcon_loss(
                        g, z_l, gw_indices, margin=args.supcon_margin,
                        gw_event_time_mjd=batch_event_time_mjd,
                        opt_event_time_mjd=resolve_optical_candidate_time_mjd(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        ),
                        extra_neg_z=extra_neg_z_itc,
                        extra_neg_opt_event_time_mjd=extra_neg_event_time_mjd,
                    )
                else:
                    itc_loss, sim_g2o, sim_g2o_for_loss = model.compute_itc_loss(
                        g, z_l, gw_indices, mask=args.mask_itc,
                        gw_event_time_mjd=batch_event_time_mjd,
                        opt_event_time_mjd=resolve_optical_candidate_time_mjd(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        ),
                        extra_neg_z=extra_neg_z_itc,
                        extra_neg_opt_event_time_mjd=extra_neg_event_time_mjd,
                    )

                feat_g_hard, feat_o_hard = model.get_contrastive_embeddings(g, z_l)
                labels_pos = torch.ones(batch_size, device=device, dtype=torch.long)
                labels_neg = torch.zeros(batch_size, device=device, dtype=torch.long)
                gallery_loss = torch.zeros((), device=device)
                gallery_hard_loss = torch.zeros((), device=device)
                gallery_hard_weight = 0.0

                if cls_branch_enabled:
                    # Compute per-pair credible level for dual fusion
                    cred_level = compute_credible_level(gw_m, opt_coords) if need_cred_level else None
                    if _opt_first_detection_mjd is not None and batch_event_time_mjd is not None:
                        dt_pos = compute_time_delta_days(_opt_first_detection_mjd, batch_event_time_mjd)
                    else:
                        dt_pos = torch.zeros((batch_size,), device=device, dtype=torch.float32)
                    logits_pos = model.fusion_logits(
                        g, h_l, z_l=z_l, H_gw=H_gw, cred_level=cred_level,
                        gw_s=gw_s, gw_m=gw_m, opt_coords=opt_coords,
                        dt_days=dt_pos,
                    )
                    pos_loss = model.cls_criterion(logits_pos, labels_pos)

                    hard_branch_cpu_start = time.perf_counter() if should_perf_log else None
                    easy_idx = sample_easy_negatives(batch_size, device)
                    choose_hard_mask = None
                    inbatch_hard_idx = None
                    active_rows = None
                    if easy_idx is None:
                        neg_idx = None
                    else:
                        neg_idx = easy_idx.clone()
                        inbatch_hard_idx = easy_idx.clone()
                        if hard_neg_ratio >= 1:
                            choose_hard_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
                        elif hard_neg_ratio <= 0:
                            choose_hard_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
                        else:
                            choose_hard_mask = torch.rand(batch_size, device=device) < hard_neg_ratio

                        if choose_hard_mask.any():
                            active_rows = torch.nonzero(choose_hard_mask, as_tuple=False).squeeze(-1)
                            inbatch_start = time.perf_counter() if should_perf_log else None
                            if batch_event_time_mjd is not None:
                                hard_active_idx, _, _ = sample_inbatch_hard_negatives_with_time(
                                    sim_g2o=sim_g2o,
                                    gw_indices=gw_indices,
                                    batch_event_time_mjd=batch_event_time_mjd,
                                    candidate_time_mjd=resolve_optical_candidate_time_mjd(
                                        _opt_first_detection_mjd, batch_event_time_mjd
                                    ),
                                    window_days=args._hardneg_window_days,
                                    min_candidates=args.hardneg_min_candidates,
                                    semi_hard=args.semi_hard,
                                    semi_hard_margin=args.semi_hard_margin,
                                    fallback_mode=args.hardneg_fallback_mode,
                                    active_rows=active_rows,
                                )
                            elif args.semi_hard:
                                full_hard_idx = model.sample_semi_hard_negatives(
                                    sim_g2o, gw_indices, margin=args.semi_hard_margin
                                )
                                hard_active_idx = full_hard_idx[active_rows]
                            else:
                                full_hard_idx = model.sample_hard_negatives(sim_g2o, gw_indices)
                                hard_active_idx = full_hard_idx[active_rows]
                            inbatch_hard_idx[active_rows] = hard_active_idx
                            neg_idx[active_rows] = hard_active_idx
                            if should_perf_log and inbatch_start is not None:
                                inbatch_hard_mine_ms = (time.perf_counter() - inbatch_start) * 1000.0

                    if neg_idx is None:
                        hard_loss = torch.zeros((), device=device)
                        logits_hard = torch.zeros((batch_size, 2), device=device, dtype=g.dtype)
                        dt_hard = None
                    else:
                        h_l_hard = h_l[neg_idx].clone()
                        z_l_hard = z_l[neg_idx].clone()
                        coords_hard = opt_coords[neg_idx].clone()
                        if batch_event_time_mjd is not None:
                            dt_hard = compute_time_delta_days(
                                gather_candidate_optical_time_mjd(
                                    _opt_first_detection_mjd, batch_event_time_mjd, neg_idx
                                ),
                                batch_event_time_mjd,
                            )
                        else:
                            dt_hard = torch.zeros((batch_size,), device=device, dtype=torch.float32)
                        if choose_hard_mask is None:
                            choose_hard_mask = torch.ones(batch_size, device=device, dtype=torch.bool)
                        # NOTE: encode_hard_negative_with_time_shift removed.
                        # Hard negatives now use the first-detection-aligned features
                        # already encoded in h_l[neg_idx], z_l[neg_idx] above.

                        # B5 memory-bank mining: replace selected hard rows with cross-batch candidates.
                        if (
                            args.hardneg_memory_bank_enable
                            and hard_neg_ratio > 0
                            and choose_hard_mask is not None
                            and choose_hard_mask.any()
                            and batch_event_time_mjd is not None
                            and (global_step % args.hardneg_memory_interval == 0)
                        ):
                            mem_rows = active_rows
                            if mem_rows is None:
                                mem_rows = torch.nonzero(choose_hard_mask, as_tuple=False).squeeze(-1)
                            if mem_rows.numel() > int(args.hardneg_memory_max_rows):
                                perm = torch.randperm(mem_rows.numel(), device=device)[: int(args.hardneg_memory_max_rows)]
                                mem_rows = mem_rows[perm]

                            temperature = model.log_temp.exp().clamp(min=model.temp_min, max=model.temp_max)
                            if should_perf_log and device.type == "cuda":
                                mem_timer = _start_cuda_timer(True)
                            else:
                                mem_timer = None
                                mem_cpu_start = time.perf_counter() if should_perf_log else None
                            mem_idx, _, mem_fallback, mem_attempted = hardneg_memory.mine(
                                feat_g=feat_g_hard,
                                gw_indices=gw_indices,
                                batch_event_time_mjd=batch_event_time_mjd,
                                temperature=temperature,
                                window_days=args._hardneg_window_days,
                                min_candidates=args.hardneg_min_candidates,
                                global_step=global_step,
                                active_rows=mem_rows,
                            )
                            if should_perf_log:
                                if mem_timer is not None:
                                    memory_hard_mine_ms = _stop_cuda_timer(mem_timer)
                                elif mem_cpu_start is not None:
                                    memory_hard_mine_ms = (time.perf_counter() - mem_cpu_start) * 1000.0
                            hardneg_mem_attempted_epoch += int(mem_attempted)
                            hardneg_mem_fallback_epoch += int(mem_fallback)
                            use_mem_mask = mem_idx >= 0
                            if use_mem_mask.any():
                                if should_perf_log and device.type == "cuda":
                                    fetch_timer = _start_cuda_timer(True)
                                else:
                                    fetch_timer = None
                                    fetch_cpu_start = time.perf_counter() if should_perf_log else None
                                fetched = hardneg_memory.fetch(
                                    mem_idx[use_mem_mask],
                                    device=device,
                                    h_l_dtype=h_l.dtype,
                                    z_l_dtype=z_l.dtype,
                                    opt_dtype=opt_t.dtype,
                                )
                                if should_perf_log:
                                    if fetch_timer is not None:
                                        memory_fetch_ms = _stop_cuda_timer(fetch_timer)
                                    elif fetch_cpu_start is not None:
                                        memory_fetch_ms = (time.perf_counter() - fetch_cpu_start) * 1000.0
                                if fetched is not None:
                                    use_rows = mem_rows[use_mem_mask]
                                    (
                                        _h_sel,
                                        _z_sel,
                                        c_sel,
                                        t_sel,
                                        _gw_idx_sel,
                                        opt_t_sel,
                                        opt_v_sel,
                                        opt_mask_sel,
                                        opt_err_sel,
                                    ) = fetched
                                    # Use stored features directly (no time-shift re-encoding)
                                    h_sel_re = _h_sel.to(device=h_l.device, dtype=h_l.dtype)
                                    z_sel_re = _z_sel.to(device=z_l.device, dtype=z_l.dtype)
                                    if batch_event_time_mjd is not None:
                                        dt_sel = compute_time_delta_days(t_sel, batch_event_time_mjd[use_rows])
                                    else:
                                        dt_sel = torch.zeros((use_rows.numel(),), device=device, dtype=torch.float32)
                                    h_l_hard[use_rows] = h_sel_re
                                    z_l_hard[use_rows] = z_sel_re
                                    coords_hard[use_rows] = c_sel
                                    dt_hard[use_rows] = dt_sel
                                    hardneg_mem_selected_epoch += int(use_rows.numel())

                        cred_level_hard = compute_credible_level(gw_m, coords_hard) if need_cred_level else None
                        logits_hard = model.fusion_logits(
                            g, h_l_hard, z_l=z_l_hard, H_gw=H_gw, cred_level=cred_level_hard,
                            gw_s=gw_s, gw_m=gw_m, opt_coords=coords_hard,
                            dt_days=dt_hard,
                        )
                        hard_loss = model.cls_criterion(logits_hard, labels_neg)
                        if choose_hard_mask is not None:
                            hard_count = int(choose_hard_mask.sum().item())
                            hardneg_total_epoch += hard_count
                            if dt_hard is not None and batch_event_time_mjd is not None and hard_count > 0:
                                abs_dt = dt_hard[choose_hard_mask].abs()
                                assigned = torch.zeros_like(abs_dt, dtype=torch.bool)
                                for li, wnd in enumerate(args._hardneg_window_days):
                                    hit = (~assigned) & (abs_dt <= float(wnd))
                                    cnt = int(hit.sum().item())
                                    hardneg_level_hits_epoch[li] += cnt
                                    assigned |= hit
                                hardneg_fallback_epoch += int((~assigned).sum().item())
                    if should_perf_log and hard_branch_cpu_start is not None:
                        hard_branch_total_ms = (time.perf_counter() - hard_branch_cpu_start) * 1000.0

                    gallery_extra_h_l = None
                    gallery_extra_z_l = None
                    gallery_extra_coords = None
                    gallery_extra_event_time = None
                    if has_negatives:
                        z_l_neg, h_l_neg = model.encode_optical(
                            neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err
                        )
                        cred_level_neg = compute_credible_level(gw_m, neg_coords) if need_cred_level else None
                        neg_zero_time_mjd_for_cls = _neg_zero_time_mjd_cls_base
                        if neg_zero_time_mjd_for_cls is None:
                            neg_zero_time_mjd_for_cls = _neg_zero_time_mjd_base
                        neg_event_time = (
                            neg_zero_time_mjd_for_cls.to(device=device, dtype=torch.float32)
                            if neg_zero_time_mjd_for_cls is not None
                            else None
                        )
                        dt_neg = (
                            compute_time_delta_days(neg_event_time, batch_event_time_mjd)
                            if (neg_event_time is not None and batch_event_time_mjd is not None)
                            else torch.zeros((batch_size,), device=device, dtype=torch.float32)
                        )
                        logits_neg = model.fusion_logits(
                            g, h_l_neg, z_l=z_l_neg, H_gw=H_gw, cred_level=cred_level_neg,
                            gw_s=gw_s, gw_m=gw_m, opt_coords=neg_coords,
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
                        pos_loss, hard_loss, neg_loss, has_negatives, args
                    )

                    gallery_loss = torch.zeros((), device=device)
                    gallery_hard_loss = torch.zeros((), device=device)
                    gallery_hard_weight = 0.0
                    total_loss = (
                        itc_weight * itc_loss
                        + cls_weight * cls_loss
                    )
                else:
                    logits_pos = torch.zeros((batch_size, 2), device=device, dtype=g.dtype)
                    logits_hard = torch.zeros((batch_size, 2), device=device, dtype=g.dtype)
                    logits_neg = torch.zeros((batch_size, 2), device=device, dtype=g.dtype) if has_negatives else None
                    pos_loss = torch.zeros((), device=device)
                    hard_loss = torch.zeros((), device=device)
                    neg_loss = torch.zeros((), device=device) if has_negatives else None
                    cls_loss = torch.zeros((), device=device)
                    gallery_loss = torch.zeros((), device=device)
                    gallery_hard_loss = torch.zeros((), device=device)
                    gallery_hard_weight = 0.0
                    dt_hard = None
                    if is_retrieval_active(args, epoch) and has_negatives:
                        z_l_neg, h_l_neg = model.encode_optical(
                            neg_coords, neg_t, neg_v, opt_ref_t, neg_mask, neg_err
                        )
                        gallery_extra_h_l = h_l_neg
                        gallery_extra_z_l = z_l_neg
                        gallery_extra_coords = neg_coords
                        neg_event_time = (
                            _neg_zero_time_mjd_cls_base.to(device=device, dtype=torch.float32)
                            if _neg_zero_time_mjd_cls_base is not None
                            else (
                                _neg_zero_time_mjd_base.to(device=device, dtype=torch.float32)
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
                if is_retrieval_active(args, epoch) and gallery_extra_h_l is not None:
                    h_candidates = [h_l]
                    z_candidates = [z_l]
                    coords_candidates = [opt_coords]
                    candidate_gw_indices = [gw_indices]
                    candidate_time_blocks = []
                    if batch_event_time_mjd is not None:
                        pos_gallery_time = resolve_optical_candidate_time_mjd(
                            _opt_first_detection_mjd, batch_event_time_mjd
                        ).to(device=device, dtype=torch.float32)
                        candidate_time_blocks.append(pos_gallery_time)
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
                            torch.full((batch_size,), -1, device=device, dtype=gw_indices.dtype)
                        )
                        if batch_event_time_mjd is not None:
                            if gallery_extra_event_time is None:
                                candidate_time_blocks.append(
                                    torch.full((batch_size,), float("nan"), device=device, dtype=torch.float32)
                                )
                            else:
                                candidate_time_blocks.append(
                                    gallery_extra_event_time.to(device=device, dtype=torch.float32)
                                )
                    h_gallery = torch.cat(h_candidates, dim=0)
                    z_gallery = torch.cat(z_candidates, dim=0)
                    coords_gallery = torch.cat(coords_candidates, dim=0)
                    gallery_gw_indices = torch.cat(candidate_gw_indices, dim=0)
                    gallery_pos_mask_full = gw_indices.unsqueeze(1) == gallery_gw_indices.unsqueeze(0)
                    gallery_candidate_time = torch.cat(candidate_time_blocks, dim=0) if candidate_time_blocks else None
                    dt_gallery = build_gallery_time_delta_matrix(
                        batch_event_time_mjd,
                        gallery_candidate_time,
                        positive_mask=gallery_pos_mask_full,
                        distractor_time_mode=args.gallery_distractor_time_mode,
                        distractor_time_window_days=args.gallery_distractor_time_window_days,
                    )

                    # Query subsampling: randomly select a subset of queries to
                    # bound GPU memory when scoring against the full candidate gallery.
                    max_queries = int(getattr(args, "max_gallery_queries", 0) or 0)
                    n_q_total = int(g.size(0))
                    if max_queries > 0 and n_q_total > max_queries:
                        q_idx = torch.randperm(n_q_total, device=device)[:max_queries]
                        g_q = g[q_idx]
                        H_q = H_gw[q_idx] if H_gw is not None else None
                        gw_s_q = gw_s[q_idx] if gw_s is not None else None
                        gw_m_q = gw_m[q_idx] if gw_m is not None else None
                        dt_q = dt_gallery[q_idx, :] if dt_gallery is not None else None
                        gallery_pos_mask = gallery_pos_mask_full[q_idx, :]
                    else:
                        g_q, H_q, gw_s_q, gw_m_q = g, H_gw, gw_s, gw_m
                        dt_q = dt_gallery
                        gallery_pos_mask = gallery_pos_mask_full

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
                    gallery_loss = compute_fusion_gallery_nce_loss(gallery_scores, gallery_pos_mask)
                    total_loss = total_loss + float(args.gallery_loss_weight) * gallery_loss
                    gallery_hard_weight = compute_gallery_hard_neg_weight(args, epoch)
                    if gallery_hard_weight > 0.0:
                        gallery_hard_loss = compute_fusion_gallery_hard_nce_loss(
                            gallery_scores, gallery_pos_mask,
                            topk=int(args.gallery_hard_neg_topk),
                        )
                        total_loss = total_loss + (
                            float(args.gallery_loss_weight)
                            * gallery_hard_weight
                            * gallery_hard_loss
                        )

            scaler.scale(total_loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if lr_scheduler is not None:
                lr_scheduler.step()
            clamp_temperature(model, args)

            total_val = total_loss.item()
            itc_val = itc_loss.item()
            pos_loss_val = pos_loss.item()
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
                pred_gw = gw_indices[itc_preds]
                itc_acc = (gw_indices == pred_gw).float().mean().item()

                if cls_branch_enabled:
                    pos_acc = (logits_pos.argmax(dim=1) == labels_pos).float().mean().item()
                    hard_acc = (logits_hard.argmax(dim=1) == labels_neg).float().mean().item()
                    neg_acc = None
                    if has_negatives:
                        neg_acc = (logits_neg.argmax(dim=1) == labels_neg).float().mean().item()
                else:
                    pos_acc = 0.0
                    hard_acc = 0.0
                    neg_acc = 0.0 if has_negatives else None
                current_temp = model.log_temp.exp().item()

            if cls_branch_enabled and batch_event_time_mjd is not None:
                hardneg_memory.update(
                    feat_o=feat_o_hard,
                    h_l=h_l,
                    z_l=z_l,
                    opt_coords=opt_coords,
                    candidate_time_mjd=resolve_optical_candidate_time_mjd(
                        _opt_first_detection_mjd, batch_event_time_mjd
                    ),
                    gw_indices=gw_indices,
                    opt_t=opt_t_raw,
                    opt_v=opt_v_raw,
                    opt_mask=opt_mask_raw,
                    opt_err=opt_err_raw,
                )

            step_time_ms = 0.0
            if should_perf_log and step_cpu_start is not None:
                step_time_ms = (time.perf_counter() - step_cpu_start) * 1000.0

            if should_tb_log:
                writer.add_scalar('Train/Batch_Total_Loss', total_val, global_step)
                writer.add_scalar('Train/Batch_ITC_Loss', itc_val, global_step)
                writer.add_scalar('Train/Batch_CLS_Loss', cls_val, global_step)
                writer.add_scalar('Train/Batch_Gallery_Loss', gallery_loss_val, global_step)
                writer.add_scalar('Train/Batch_GalleryHard_Loss', gallery_hard_loss_val, global_step)
                writer.add_scalar('Train/Batch_GalleryHard_Weight', gallery_hard_weight, global_step)
                writer.add_scalar('Train/Batch_Pos_Loss', pos_loss_val, global_step)
                writer.add_scalar('Train/Batch_HardNeg_Loss', hard_loss_val, global_step)
                if has_negatives:
                    writer.add_scalar('Train/Batch_ExtraNeg_Loss', neg_loss_val, global_step)
                writer.add_scalar('Train/Batch_ITC_Acc', itc_acc, global_step)
                writer.add_scalar('Train/Batch_Pos_Acc', pos_acc, global_step)
                writer.add_scalar('Train/Batch_HardNeg_Acc', hard_acc, global_step)
                if has_negatives and neg_acc is not None:
                    writer.add_scalar('Train/Batch_ExtraNeg_Acc', neg_acc, global_step)
                writer.add_scalar('Train/Itc_Weight', itc_weight, global_step)
                writer.add_scalar('Train/Cls_Weight', cls_weight, global_step)
                writer.add_scalar('Train/HardNeg_Ratio', hard_neg_ratio, global_step)
                writer.add_scalar('Train/Temperature', current_temp, global_step)
                writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                if dt_hard is not None and batch_event_time_mjd is not None:
                    dt_hard_mean, dt_hard_std = summarize_time_delta(dt_hard)
                    writer.add_scalar('Train/TimeShift/hard_delta_days_mean', dt_hard_mean, global_step)
                    writer.add_scalar('Train/TimeShift/hard_delta_days_std', dt_hard_std, global_step)
                    hard_q = summarize_abs_time_delta_quantiles(dt_hard, quantiles=(0.5, 0.9, 0.95))
                    writer.add_scalar('Train/TimeShift/hard_delta_abs_p50', hard_q["p50"], global_step)
                    writer.add_scalar('Train/TimeShift/hard_delta_abs_p90', hard_q["p90"], global_step)
                    writer.add_scalar('Train/TimeShift/hard_delta_abs_p95', hard_q["p95"], global_step)
                if hardneg_total_epoch > 0:
                    writer.add_scalar(
                        'Train/HardNeg/hardneg_window_uncovered_rate',
                        float(hardneg_fallback_epoch) / float(max(1, hardneg_total_epoch)),
                        global_step,
                    )
                    for li, wnd in enumerate(args._hardneg_window_days):
                        writer.add_scalar(
                            f'Train/HardNeg/window_level_{li}_hit_rate',
                            float(hardneg_level_hits_epoch[li]) / float(max(1, hardneg_total_epoch)),
                            global_step,
                        )
                        writer.add_scalar(
                            f'Train/HardNeg/window_level_{li}_days',
                            float(wnd),
                            global_step,
                        )
                if hardneg_mem_attempted_epoch > 0:
                    writer.add_scalar(
                        'Train/HardNeg/hardneg_bank_fallback_rate',
                        float(hardneg_mem_fallback_epoch) / float(max(1, hardneg_mem_attempted_epoch)),
                        global_step,
                    )
                    writer.add_scalar(
                        'Train/HardNeg/hardneg_bank_selected_rate',
                        float(hardneg_mem_selected_epoch) / float(max(1, hardneg_mem_attempted_epoch)),
                        global_step,
                    )
                writer.add_scalar('Train/Perf/step_time_ms', step_time_ms, global_step)
                writer.add_scalar('Train/Perf/inbatch_hard_mine_ms', inbatch_hard_mine_ms, global_step)
                writer.add_scalar('Train/Perf/memory_hard_mine_ms', memory_hard_mine_ms, global_step)
                writer.add_scalar('Train/Perf/memory_fetch_ms', memory_fetch_ms, global_step)
                writer.add_scalar('Train/Perf/hard_branch_total_ms', hard_branch_total_ms, global_step)

            if batch_idx % 100 == 0:
                postfix = {
                    'Total': f"{total_val:.4f}",
                    'ITC': f"{itc_val:.4f}",
                    'CLS': f"{cls_val:.4f}",
                    'ITC_Acc': f"{itc_acc:.2f}"
                }
                if dt_hard is not None and batch_event_time_mjd is not None:
                    dt_hard_mean, _ = summarize_time_delta(dt_hard)
                    postfix['hard_dt'] = f"{dt_hard_mean:.2f}"
                if args.gallery_loss_weight > 0:
                    postfix['Gallery'] = f"{gallery_loss_val:.4f}"
                    if gallery_hard_weight > 0:
                        postfix['GalleryHard'] = f"{gallery_hard_loss_val:.4f}"
                if last_selection_score is not None:
                    postfix[f"{args.best_ckpt_metric}"] = f"{last_selection_score:.4f}"
                pbar.set_postfix(postfix)

            # Memory cleanup: delete large tensors to free memory
            del g, z_l, h_l, sim_g2o, sim_g2o_detached, itc_loss, total_loss
            del logits_pos, logits_hard, pos_loss, hard_loss, cls_loss, gallery_loss, gallery_hard_loss
            if has_negatives:
                del logits_neg, neg_loss
            
            # Periodic aggressive memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                if device.type == 'cuda':
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

        writer.add_scalar('Train/Epoch_Total_Loss', avg_total, epoch)
        writer.add_scalar('Train/Epoch_ITC_Loss', avg_itc, epoch)
        writer.add_scalar('Train/Epoch_CLS_Loss', avg_cls, epoch)
        
        # Flush tensorboard to prevent memory accumulation
        writer.flush()

        stop_early = False
        if val_loader is not None:
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                args,
                epoch,
                amp_dtype=amp_dtype,
                gw_event_time_mjd_table=gw_event_time_mjd_table,
            )
            _close_loader_dataset_handles(
                val_loader,
                cache_in_memory=bool(args.cache_in_memory),
                label="Validation DataLoader",
            )
            if val_metrics is not None:
                print(
                    f"Val Avg Total: {val_metrics['total']:.4f} | "
                    f"ITC: {val_metrics['itc']:.4f} | CLS: {val_metrics['cls']:.4f}"
                )
                writer.add_scalar('Val/Epoch_Total_Loss', val_metrics['total'], epoch)
                writer.add_scalar('Val/Epoch_ITC_Loss', val_metrics['itc'], epoch)
                writer.add_scalar('Val/Epoch_CLS_Loss', val_metrics['cls'], epoch)
                writer.add_scalar('Val/Epoch_Gallery_Loss', val_metrics.get('gallery', 0.0), epoch)
                writer.add_scalar('Val/Epoch_GalleryHard_Loss', val_metrics.get('gallery_hard', 0.0), epoch)
                # Retrieval metrics
                ret = val_metrics.get('retrieval', {})
                for k, v in ret.items():
                    writer.add_scalar(f'Val/Retrieval/{k}', v, epoch)

                # Classification metrics
                cls_m = val_metrics.get('classification', {})
                for k in ('auroc', 'auprc', 'f1_optimal', 'f1_threshold', 'ece'):
                    if k in cls_m:
                        writer.add_scalar(f'Val/Classification/{k}', cls_m[k], epoch)
                for k in ('acc_pos', 'acc_hard', 'acc_neg', 'acc_neg_gw', 'acc_total'):
                    if k in cls_m:
                        writer.add_scalar(f'Val/Classification/{k}', cls_m[k], epoch)

                # Embedding quality metrics
                emb = val_metrics.get('embedding', {})
                for k, v in emb.items():
                    writer.add_scalar(f'Val/Embedding/{k}', v, epoch)
                dt_m = val_metrics.get("time_delta", {})
                for key, value in dt_m.items():
                    writer.add_scalar(f'Val/TimeShift/{key}', value, epoch)
                hard_m = val_metrics.get("hardneg_mining", {})
                for k, v in hard_m.items():
                    writer.add_scalar(f'Val/HardNeg/{k}', v, epoch)

                # Print summary of new metrics
                r1 = ret.get('g2o_recall_at_1', 0)
                r5 = ret.get('g2o_recall_at_5', 0)
                mrr = ret.get('g2o_mrr', 0)
                fr1 = ret.get('fusion_gallery_recall_at_1', 0)
                fmrr = ret.get('fusion_gallery_mrr', 0)
                auroc = cls_m.get('auroc', 0)
                auprc = cls_m.get('auprc', 0)
                align = emb.get('alignment', 0)
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
                    f'Val/BestCkpt/{args.best_ckpt_metric}',
                    float(current_selection_score),
                    epoch,
                )

                if not best_selection_eligible:
                    print(
                        "  Best-CKPT selection pending: "
                        f"{describe_best_ckpt_selection_pending(args, epoch, hard_neg_ratio)}"
                    )
                else:
                    prev_best = best_val_score
                    improved = (
                        prev_best is None
                        or current_selection_score > prev_best + args.early_stop_min_delta
                    )
                    if improved:
                        best_val_score = current_selection_score
                        best_epoch_idx = int(epoch)
                        is_best_epoch = True
                        best_val_metrics = {
                            "best_ckpt_metric": args.best_ckpt_metric,
                            "best_ckpt_score": current_selection_score,
                            "best_val_acc_total": val_metrics.get('classification', {}).get('acc_total', 0),
                            "best_val_loss": val_metrics['total'],
                            "best_epoch": epoch,
                            "best_epoch_1based": epoch + 1,
                            "best_tracking_start_epoch": best_tracking_start_epoch,
                            "best_tracking_start_epoch_1based": (
                                best_tracking_start_epoch + 1 if best_tracking_start_epoch is not None else None
                            ),
                            "val_itc_loss": val_metrics.get('itc', 0),
                            "val_cls_loss": val_metrics.get('cls', 0),
                            "val_recall_at_1": val_metrics.get('retrieval', {}).get('g2o_recall_at_1', 0),
                            "val_recall_at_5": val_metrics.get('retrieval', {}).get('g2o_recall_at_5', 0),
                            "val_mrr": val_metrics.get('retrieval', {}).get('g2o_mrr', 0),
                            "val_auroc": val_metrics.get('classification', {}).get('auroc', 0),
                            "val_auprc": val_metrics.get('classification', {}).get('auprc', 0),
                        }
                        epochs_no_improve = 0
                        best_ckpt = os.path.join(args.ckpt_path, "ALBEF", "albef_best.pth")
                        os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                            'acc_total': val_metrics.get('classification', {}).get('acc_total', 0),
                            'selection_metric': args.best_ckpt_metric,
                            'selection_score': current_selection_score,
                            'selection_eligible': bool(best_selection_eligible),
                            'best_tracking_start_epoch': best_tracking_start_epoch,
                            'loss': val_metrics['total'],
                            'args': vars(args),
                            'dataset_window_metadata': getattr(args, "_dataset_window_metadata", None),
                            'effective_input_window_metadata': getattr(args, "_effective_input_window_metadata", None),
                        }, best_ckpt)
                        print(
                            f"Saved best checkpoint ({args.best_ckpt_metric}="
                            f"{current_selection_score:.4f}, epoch={epoch+1}): {best_ckpt}"
                        )
                    else:
                        epochs_no_improve += 1
                        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                            print("Early stopping triggered.")
                            stop_early = True

                best_score_str = "N/A" if best_val_score is None else f"{best_val_score:.4f}"
                best_epoch_str = "N/A" if best_epoch_idx is None else str(best_epoch_idx + 1)
                print(
                    f"  Best-CKPT metric[{args.best_ckpt_metric}]: current={current_selection_score:.4f}, "
                    f"best={best_score_str}@epoch{best_epoch_str}, "
                    f"eligible={int(best_selection_eligible)}, is_best={int(is_best_epoch)}"
                )

        # --- OOD (out-of-distribution) validation on independent test set ---
        if ood_val_loader is not None:
            ood_metrics = evaluate(
                model,
                ood_val_loader,
                device,
                args,
                epoch,
                amp_dtype=amp_dtype,
                gw_event_time_mjd_table=ood_event_time_mjd_table,
            )
            _close_loader_dataset_handles(
                ood_val_loader,
                cache_in_memory=False,
                label="OOD DataLoader",
            )
            if ood_metrics is not None:
                writer.add_scalar('OOD/Epoch_Total_Loss', ood_metrics['total'], epoch)
                writer.add_scalar('OOD/Epoch_ITC_Loss', ood_metrics['itc'], epoch)
                writer.add_scalar('OOD/Epoch_CLS_Loss', ood_metrics['cls'], epoch)
                ood_ret = ood_metrics.get('retrieval', {})
                for k, v in ood_ret.items():
                    writer.add_scalar(f'OOD/Retrieval/{k}', v, epoch)
                ood_cls = ood_metrics.get('classification', {})
                for k in ('auroc', 'auprc', 'f1_optimal', 'ece', 'acc_total'):
                    if k in ood_cls:
                        writer.add_scalar(f'OOD/Classification/{k}', ood_cls[k], epoch)
                ood_emb = ood_metrics.get('embedding', {})
                for k, v in ood_emb.items():
                    writer.add_scalar(f'OOD/Embedding/{k}', v, epoch)
                ood_dt = ood_metrics.get("time_delta", {})
                for key, value in ood_dt.items():
                    writer.add_scalar(f'OOD/TimeShift/{key}', value, epoch)
                ood_hard_m = ood_metrics.get("hardneg_mining", {})
                for k, v in ood_hard_m.items():
                    writer.add_scalar(f'OOD/HardNeg/{k}', v, epoch)
                ood_r1 = ood_ret.get('g2o_recall_at_1', 0)
                ood_auroc = ood_cls.get('auroc', 0)
                ood_align = ood_emb.get('alignment', 0)
                print(
                    f"  OOD: R@1={ood_r1:.4f} AUROC={ood_auroc:.4f} align={ood_align:.4f}"
                )
                writer.flush()

        if not getattr(args, 'skip_epoch_checkpoints', False):
            checkpoint_path = os.path.join(args.ckpt_path, "ALBEF", f"albef_epoch_{epoch+1}.pth")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_total,
                'args': vars(args),
                'dataset_window_metadata': getattr(args, "_dataset_window_metadata", None),
                'effective_input_window_metadata': getattr(args, "_effective_input_window_metadata", None),
            }, checkpoint_path)
        
        # End-of-epoch memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        if stop_early:
            break

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
    _close_loader_dataset_handles(
        ood_val_loader,
        cache_in_memory=False,
        label="OOD DataLoader",
    )

    writer.close()

    # Write trial results JSON for HPO collection
    if best_val_metrics:
        best_val_metrics["final_epoch"] = epoch
        best_val_metrics["dataset_window_metadata"] = getattr(args, "_dataset_window_metadata", None)
        best_val_metrics["effective_input_window_metadata"] = getattr(args, "_effective_input_window_metadata", None)
        result_path = os.path.join(args.ckpt_path, "ALBEF", "trial_results.json")
        summary_dir = os.path.dirname(result_path)
        os.makedirs(summary_dir, exist_ok=True)
        for summary_name in ("trial_results.json", "train_summary.json", "best_checkpoint_summary.json"):
            summary_path = os.path.join(summary_dir, summary_name)
            with open(summary_path, "w") as f:
                json.dump(best_val_metrics, f, indent=2)
        print(f"Trial results saved to: {result_path}")


if __name__ == "__main__":
    """
    Example usage:
    python Model/ALBEF_train.py --data_path data/LSST_KN_BNS/combined_dataset.h5 \
        --epochs 2 --batch_size 32 --steps_per_epoch 10 --ckpt_path data/model/checkpoints
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="training_data.h5")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Path to independent OOD/dev HDF5 used for optional OOD monitoring.")
    parser.add_argument("--test_steps", type=int, default=25,
                        help="Number of OOD monitoring steps per epoch")
    parser.add_argument("--enable_ood_monitoring", action='store_true',
                        help="Opt-in OOD monitoring during training. Do NOT enable when test_data_path is final test set.")
    parser.add_argument("--neg_data_path", type=str, default=None)
    parser.add_argument("--neg_group", type=str, default="events/optical_data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Max norm for gradient clipping (0 to disable)")
    parser.add_argument("--lr_scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--persistent_workers", type=int, default=1)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--cache_in_memory", action='store_true')
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--val_steps_per_epoch", type=int, default=None)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                        help="Minimum improvement required on best_ckpt_metric to reset early stopping.")
    parser.add_argument("--best_ckpt_metric", type=str, default="auprc",
                        choices=[
                            "auprc", "auroc", "f1_optimal", "acc_total", "cls_composite_auprc_auroc",
                            "g2o_recall_at_1", "g2o_recall_at_5", "g2o_mrr",
                            "fusion_gallery_recall_at_1", "fusion_gallery_recall_at_5", "fusion_gallery_mrr",
                        ],
                        help="In-domain validation metric used for selecting best checkpoint. Supports classification and retrieval metrics.")
    parser.add_argument("--n_ref", type=int, default=64)
    parser.add_argument("--ref_start", type=float, default=-0.3)
    parser.add_argument("--ref_end", type=float, default=0.6)
    parser.add_argument("--ref_dim", type=int, default=64)
    parser.add_argument("--enc_dim", type=int, default=128)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--optical_curve_dim", type=int, default=None,
                        help="Physical-dual optical curve feature dimension (default: enc_dim).")
    parser.add_argument("--optical_coord_dim", type=int, default=None,
                        help="Physical-dual optical coordinate feature dimension (default: enc_dim).")
    parser.add_argument("--optical_curve_hidden_dim", type=int, default=None,
                        help="Hidden size for residual optical curve refiner; <=0 disables it.")
    parser.add_argument("--contrastive_hidden_dim", type=int, default=None,
                        help="Hidden size for physical-dual contrastive heads (default: max(input_dim, proj_dim)).")
    parser.add_argument("--fusion_attn_dim", type=int, default=None)
    parser.add_argument("--fusion_hidden_dim", type=int, default=None)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for classification loss (0 to disable)")
    parser.add_argument("--temp_init", type=float, default=0.07)
    parser.add_argument("--temp_final", type=float, default=None)
    parser.add_argument("--temp_min", type=float, default=0.01)
    parser.add_argument("--temp_max", type=float, default=100.0)
    parser.add_argument("--temp_schedule", type=str, default="learned", choices=["learned", "fixed", "cosine"])
    parser.add_argument("--time_compat_weight", type=float, default=0.6,
                        help="Weight for time-compatibility penalty on ITC logits (<=0 to disable).")
    parser.add_argument("--time_compat_tau_days", type=float, default=30.0,
                        help="Time scale tau (days) for time-compatibility penalty.")
    parser.add_argument("--time_compat_power", type=float, default=2.0,
                        help="Power for |Δt/tau|^power in time-compatibility penalty.")
    parser.add_argument("--time_compat_max_penalty", type=float, default=8.0,
                        help="Maximum absolute logit penalty for time compatibility.")
    parser.add_argument("--use_time_delta_cls_feature", action=argparse.BooleanOptionalAction, default=False,
                        help="Append normalized |delta time| to the physical_dual_hgw classifier input.")
    parser.add_argument("--time_delta_cls_scale_days", type=float, default=None,
                        help="Scale in days for the CLS time-delta feature (default: time_compat_tau_days).")
    parser.add_argument("--time_delta_cls_clip", type=float, default=10.0,
                        help="Maximum normalized |delta time| value appended to the CLS fusion head.")
    parser.add_argument("--fusion_physical_weight", type=float, default=1.0,
                        help="Fixed multiplier for the physical-consistency fused feature.")
    parser.add_argument("--fusion_spatial_weight", type=float, default=1.0,
                        help="Fixed multiplier for the spatial-consistency fused feature.")
    parser.add_argument("--mtan_snr_s0", type=float, default=3.0,
                        help="mTAN SNR compatibility threshold s0.")
    parser.add_argument("--mtan_snr_beta", type=float, default=1.0,
                        help="mTAN SNR compatibility slope beta.")
    parser.add_argument("--mtan_snr_clip_min", type=float, default=-8.0,
                        help="mTAN SNR clipping lower bound.")
    parser.add_argument("--mtan_snr_clip_max", type=float, default=20.0,
                        help="mTAN SNR clipping upper bound.")
    parser.add_argument("--mtan_snr_eps", type=float, default=1e-9,
                        help="Numerical epsilon for mTAN SNR denominator.")
    parser.add_argument("--mtan_lupt_psfflux_zp", type=float, default=None,
                        help="Optional override for luptitude psfFlux zero point used by mTAN.")
    parser.add_argument("--mtan_lupt_k", type=float, default=None,
                        help="Optional override for luptitude softening scale k used by mTAN.")
    parser.add_argument("--mtan_lupt_m5_mag", type=str, default=None,
                        help="Optional override: 6 comma-separated m5 mags in order u,g,r,i,z,Y.")
    parser.add_argument("--nonkn_cls_base_field", type=str, default="zero_time_mjd_cls_base",
                        help="Negative H5 field used as base absolute time for extra-negative dt.")
    parser.add_argument("--gw_dropout", type=float, default=0.1)
    parser.add_argument("--opt_dropout", type=float, default=0.1)
    parser.add_argument("--proj_dropout", type=float, default=0.0,
                        help="Projection head dropout to prevent ITC overfitting (default: 0.0)")
    parser.add_argument("--feature_dropout", type=float, default=0.0,
                        help="Dropout applied to encoder features before ITC/CLS heads")
    parser.add_argument("--itc_weight", type=float, default=1.0)
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--cls_pos_weight", type=float, default=1.0,
                        help="Positive class weight for CLS loss")
    parser.add_argument("--cls_neg_weight", type=float, default=1.0,
                        help="Negative class weight for CLS loss")
    parser.add_argument("--cls_extra_neg_weight", type=float, default=1.0,
                        help="Extra negative class weight for CLS loss")
    parser.add_argument("--cls_ramp_epochs", type=int, default=0,
                        help="Epochs to ramp CLS weight from 0 to cls_weight (0 to disable)")
    parser.add_argument("--retrieval_start_epoch", type=int, default=0,
                        help="Epoch to start retrieval/gallery training (0 = start from epoch 0).")
    parser.add_argument("--gallery_loss_weight", type=float, default=0.0,
                        help="Weight for batch-level fusion mini-gallery retrieval loss (0 to disable).")
    parser.add_argument("--compute_cls_metrics", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute validation pair/triplet classification metrics even when CLS loss is disabled.")
    parser.add_argument("--compute_fusion_gallery_metrics", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute validation fusion-gallery retrieval metrics even when gallery loss is disabled.")
    parser.add_argument("--fusion_gallery_metrics_start_epoch", type=int, default=0,
                        help="Epoch to start validation-only fusion-gallery metrics.")
    parser.add_argument("--gallery_score_chunk_size", type=int, default=8192,
                        help="Approximate number of flattened query-candidate pairs scored per gallery chunk.")
    parser.add_argument("--max_gallery_queries", type=int, default=0,
                        help="Max queries for fusion gallery scoring during training (0 = use all queries).")
    parser.add_argument("--gallery_include_extra_negatives", action=argparse.BooleanOptionalAction, default=True,
                        help="Include external non-KN negatives in the fusion mini-gallery loss when available.")
    parser.add_argument("--gallery_distractor_time_mode", type=str, default="actual",
                        choices=["actual", "synthetic_after_gw"],
                        help="Time-delta policy for non-matching fusion mini-gallery distractors.")
    parser.add_argument("--gallery_distractor_time_window_days", type=float, default=30.0,
                        help="Upper bound in days for synthetic_after_gw gallery distractor dt sampling.")
    parser.add_argument("--gallery_hard_neg_enable", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable top-k hard-negative mining inside the fusion mini-gallery retrieval loss.")
    parser.add_argument("--gallery_hard_neg_topk", type=int, default=32,
                        help="Number of top-scoring negative candidates to keep per query in the hard gallery NCE term.")
    parser.add_argument("--gallery_hard_neg_weight", type=float, default=0.5,
                        help="Weight of the hard-gallery NCE auxiliary loss term.")
    parser.add_argument("--gallery_hard_neg_start_after_retrieval_epochs", type=int, default=2,
                        help="Epochs after retrieval_start_epoch to begin gallery hard-negative mining.")
    parser.add_argument("--gallery_hard_neg_ramp_epochs", type=int, default=2,
                        help="Epochs to ramp gallery hard-negative weight from 0 to gallery_hard_neg_weight.")
    parser.add_argument("--itc_decay_start_epoch", type=int, default=0,
                        help="Epoch to start decaying ITC weight (ignored if itc_decay_epochs <= 0)")
    parser.add_argument("--itc_decay_epochs", type=int, default=0,
                        help="Epochs to decay ITC weight (0 to disable)")
    parser.add_argument("--itc_decay_ratio", type=float, default=0.0,
                        help="Fractional decay of ITC weight by the end of itc_decay_epochs")
    parser.add_argument("--itc_label_smoothing", type=float, default=0.0,
                        help="Label smoothing for ITC loss (0 to disable)")
    parser.add_argument("--itc_loss_type", type=str, default="infonce",
                        choices=["infonce", "supcon"],
                        help="ITC loss type: 'infonce' (original) or 'supcon' (supervised contrastive)")
    parser.add_argument("--supcon_margin", type=float, default=0.0,
                        help="Margin for SupCon loss to enforce separation between positives and negatives (default: 0.0)")
    parser.add_argument("--samples_per_gw", type=int, default=4,
                        help="Number of optical samples per GW event for SupCon (default: 4)")
    parser.add_argument("--min_lc_per_gw", type=int, default=2,
                        help="Minimum light curves required for a GW to be eligible for SupCon")
    parser.add_argument("--mask_itc", action='store_true', help="Mask same-event pairs in ITC loss")
    parser.add_argument("--hard_neg_start_epoch", type=int, default=0)
    parser.add_argument("--hard_neg_ramp_epochs", type=int, default=0,
                        help="Epochs to ramp hard negative ratio to 1.0 (0 to disable)")
    parser.add_argument("--semi_hard", action='store_true',
                        help="Use semi-hard negative sampling instead of hardest negative")
    parser.add_argument("--semi_hard_margin", type=float, default=0.2,
                        help="Relative margin for semi-hard band: sim in [(1-m)*pos_median, pos_median], m in (0,1)")
    parser.add_argument("--hardneg_time_window_days", type=str, default="30,60,120",
                        help="Comma-separated adaptive windows (days) for hard-negative mining.")
    parser.add_argument("--hardneg_min_candidates", type=int, default=4,
                        help="Minimum candidate count before accepting current hard-negative time window.")
    parser.add_argument("--hardneg_fallback_mode", type=str, default="inbatch_semihard",
                        choices=["inbatch_semihard"],
                        help="Fallback mode when adaptive window mining has no candidates.")
    parser.add_argument("--hardneg_memory_bank_enable", action='store_true',
                        help="Enable cross-batch memory bank mining for hard negatives.")
    parser.add_argument("--hardneg_memory_bank_size", type=int, default=16384,
                        help="Memory bank capacity for hard-negative optical candidates.")
    parser.add_argument("--hardneg_memory_topk", type=int, default=24,
                        help="Top-k memory candidates by similarity before time-window filtering.")
    parser.add_argument("--hardneg_memory_warmup_steps", type=int, default=200,
                        help="Warmup steps before enabling memory-bank mining.")
    parser.add_argument("--hardneg_memory_interval", type=int, default=2,
                        help="Run memory-bank hard-negative replacement every N steps.")
    parser.add_argument("--hardneg_memory_max_rows", type=int, default=256,
                        help="Maximum hard rows per step to query from memory bank.")
    parser.add_argument("--cls_start_epoch", type=int, default=0,
                        help="Epoch to start CLS training. Before this epoch, only ITC loss is used (for staged training)")
    parser.add_argument("--use_lightweight_gw", action='store_true',
                        help="Use lightweight GW encoder (~100K params) instead of ResNet-18 (~11M params) to prevent overfitting on small GW datasets")
    parser.add_argument("--dual_fusion", action='store_true',
                        help="Use dual cross-attention fusion (optical→GW + GW→optical) with per-pair credible level")
    parser.add_argument("--fusion_mode", type=str, default=None,
                        help="Fusion mode: legacy_g2o | legacy_dual | physical_dual_hgw | concat_proj. If unset, inferred from --dual_fusion.")
    parser.add_argument("--use_similarity_as_cls_input", action=argparse.BooleanOptionalAction, default=False,
                        help="Append pair ITC similarity to classifier input in physical_dual_hgw mode.")
    parser.add_argument("--use_cred_level_feature", action=argparse.BooleanOptionalAction, default=False,
                        help="Append cred_level to classifier input in physical_dual_hgw mode.")
    # GW数据增强参数
    parser.add_argument("--gw_aug_noise", type=float, default=0.05,
                        help="GW skymap augmentation noise std (default: 0.05)")
    parser.add_argument("--gw_aug_jitter", type=float, default=0.02,
                        help="GW scalar augmentation jitter ratio (default: 0.02)")
    parser.add_argument("--gw_aug_dropout", type=float, default=0.1,
                        help="GW channel dropout probability (default: 0.1)")
    # Optical data augmentation parameters
    parser.add_argument("--opt_aug_noise", type=float, default=0.0,
                        help="Optical flux noise scale relative to errors (default: 0.0)")
    parser.add_argument("--opt_aug_time_jitter", type=float, default=0.0,
                        help="Optical time jitter std (default: 0.0)")
    parser.add_argument("--opt_aug_dropout", type=float, default=0.0,
                        help="Optical observation dropout probability (default: 0.0)")
    parser.add_argument("--opt_aug_band_dropout", type=float, default=0.0,
                        help="Optical band dropout probability (default: 0.0)")
    parser.add_argument("--hpo_trial_number", type=int, default=None,
                        help="Optuna trial number (set automatically by HPO, not for manual use)")
    parser.add_argument("--skip_epoch_checkpoints", action='store_true',
                        help="Skip per-epoch checkpoint saves (only save best checkpoint). Useful for HPO.")
    parser.add_argument("--default_json_config", type=str, default=None,
                        help="Default JSON config file path. Loaded before --json_config.")
    parser.add_argument("--json_config", type=str, default=None,
                        help="JSON config file path. Values set defaults; CLI args override them.")

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
