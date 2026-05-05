"""
Comprehensive evaluation script for GW-KN ALBEF model.

Supports:
    - Batch-mode retrieval metrics (same as training eval)
    - Gallery-mode retrieval (realistic candidate pool evaluation)
    - Classification metrics (AUROC, AUPRC, F1, ECE)
    - Embedding quality metrics (alignment, uniformity)
    - Visualizations (ROC, PR curve, t-SNE, calibration, logits distribution)

New Features:
    - Support for external negative samples (non-KN transients)
    - Logits distribution plot for four pair types:
      * Positive pairs (GW, KN): matched GW-KN optical pairs
      * Optical negatives (GW, nonKN): GW paired with non-KN transients
      * GW negatives (GW_has_kn0, KN): negative-GW paired with KN optical
      * Semi-hard negatives (GW_wrong, KN): mismatched GW paired with KN optical

Usage:
    python test_evaluate.py \\
        --checkpoint ${BASE_DIR}/data/model/checkpoints/supcon_v2/ALBEF/albef_best.pth \\
        --test_data_path ${BASE_DIR}/data/LSST_KN_BNS/combined_dataset_with_neg_gw.h5 \\
        --neg_data_path ${BASE_DIR}/data/ELASTICC2_TRAIN/negative_dataset.h5 \\
        --neg_group ELASTICC2_TRAIN/optical_data \\
        --output_dir eval_results
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

# Add parent dir for local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import (
    RelationalHDF5Dataset,
    BalancedGWBatchedSampler,
    _build_dataloader,
    _read_root_time_window_attrs,
    apply_runtime_input_window_torch,
    build_effective_input_window_metadata,
)
from model import GWOpticalALBEFModel, normalize_fusion_mode, migrate_time_embed_state_dict
from metrics import (
    compute_retrieval_metrics,
    compute_classification_metrics,
    compute_embedding_metrics,
)


_BASE_DIR = os.environ.get('BASE_DIR', '/fred/oz016/bgao_kn')


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate GW-KN ALBEF model")

    # Checkpoint
    p.add_argument("--checkpoint", type=str, required=True)

    # Data source
    p.add_argument("--test_data_path", type=str,
                   default=f"{_BASE_DIR}/data/LSST_KN_BNS/combined_dataset_with_neg_gw.h5",
                   help="Independent test HDF5 file")
    p.add_argument("--neg_data_path", type=str,
                   default=f"{_BASE_DIR}/data/ELASTICC2_TRAIN/negative_dataset.h5",
                   help="Path to negative (non-KN transient) HDF5 file")
    p.add_argument("--neg_group", type=str, default="ELASTICC2_TRAIN/optical_data",
                   help="HDF5 group path for negative optical data")
    p.add_argument("--n_neg_samples", type=int, default=5000,
                   help="Number of optical negative samples to use for logits distribution")
    p.add_argument("--test_steps", type=int, default=None,
                   help="Number of test sampling steps. If None, auto-computed to match n_neg_samples")

    # Eval parameters
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--gallery_sizes", type=str, default="100,500,1000",
                   help="Comma-separated gallery sizes for gallery-mode eval")
    p.add_argument("--gallery_trials", type=int, default=10,
                   help="Number of random gallery compositions per GW event")

    # Training config (required if checkpoint lacks saved 'args')
    p.add_argument("--config", type=str, default=None,
                   help="Path to training config JSON (e.g. args/ALBEF_supcon.json)")

    # Output
    p.add_argument("--output_dir", type=str, default="eval_results")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--amp_dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help=(
            "Autocast precision for CUDA eval. "
            "'auto' uses bf16 when supported, otherwise disables AMP (fp32)."
        ),
    )
    p.add_argument("--no_plots", action="store_true",
                   help="Skip generating plots (useful on headless machines)")
    p.add_argument("--neg_time_offset_enable", action="store_true", default=None,
                   help="Enable time offsets for external non-KN optical negatives in evaluation.")
    p.add_argument("--neg_offset_dist_npz", type=str, default=None,
                   help="NPZ path for negative time-offset distribution.")
    p.add_argument("--neg_offset_dist_key", type=str, default=None,
                   help="Key in NPZ for negative offset distribution.")
    p.add_argument("--neg_offset_eval_mode", type=str, default=None,
                   help="Eval offset mode: quantile_ensemble|median|zero.")
    p.add_argument("--neg_offset_eval_quantiles", type=str, default=None,
                   help="Comma-separated quantiles for quantile_ensemble mode.")
    p.add_argument("--neg_offset_scale_days_divisor", type=float, default=None,
                   help="Convert day offsets to model time unit by dividing this value.")
    p.add_argument("--nonkn_cls_base_field", type=str, default=None,
                   help="Negative dataset field name used for classification extra-negative dt.")
    p.add_argument("--report_dt_bins", dest="report_dt_bins", action="store_true", default=None,
                   help="Report |dt| bucketed classification metrics from triplet logits.")
    p.add_argument("--no_report_dt_bins", dest="report_dt_bins", action="store_false",
                   help="Disable |dt| bucketed classification metrics.")
    p.add_argument("--dt_bin_edges", type=str, default=None,
                   help="Comma-separated |dt| edges in days, e.g. '0,30,90,180,365,730,1460,inf'.")
    p.add_argument("--report_dt_macro", dest="report_dt_macro", action="store_true", default=None,
                   help="Report macro-averaged |dt| bin metrics (bins with both pos/neg only).")
    p.add_argument("--no_report_dt_macro", dest="report_dt_macro", action="store_false",
                   help="Disable macro-averaged |dt| bin metrics.")

    return p.parse_args()


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
        raise ValueError("neg_offset_eval_quantiles produced empty list.")
    return vals


def parse_day_windows(text: str) -> List[float]:
    vals: List[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if v <= 0:
            raise ValueError(f"Invalid window value {v}; expected > 0.")
        vals.append(v)
    if not vals:
        return []
    vals = sorted(set(vals))
    return vals


def normalize_lupt_m5_mag(value) -> Tuple[float, float, float, float, float, float]:
    """Normalize mTAN luptitude m5 values from JSON/checkpoint/HDF5 sources."""
    if value is None:
        value = (23.9, 25.0, 24.7, 24.0, 23.3, 22.1)
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(np.asarray(value).reshape(-1))
    if len(parts) != 6:
        raise ValueError(f"mtan_lupt_m5_mag must contain 6 values in u,g,r,i,z,Y order; got {len(parts)}.")
    try:
        out = tuple(float(part) for part in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError("mtan_lupt_m5_mag contains non-numeric values.") from exc
    if not np.all(np.isfinite(np.asarray(out, dtype=np.float64))):
        raise ValueError("mtan_lupt_m5_mag values must be finite.")
    return out


def resolve_mtan_eval_config(test_data_path: str, saved_args: Dict[str, object]) -> Dict[str, object]:
    cfg: Dict[str, object] = {
        "mtan_snr_s0": float(saved_args.get("mtan_snr_s0", 3.0)),
        "mtan_snr_beta": float(saved_args.get("mtan_snr_beta", 1.0)),
        "mtan_snr_clip_min": float(saved_args.get("mtan_snr_clip_min", -8.0)),
        "mtan_snr_clip_max": float(saved_args.get("mtan_snr_clip_max", 20.0)),
        "mtan_snr_eps": float(saved_args.get("mtan_snr_eps", 1e-9)),
        "mtan_lupt_psfflux_zp": float(saved_args.get("mtan_lupt_psfflux_zp", 31.4)),
        "mtan_lupt_k": float(saved_args.get("mtan_lupt_k", 1.0)),
        "mtan_lupt_m5_mag": normalize_lupt_m5_mag(saved_args.get("mtan_lupt_m5_mag")),
    }
    if test_data_path and os.path.exists(test_data_path):
        try:
            with h5py.File(test_data_path, "r") as f:
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
                        cfg["mtan_lupt_m5_mag"] = tuple(float(x) for x in m5.tolist())
        except Exception as exc:
            print(f"WARNING: failed to read mTAN attrs from test H5 ({test_data_path}): {exc}")
    return cfg


def parse_dt_bin_edges(text: str) -> List[float]:
    vals: List[float] = []
    for part in str(text).split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part in ("inf", "+inf", "infinity", "+infinity"):
            vals.append(float("inf"))
        else:
            vals.append(float(part))
    if len(vals) < 2:
        raise ValueError("dt_bin_edges must contain at least two edges.")
    for i in range(1, len(vals)):
        if not (vals[i] > vals[i - 1]):
            raise ValueError("dt_bin_edges must be strictly increasing.")
    return vals


def choose_value(cli_value, ckpt_args, key, default=None):
    if cli_value is not None:
        return cli_value
    if isinstance(ckpt_args, dict) and key in ckpt_args and ckpt_args[key] is not None:
        return ckpt_args[key]
    return default


def apply_time_offsets(opt_t, opt_mask, delta_days, scale_divisor):
    valid = (opt_mask.sum(dim=-1) > 0).to(dtype=opt_t.dtype)
    shift = (delta_days.to(device=opt_t.device, dtype=opt_t.dtype) / float(scale_divisor)).unsqueeze(1)
    return opt_t + shift * valid


def compute_time_delta_days(opt_zero_time_mjd, gw_anchor_time_mjd):
    dt = opt_zero_time_mjd.to(torch.float32) - gw_anchor_time_mjd.to(torch.float32)
    dt = torch.where(torch.isfinite(dt), dt, torch.zeros_like(dt))
    return dt


def compute_g2o_similarity_with_time_compat(
    model,
    feat_g,
    feat_o,
    gw_event_time_mjd=None,
    opt_event_time_mjd=None,
):
    """
    Match training-side ITC/SupCon similarity:
      sim = feat_g @ feat_o^T / T + time_compat_bias
    """
    core_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    temperature = core_model.log_temp.exp().clamp(
        min=core_model.temp_min, max=core_model.temp_max
    )
    sim_g2o = torch.matmul(feat_g, feat_o.T) / temperature

    if gw_event_time_mjd is not None and hasattr(core_model, "_build_time_compat_bias"):
        time_bias = core_model._build_time_compat_bias(gw_event_time_mjd, opt_event_time_mjd)
        if time_bias is not None:
            sim_g2o = sim_g2o + time_bias.to(device=sim_g2o.device, dtype=sim_g2o.dtype)
    return sim_g2o


def encode_hard_negative_with_time_shift(
    model,
    *,
    opt_coords,
    opt_t,
    opt_v,
    opt_mask,
    opt_err,
    opt_ref_t,
    candidate_gw_time_mjd,
    anchor_gw_time_mjd,
    scale_divisor,
):
    if candidate_gw_time_mjd is None or anchor_gw_time_mjd is None:
        delta_days = torch.zeros((opt_t.size(0),), device=opt_t.device, dtype=torch.float32)
        shifted_opt_t = opt_t
    else:
        delta_days = compute_time_delta_days(candidate_gw_time_mjd, anchor_gw_time_mjd)
        shifted_opt_t = apply_time_offsets(opt_t, opt_mask, delta_days, scale_divisor)
    z_l_hard, h_l_hard = model.encode_optical(
        opt_coords, shifted_opt_t, opt_v, opt_ref_t, opt_mask, opt_err
    )
    return z_l_hard, h_l_hard, delta_days


def _init_dt_stats():
    return {
        "positive": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "nonfinite": 0},
        "optical_negative": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "nonfinite": 0},
        "gw_negative": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "nonfinite": 0},
        "hard_negative": {"sum": 0.0, "sum_sq": 0.0, "count": 0, "nonfinite": 0},
    }


def _accumulate_dt_stats(stats, key, dt_days):
    finite = torch.isfinite(dt_days)
    n_nonfinite = int((~finite).sum().item())
    stats[key]["nonfinite"] += n_nonfinite
    if not finite.any():
        return
    vals = dt_days[finite].to(torch.float32)
    stats[key]["sum"] += float(vals.sum().item())
    stats[key]["sum_sq"] += float((vals * vals).sum().item())
    stats[key]["count"] += int(vals.numel())


def _finalize_dt_stats(stats):
    out = {}
    for key, entry in stats.items():
        count = int(entry["count"])
        if count <= 0:
            out[key] = {"mean_days": 0.0, "std_days": 0.0, "count": 0, "nonfinite": int(entry["nonfinite"])}
            continue
        mean = entry["sum"] / float(count)
        var = max(0.0, entry["sum_sq"] / float(count) - mean * mean)
        out[key] = {
            "mean_days": float(mean),
            "std_days": float(var ** 0.5),
            "count": count,
            "nonfinite": int(entry["nonfinite"]),
        }
    return out


def _summarize_dt_distribution(dt_days, signed_quantiles=(0.05, 0.5, 0.95), abs_quantiles=(0.5, 0.9, 0.95)):
    if dt_days is None:
        return {}
    vals = torch.as_tensor(dt_days, dtype=torch.float32).flatten()
    finite = torch.isfinite(vals)
    n_nonfinite = int((~finite).sum().item())
    if not finite.any():
        return {"count": 0, "nonfinite": n_nonfinite}
    v = vals[finite]
    out = {
        "count": int(v.numel()),
        "nonfinite": n_nonfinite,
        "mean_days": float(v.mean().item()),
        "std_days": float(torch.std(v, unbiased=False).item()) if v.numel() > 1 else 0.0,
        "min_days": float(v.min().item()),
        "max_days": float(v.max().item()),
    }
    if signed_quantiles:
        q = torch.tensor(list(signed_quantiles), device=v.device, dtype=v.dtype)
        q_vals = torch.quantile(v, q)
        out["quantiles_days"] = {
            f"p{int(float(qi) * 100)}": float(q_vals[i].item()) for i, qi in enumerate(signed_quantiles)
        }
    abs_v = v.abs()
    abs_out = {
        "mean_days": float(abs_v.mean().item()),
        "std_days": float(torch.std(abs_v, unbiased=False).item()) if abs_v.numel() > 1 else 0.0,
        "min_days": float(abs_v.min().item()),
        "max_days": float(abs_v.max().item()),
    }
    if abs_quantiles:
        q = torch.tensor(list(abs_quantiles), device=abs_v.device, dtype=abs_v.dtype)
        q_vals = torch.quantile(abs_v, q)
        abs_out["quantiles_days"] = {
            f"p{int(float(qi) * 100)}": float(q_vals[i].item()) for i, qi in enumerate(abs_quantiles)
        }
    out["abs"] = abs_out
    return out


def compute_dt_bin_metrics(probs, labels, abs_dt_days, edges):
    """
    Compute classification metrics in |dt| buckets.
    Buckets are [e_i, e_{i+1}) except last bucket [e_{n-2}, e_{n-1}].
    """
    if probs is None or labels is None or abs_dt_days is None:
        return {}
    if len(probs) == 0:
        return {}
    edges = list(edges)
    out = {
        "edges_days": [float(e) if np.isfinite(e) else "inf" for e in edges],
        "bins": [],
    }
    finite = torch.isfinite(abs_dt_days) & torch.isfinite(probs) & torch.isfinite(labels.to(torch.float32))
    if not finite.any():
        return out
    p = probs[finite]
    y = labels[finite]
    d = abs_dt_days[finite]
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if np.isfinite(hi):
            mask = (d >= float(lo)) & (d < float(hi))
            label = f"[{lo},{hi})"
        else:
            mask = d >= float(lo)
            label = f"[{lo},inf)"
        n = int(mask.sum().item())
        if n <= 0:
            out["bins"].append({"range": label, "n": 0})
            continue
        n_pos = int((y[mask] == 1).sum().item())
        n_neg = int((y[mask] == 0).sum().item())
        preds = (p[mask] >= 0.5).long()
        tp = int(((preds == 1) & (y[mask] == 1)).sum().item())
        fp = int(((preds == 1) & (y[mask] == 0)).sum().item())
        tn = int(((preds == 0) & (y[mask] == 0)).sum().item())
        fn = int(((preds == 0) & (y[mask] == 1)).sum().item())
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_conf = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        entry = {
            "range": label,
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "precision": precision,
            "recall": recall,
            "f1_confusion": f1_conf,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        if n_pos > 0 and n_neg > 0:
            m = compute_classification_metrics(p[mask], y[mask])
            entry.update(
                {
                    "auroc": float(m.get("auroc", 0.0)),
                    "auprc": float(m.get("auprc", 0.0)),
                    "f1_optimal": float(m.get("f1_optimal", 0.0)),
                    "ece": float(m.get("ece", 0.0)),
                }
            )
        out["bins"].append(entry)
    return out


def compute_dt_macro_metrics(dt_bins: Dict[str, object]) -> Dict[str, float]:
    """Macro-average metrics over |dt| bins that contain both pos and neg."""
    out: Dict[str, float] = {}
    bins = dt_bins.get("bins", []) if isinstance(dt_bins, dict) else []
    if not bins:
        return out
    eligible = [b for b in bins if b.get("n_pos", 0) > 0 and b.get("n_neg", 0) > 0]
    if not eligible:
        out["bins_used"] = 0
        out["bins_total"] = len(bins)
        return out
    for key in ("auroc", "auprc", "f1_optimal", "ece"):
        vals = [float(b.get(key, 0.0)) for b in eligible if key in b]
        if vals:
            out[key] = float(np.mean(vals))
    out["bins_used"] = int(len(eligible))
    out["bins_total"] = int(len(bins))
    return out


class EvalNegativeTimeOffsetPolicy:
    def __init__(
        self,
        enabled: bool,
        dist_npz: Optional[str],
        dist_key: str,
        eval_mode: str,
        eval_quantiles: str,
        scale_divisor: float,
        seed: int,
        bank_size: int,
    ):
        self.enabled = bool(enabled)
        self.scale_divisor = float(scale_divisor)
        if self.scale_divisor <= 0:
            raise ValueError("neg_offset_scale_days_divisor must be > 0.")

        self.eval_mode = str(eval_mode).strip().lower()
        self.eval_offsets_days: List[float] = [0.0]
        self.rng = np.random.default_rng(int(seed))
        self.sample_bank: Optional[np.ndarray] = None
        self.info: Dict[str, object] = {
            "enabled": self.enabled,
            "scale_divisor": self.scale_divisor,
            "eval_mode": self.eval_mode,
            "seed": int(seed),
        }

        if not self.enabled:
            self.info["eval_offsets_days"] = [0.0]
            return

        if dist_npz is None:
            raise ValueError("neg_time_offset_enable=true requires neg_offset_dist_npz.")
        npz_path = Path(dist_npz)
        if not npz_path.exists():
            raise FileNotFoundError(f"Negative offset distribution file not found: {npz_path}")

        with np.load(npz_path, allow_pickle=False) as npz:
            if dist_key not in npz:
                raise KeyError(
                    f"neg_offset_dist_key '{dist_key}' not found in {npz_path}. "
                    f"Available keys: {list(npz.keys())}"
                )
            values = np.asarray(npz[dist_key], dtype=np.float64).reshape(-1)

        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                f"Negative offset distribution is empty after filtering NaN/Inf: "
                f"{npz_path}:{dist_key}"
            )

        samples = values.astype(np.float32, copy=False)
        bank_size = max(1, int(bank_size))
        if bank_size < int(samples.shape[0]):
            bank_idx = self.rng.integers(0, int(samples.shape[0]), size=bank_size, endpoint=False)
            self.sample_bank = samples[bank_idx].astype(np.float32, copy=False)
        else:
            self.sample_bank = samples

        if self.eval_mode == "quantile_ensemble":
            q = parse_quantiles(eval_quantiles)
            q_vals = np.quantile(values, np.asarray(q, dtype=np.float64))
            self.eval_offsets_days = [float(v) for v in q_vals.tolist()]
        elif self.eval_mode == "median":
            self.eval_offsets_days = [float(np.quantile(values, 0.5))]
        elif self.eval_mode == "zero":
            self.eval_offsets_days = [0.0]
        else:
            raise ValueError(
                f"Unsupported neg_offset_eval_mode='{self.eval_mode}'. "
                "Expected one of: quantile_ensemble|median|zero."
            )

        self.info.update(
            {
                "dist_path": str(npz_path),
                "dist_key": str(dist_key),
                "dist_count": int(values.shape[0]),
                "dist_min_days": float(np.min(values)),
                "dist_max_days": float(np.max(values)),
                "dist_mean_days": float(np.mean(values)),
                "dist_std_days": float(np.std(values)),
                "bank_size": int(self.sample_bank.shape[0]) if self.sample_bank is not None else 0,
                "eval_offsets_days": [float(v) for v in self.eval_offsets_days],
            }
        )

    def sample_offsets(self, batch_size: int) -> np.ndarray:
        if not self.enabled:
            return np.zeros((int(batch_size),), dtype=np.float32)
        if self.sample_bank is None:
            raise RuntimeError("Negative time offset sample bank is not initialized.")
        idx = self.rng.integers(0, int(self.sample_bank.shape[0]), size=int(batch_size), endpoint=False)
        return self.sample_bank[idx].astype(np.float32, copy=False)


def _autocast_context(device, amp_dtype, enabled):
    """CUDA autocast context, or no-op context when disabled."""
    if device.type != "cuda" or not enabled:
        return nullcontext()
    return autocast(device_type="cuda", dtype=amp_dtype)


def _resolve_eval_amp(amp_dtype_arg, device):
    """Resolve eval AMP dtype/enable flag from CLI and device capability."""
    if device.type != "cuda":
        return torch.float32, False

    mode = str(amp_dtype_arg).lower()
    if mode == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True
        return torch.float32, False
    if mode == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True
        print("WARNING: --amp_dtype=bf16 requested but bf16 is unsupported on this GPU. Falling back to fp32.")
        return torch.float32, False
    if mode == "fp16":
        return torch.float16, True
    if mode == "fp32":
        return torch.float32, False
    raise ValueError(f"Unsupported --amp_dtype value: {amp_dtype_arg}")


def _amp_dtype_name(dtype):
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype)


def _unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _model_requires_cred_level(model):
    base = _unwrap_model(model)
    if hasattr(base, "uses_cred_level_input"):
        return bool(base.uses_cred_level_input())
    return bool(getattr(base, "dual_fusion", False))


def load_model(args, device):
    """Load model from checkpoint, reconstructing architecture from saved args or config."""
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Priority: checkpoint saved args > config JSON file
    saved_args = ckpt.get("args", {})
    if not saved_args and args.config:
        with open(args.config) as f:
            saved_args = json.load(f)
        print(f"Loaded config from {args.config}")
    elif not saved_args:
        print("WARNING: Checkpoint has no saved args and no --config provided. "
              "Using defaults which may not match training.")

    def _get(key, default):
        return saved_args.get(key, default)

    model_args = {
        "enc_dim": _get("enc_dim", 128),
        "proj_dim": _get("proj_dim", 256),
        "n_ref": _get("n_ref", 64),
        "ref_start": _get("ref_start", -0.3),
        "ref_end": _get("ref_end", 0.6),
        "ref_dim": _get("ref_dim", 64),
        "use_lightweight_gw": _get("use_lightweight_gw", False),
        "dual_fusion": _get("dual_fusion", False),
        "fusion_mode": normalize_fusion_mode(_get("fusion_mode", None), dual_fusion=_get("dual_fusion", False)),
        "use_similarity_as_cls_input": _get("use_similarity_as_cls_input", False),
        "use_cred_level_feature": _get("use_cred_level_feature", False),
        "gw_dropout": _get("gw_dropout", 0.0),
        "opt_dropout": _get("opt_dropout", 0.0),
        "proj_dropout": _get("proj_dropout", 0.0),
        "fusion_dropout": _get("fusion_dropout", 0.0),
        "feature_dropout": _get("feature_dropout", 0.0),
        "label_smoothing": _get("label_smoothing", 0.0),
        "temp_init": _get("temp_init", 0.07),
        "temp_min": _get("temp_min", 0.01),
        "temp_max": _get("temp_max", 100.0),
        "itc_label_smoothing": _get("itc_label_smoothing", 0.0),
        "fusion_attn_dim": _get("fusion_attn_dim", None),
        "fusion_hidden_dim": _get("fusion_hidden_dim", None),
        "time_compat_weight": _get("time_compat_weight", 0.6),
        "time_compat_tau_days": _get("time_compat_tau_days", 30.0),
        "time_compat_power": _get("time_compat_power", 2.0),
        "time_compat_max_penalty": _get("time_compat_max_penalty", 8.0),
        "use_time_delta_cls_feature": _get("use_time_delta_cls_feature", False),
        "time_delta_cls_scale_days": _get(
            "time_delta_cls_scale_days",
            _get("time_compat_tau_days", 30.0),
        ),
        "time_delta_cls_clip": _get("time_delta_cls_clip", 10.0),
        "fusion_physical_weight": _get("fusion_physical_weight", 1.0),
        "fusion_spatial_weight": _get("fusion_spatial_weight", 1.0),
        "neg_offset_scale_days_divisor": _get("neg_offset_scale_days_divisor", 100.0),
        "nonkn_cls_base_field": str(
            choose_value(
                args.nonkn_cls_base_field,
                saved_args,
                "nonkn_cls_base_field",
                default="zero_time_mjd_cls_base",
            )
        ),
    }
    mtan_cfg = resolve_mtan_eval_config(args.test_data_path, saved_args)
    model_args.update(mtan_cfg)
    model_args["dual_fusion"] = model_args["fusion_mode"] != "legacy_g2o"

    model = GWOpticalALBEFModel(
        enc_dim=model_args["enc_dim"],
        proj_dim=model_args["proj_dim"],
        ref_time_dim=model_args["ref_dim"],
        use_lightweight_gw=model_args["use_lightweight_gw"],
        gw_dropout=model_args["gw_dropout"],
        opt_dropout=model_args["opt_dropout"],
        proj_dropout=model_args["proj_dropout"],
        fusion_dropout=model_args["fusion_dropout"],
        feature_dropout=model_args["feature_dropout"],
        label_smoothing=model_args["label_smoothing"],
        temp_init=model_args["temp_init"],
        temp_min=model_args["temp_min"],
        temp_max=model_args["temp_max"],
        itc_label_smoothing=model_args["itc_label_smoothing"],
        fusion_attn_dim=model_args["fusion_attn_dim"],
        fusion_hidden_dim=model_args["fusion_hidden_dim"],
        dual_fusion=model_args["dual_fusion"],
        fusion_mode=model_args["fusion_mode"],
        use_similarity_as_cls_input=model_args["use_similarity_as_cls_input"],
        use_cred_level_feature=model_args["use_cred_level_feature"],
        time_compat_weight=model_args["time_compat_weight"],
        time_compat_tau_days=model_args["time_compat_tau_days"],
        time_compat_power=model_args["time_compat_power"],
        time_compat_max_penalty=model_args["time_compat_max_penalty"],
        use_time_delta_cls_feature=model_args["use_time_delta_cls_feature"],
        time_delta_cls_scale_days=model_args["time_delta_cls_scale_days"],
        time_delta_cls_clip=model_args["time_delta_cls_clip"],
        fusion_physical_weight=model_args["fusion_physical_weight"],
        fusion_spatial_weight=model_args["fusion_spatial_weight"],
        mtan_snr_s0=float(model_args["mtan_snr_s0"]),
        mtan_snr_beta=float(model_args["mtan_snr_beta"]),
        mtan_snr_clip_min=float(model_args["mtan_snr_clip_min"]),
        mtan_snr_clip_max=float(model_args["mtan_snr_clip_max"]),
        mtan_snr_eps=float(model_args["mtan_snr_eps"]),
        mtan_lupt_psfflux_zp=float(model_args["mtan_lupt_psfflux_zp"]),
        mtan_lupt_k=float(model_args["mtan_lupt_k"]),
        mtan_lupt_m5_mag=tuple(model_args["mtan_lupt_m5_mag"]),
    )
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    state_dict = ckpt["model_state_dict"]
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict = migrate_time_embed_state_dict(state_dict)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        msg = str(exc)
        if "size mismatch" in msg:
            raise RuntimeError(
                "Checkpoint is incompatible with the current classifier head shape. "
                "Use a checkpoint trained with matching fusion/time-delta settings."
            ) from exc
        raise
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}, "
          f"loss={ckpt.get('loss', '?')}")
    return model, model_args, saved_args


def build_test_dataloader(
    args,
    saved_args,
    return_zero_time_mjd=False,
    nonkn_cls_base_field="zero_time_mjd_cls_base",
):
    """Build test dataloader from independent test HDF5.
    
    The number of steps is controlled by args.test_steps. If None,
    it is auto-computed to match args.n_neg_samples (so that the number
    of sampled light curves is comparable to the number of negative samples).
    """
    from data_loader import build_gw_to_lc_mapping

    runtime_window_start = float(choose_value(None, saved_args, "ref_start", default=-0.3))
    runtime_window_end = float(choose_value(None, saved_args, "ref_end", default=0.6))
    dataset = RelationalHDF5Dataset(
        args.test_data_path,
        negative_h5_path=args.neg_data_path,
        negative_group=args.neg_group,
        return_zero_time_mjd=bool(return_zero_time_mjd),
        nonkn_cls_base_field=str(nonkn_cls_base_field),
        opt_input_window_start=runtime_window_start,
        opt_input_window_end=runtime_window_end,
    )
    gw_to_lc = build_gw_to_lc_mapping(args.test_data_path)
    n_gw = len(gw_to_lc)
    batch_size = min(args.batch_size, n_gw)
    
    # Compute steps: either user-specified or auto-computed to match n_neg_samples
    if args.test_steps is not None:
        steps = args.test_steps
    else:
        # Auto-compute: target n_samples ~ n_neg_samples
        # Each step samples batch_size light curves
        target_samples = args.n_neg_samples
        steps = max(1, (target_samples + batch_size - 1) // batch_size)
    
    print(f"Test sampling: {steps} steps x {batch_size} batch_size = ~{steps * batch_size} samples")
    
    sampler = BalancedGWBatchedSampler(
        gw_to_lc, batch_size=batch_size,
        steps_per_epoch=steps,
    )

    loader = _build_dataloader(
        dataset, sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    return loader, dataset


def build_ref_time(batch_size, n_ref, ref_start, ref_end, device, dtype):
    ref = torch.linspace(ref_start, ref_end, n_ref, dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)


def load_negative_optical_samples(
    neg_data_path,
    neg_group,
    n_samples=5000,
    seed=42,
    require_zero_time_mjd_base=False,
    require_zero_time_mjd_cls_base=False,
    nonkn_cls_base_field="zero_time_mjd_cls_base",
    runtime_input_window_start: Optional[float] = None,
    runtime_input_window_end: Optional[float] = None,
):
    """Load negative optical samples (non-KN transients) from external HDF5 file.
    
    Args:
        neg_data_path: Path to negative dataset HDF5 file
        neg_group: HDF5 group path for optical data
        n_samples: Number of samples to load
        seed: Random seed for sampling
        
    Returns:
        dict with keys: 'values', 'times', 'masks', 'errors', 'coordinates', 'types'
    """
    if neg_data_path is None or not os.path.exists(neg_data_path):
        print(f"WARNING: Negative data path not found: {neg_data_path}")
        return None
    
    rng = np.random.default_rng(seed)
    
    with h5py.File(neg_data_path, 'r', rdcc_nbytes=4 * 1024 * 1024 * 1024) as f:
        grp = f[neg_group]
        total_samples = grp['values'].shape[0]
        has_zero_time_mjd_base = 'zero_time_mjd_base' in grp
        has_zero_time_mjd_cls_base = str(nonkn_cls_base_field) in grp
        if require_zero_time_mjd_base and not has_zero_time_mjd_base:
            raise KeyError(
                f"Evaluation requires '{neg_group}/zero_time_mjd_base' in {neg_data_path}"
            )
        if require_zero_time_mjd_cls_base and not has_zero_time_mjd_cls_base:
            raise KeyError(
                f"Evaluation requires '{neg_group}/{nonkn_cls_base_field}' in {neg_data_path}"
            )
        
        # Randomly sample indices, unless n_samples<=0 / None requests the full negative pool.
        if n_samples is None:
            n_samples = total_samples
        else:
            n_samples = int(n_samples)
            if n_samples <= 0:
                n_samples = total_samples
        if n_samples >= total_samples:
            sample_indices = np.arange(total_samples, dtype=np.int64)
            n_samples = total_samples
            print(f"Loading all {n_samples} negative optical samples from {neg_data_path}")
        else:
            sample_indices = rng.choice(total_samples, size=n_samples, replace=False)
            sample_indices = np.sort(sample_indices)  # Sort for efficient HDF5 access
            print(f"Loading {n_samples} negative optical samples from {neg_data_path}")
        
        # Load data.  When sampling a random subset, sequential full-array
        # reads are faster than chunk-by-chunk fancy indexing because the
        # random indices scatter across virtually every chunk anyway.
        if n_samples < total_samples:
            full_vals = np.asarray(grp['values'][:])
            neg_data = {'values': torch.from_numpy(full_vals[sample_indices])}
            del full_vals
            full_times = np.asarray(grp['times'][:])
            neg_data['times'] = torch.from_numpy(full_times[sample_indices])
            del full_times
            full_masks = np.asarray(grp['masks'][:])
            neg_data['masks'] = torch.from_numpy(full_masks[sample_indices])
            del full_masks
            full_errs = np.asarray(grp['errors'][:])
            neg_data['errors'] = torch.from_numpy(full_errs[sample_indices])
            del full_errs
            neg_data['coordinates'] = torch.from_numpy(grp['coordinates'][:][sample_indices])
        else:
            neg_data = {
                'values': torch.from_numpy(grp['values'][:]),
                'times': torch.from_numpy(grp['times'][:]),
                'masks': torch.from_numpy(grp['masks'][:]),
                'errors': torch.from_numpy(grp['errors'][:]),
                'coordinates': torch.from_numpy(grp['coordinates'][:]),
            }
        if has_zero_time_mjd_base:
            neg_data['zero_time_mjd_base'] = torch.from_numpy(grp['zero_time_mjd_base'][sample_indices])
        if has_zero_time_mjd_cls_base:
            neg_data['zero_time_mjd_cls_base'] = torch.from_numpy(grp[str(nonkn_cls_base_field)][sample_indices])
        
        # Load types if available (bulk read to avoid per-element HDF5 access)
        if 'types' in grp:
            raw_types = np.asarray(grp['types'][sample_indices])
            neg_data['types'] = [t.decode() if isinstance(t, bytes) else t for t in raw_types]

    if runtime_input_window_start is not None and runtime_input_window_end is not None:
        cropped_time, cropped_val, cropped_mask, cropped_err, _ = apply_runtime_input_window_torch(
            neg_data['times'],
            neg_data['values'],
            neg_data['masks'],
            neg_data['errors'],
            None,
            window_start=float(runtime_input_window_start),
            window_end=float(runtime_input_window_end),
        )
        neg_data['times'] = cropped_time
        neg_data['values'] = cropped_val
        neg_data['masks'] = cropped_mask
        neg_data['errors'] = cropped_err

    return neg_data


class DtMatchSampler:
    """Sample target |dt| values based on positive-pair distribution."""
    def __init__(
        self,
        strategy: str,
        window_days: float,
        quantiles: Optional[List[float]],
        target: str,
        apply_to: List[str],
        seed: int = 42,
    ):
        self.strategy = str(strategy).strip().lower()
        self.window_days = float(window_days)
        self.quantiles = list(quantiles or [])
        self.target = str(target).strip().lower()
        self.apply_to = list(apply_to or [])
        self.enabled = self.strategy in ("window", "quantile")
        self._rng = np.random.default_rng(int(seed))

    def sample_targets(self, dt_pos: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.enabled or dt_pos is None:
            return None, None
        if dt_pos.numel() == 0:
            return None, None
        dt_pos_f = dt_pos.to(torch.float32)
        abs_pos = dt_pos_f.abs()
        if self.strategy == "quantile":
            if not self.quantiles:
                return None, None
            abs_cpu = abs_pos.detach().cpu().numpy()
            qvals = np.quantile(abs_cpu, np.asarray(self.quantiles, dtype=np.float64))
            if qvals.ndim == 0:
                qvals = np.array([float(qvals)], dtype=np.float64)
            idx = self._rng.integers(0, len(qvals), size=abs_pos.numel())
            target_abs = torch.as_tensor(qvals[idx], device=dt_pos.device, dtype=torch.float32)
        else:
            target_abs = abs_pos
        sign = torch.sign(dt_pos_f)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        target_signed = target_abs * sign
        return target_abs, target_signed


class NegTimeIndex:
    """Index of negative-GW times for fast window sampling around target |dt|."""
    def __init__(self, times: torch.Tensor, indices: np.ndarray):
        self.enabled = False
        self._sorted_times = None
        self._sorted_to_orig = None
        if times is None or indices is None or len(indices) == 0:
            return
        t = np.asarray(times.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if t.shape[0] <= 0 or idx.shape[0] <= 0:
            return
        valid = np.isfinite(t)
        if not valid.any():
            return
        t = t[valid]
        idx = idx[valid]
        order = np.argsort(t, kind="mergesort")
        self._sorted_times = t[order]
        self._sorted_to_orig = idx[order]
        self.enabled = True

    def _uniform(self) -> int:
        total = len(self._sorted_to_orig) if self._sorted_to_orig is not None else 0
        if total <= 0:
            return 0
        return int(self._sorted_to_orig[np.random.randint(0, total)])

    def sample_one(self, anchor_time: float, target_abs: float, window_days: float) -> Tuple[int, bool]:
        if (not self.enabled) or (not np.isfinite(anchor_time)) or (not np.isfinite(target_abs)):
            return self._uniform(), False
        if window_days <= 0:
            return self._uniform(), False
        times = self._sorted_times
        sorted_to_orig = self._sorted_to_orig
        t_abs = float(abs(target_abs))
        centers = [anchor_time - t_abs, anchor_time + t_abs]
        if t_abs <= 0:
            centers = [anchor_time]
        ranges = []
        for c in centers:
            l = int(np.searchsorted(times, c - window_days, side="left"))
            r = int(np.searchsorted(times, c + window_days, side="right"))
            if r > l:
                ranges.append((l, r))
        if not ranges:
            return self._uniform(), False
        counts = [r - l for l, r in ranges]
        total = int(sum(counts))
        draw = int(np.random.randint(0, total))
        for (l, r), cnt in zip(ranges, counts):
            if draw < cnt:
                return int(sorted_to_orig[l + draw]), True
            draw -= cnt
        return self._uniform(), False

    def sample_batch(self, anchor_times: torch.Tensor, target_abs: torch.Tensor, window_days: float) -> Tuple[np.ndarray, int, int]:
        anchor_np = np.asarray(anchor_times.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        target_np = np.asarray(target_abs.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        out = np.empty((anchor_np.shape[0],), dtype=np.int64)
        used = 0
        fallback = 0
        for i, (a, t) in enumerate(zip(anchor_np, target_np)):
            idx, ok = self.sample_one(float(a), float(t), float(window_days))
            out[i] = int(idx)
            if ok:
                used += 1
            else:
                fallback += 1
        return out, int(used), int(fallback)


def load_negative_gw_indices(test_data_path):
    """Load GW indices where has_kn == 0 from test HDF5."""
    if test_data_path is None or not os.path.exists(test_data_path):
        print(f"WARNING: Test data path not found for GW negatives: {test_data_path}")
        return np.array([], dtype=np.int64)

    with h5py.File(test_data_path, 'r') as f:
        if 'events/gw_data/has_kn' not in f:
            print("WARNING: 'events/gw_data/has_kn' not found. GW negatives will be disabled.")
            return np.array([], dtype=np.int64)
        has_kn = f['events/gw_data/has_kn'][:]

    neg_gw_indices = np.where(has_kn == 0)[0].astype(np.int64)
    if neg_gw_indices.size == 0:
        print("WARNING: No has_kn=0 GW events found. GW negatives will be disabled.")
    else:
        print(f"Loaded {neg_gw_indices.size} GW negatives (has_kn=0).")
    return neg_gw_indices


def load_gw_event_time_mjd_table(test_data_path, device, required=False):
    ds_path = "events/gw_data/event_time_mjd"
    if test_data_path is None or not os.path.exists(test_data_path):
        if required:
            raise FileNotFoundError(f"test_data_path not found for event_time_mjd: {test_data_path}")
        print(f"WARNING: Test data path not found for event_time_mjd: {test_data_path}")
        return None
    with h5py.File(test_data_path, "r") as f:
        n_gw = int(f["events/gw_data/scalars"].shape[0])
        if ds_path not in f:
            if required:
                raise KeyError(f"Required field missing: {ds_path} in {test_data_path}")
            print(f"WARNING: '{ds_path}' missing in {test_data_path}.")
            return None
        arr = np.asarray(f[ds_path][:], dtype=np.float32)
    if arr.shape[0] != n_gw:
        if required:
            raise ValueError(
                f"event_time_mjd length mismatch in {test_data_path}: got {arr.shape[0]}, expected {n_gw}"
            )
        print(
            f"WARNING: event_time_mjd length mismatch in {test_data_path}: got {arr.shape[0]}, expected {n_gw}"
        )
        return None
    return torch.from_numpy(arr).to(device=device)


def _normalize_source_type(raw_value):
    """Normalize source labels from HDF5 values to stable lowercase strings."""
    if isinstance(raw_value, (bytes, np.bytes_)):
        raw_value = raw_value.decode("utf-8", errors="ignore")
    label = str(raw_value).strip().lower()
    if label in ("", "none", "nan"):
        return "unknown"
    return label


def _lookup_source_type(gw_idx, gw_source_types):
    """Safe source lookup by GW index."""
    if gw_source_types is None:
        return "unknown"
    idx = int(gw_idx)
    if idx < 0 or idx >= len(gw_source_types):
        return "unknown"
    return gw_source_types[idx]


def load_gw_source_types(test_data_path):
    """Load per-GW source_type labels from test HDF5 for by-source metrics."""
    if test_data_path is None or not os.path.exists(test_data_path):
        print(f"WARNING: Test data path not found for source_type: {test_data_path}")
        return None

    with h5py.File(test_data_path, 'r') as f:
        if 'events/gw_data/source_type' not in f:
            print("WARNING: 'events/gw_data/source_type' not found. classification_by_source will be skipped.")
            return None
        raw_source_types = f['events/gw_data/source_type'][:]

    source_types = [_normalize_source_type(v) for v in raw_source_types]
    uniq, counts = np.unique(np.asarray(source_types, dtype=object), return_counts=True)
    source_summary = ", ".join(f"{u}={int(c)}" for u, c in zip(uniq, counts))
    print(f"Loaded GW source_type labels: {source_summary}")
    return source_types


def _is_dual_fusion_model(model):
    """Return whether current model uses dual fusion branch."""
    if hasattr(model, "_orig_mod"):
        return bool(getattr(model._orig_mod, "dual_fusion", False))
    return bool(getattr(model, "dual_fusion", False))


def _count_non_finite(tensor):
    """Return number of NaN/Inf entries in a tensor."""
    return int((~torch.isfinite(tensor)).sum().item())


def _collect_non_finite(named_tensors):
    """Collect non-finite counts for a mapping of {name: tensor}."""
    bad = {}
    for name, tensor in named_tensors.items():
        n_bad = _count_non_finite(tensor)
        if n_bad > 0:
            bad[name] = n_bad
    return bad


def _filter_finite_embedding_rows(feat_g, feat_o, gw_indices):
    """Drop rows where GW/optical embeddings contain NaN/Inf."""
    finite_mask = torch.isfinite(feat_g).all(dim=1) & torch.isfinite(feat_o).all(dim=1)
    n_drop = int((~finite_mask).sum().item())
    if n_drop == 0:
        return feat_g, feat_o, gw_indices, n_drop
    return feat_g[finite_mask], feat_o[finite_mask], gw_indices[finite_mask], n_drop


@torch.no_grad()
def extract_all_embeddings(model, loader, device, model_args, gw_source_types=None,
                           amp_dtype=torch.float32, amp_enabled=False,
                           gw_event_time_mjd_table=None):
    """Run full forward pass, collecting embeddings and predictions.

    For classification evaluation, both matched (positive) and mismatched
    (negative) GW-optical pairs are created.  Negatives are formed by
    shifting the optical sequence within each batch so that each GW is
    paired with an unrelated optical LC.
    """
    all_feat_g = []
    all_feat_o = []
    all_gw_indices = []
    all_logits = []     # pos + neg logits
    all_labels = []     # 1 for matched, 0 for mismatched
    all_sim_blocks = []
    all_gw_idx_blocks = []
    all_h_l_cls = []    # optical sequence features for fusion logits
    all_z_l_cls = []    # optical cls token (dual fusion)
    all_opt_coords = [] # optical coordinates (dual fusion credible level)
    all_opt_t_raw = []  # raw optical time axis (for gallery time-shift re-encoding)
    all_opt_v_raw = []  # raw optical values
    all_opt_mask_raw = []  # raw optical masks
    all_opt_err_raw = []  # raw optical errors
    all_gw_source_labels = []

    ref_time_cache = None
    n_ref = model_args["n_ref"]
    ref_start = model_args["ref_start"]
    ref_end = model_args["ref_end"]
    dual = _is_dual_fusion_model(model)
    need_cred = _model_requires_cred_level(model)

    for batch_data in tqdm(loader, desc="Extracting embeddings"):
        # Unpack
        if len(batch_data) >= 16:
            (
                gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords,
                gw_indices, neg_t, neg_v, neg_mask, neg_err, neg_coords,
                opt_zero_time_mjd_base, _neg_zero_time_mjd_base, _neg_zero_time_mjd_cls_base,
            ) = batch_data[:16]
        elif len(batch_data) == 15:
            (
                gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords,
                gw_indices, neg_t, neg_v, neg_mask, neg_err, neg_coords,
                opt_zero_time_mjd_base, _neg_zero_time_mjd_base,
            ) = batch_data
        elif len(batch_data) == 13:
            (
                gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords,
                gw_indices, neg_t, neg_v, neg_mask, neg_err, neg_coords,
            ) = batch_data
            opt_zero_time_mjd_base = None
        else:
            if len(batch_data) >= 9:
                gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices, opt_zero_time_mjd_base = batch_data[:9]
            else:
                gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data[:8]
                opt_zero_time_mjd_base = None

        gw_s = gw_s.to(device)
        gw_m = gw_m.to(device)
        opt_t = opt_t.to(device)
        opt_v = opt_v.to(device)
        opt_mask = opt_mask.to(device)
        opt_err = opt_err.to(device)
        opt_coords = opt_coords.to(device)
        gw_indices = gw_indices.to(device).long()
        batch_event_time_mjd = None
        if gw_event_time_mjd_table is not None:
            batch_event_time_mjd = gw_event_time_mjd_table[gw_indices]
        batch_sources = None
        if gw_source_types is not None:
            gw_indices_cpu = gw_indices.detach().cpu().numpy().tolist()
            batch_sources = [_lookup_source_type(gw_idx, gw_source_types) for gw_idx in gw_indices_cpu]

        batch_size = gw_s.size(0)
        if (ref_time_cache is None or ref_time_cache.shape[0] != batch_size
                or ref_time_cache.dtype != opt_t.dtype):
            ref_time_cache = build_ref_time(
                batch_size, n_ref, ref_start, ref_end, device, opt_t.dtype
            )
        opt_ref_t = ref_time_cache

        def _forward_pass(use_amp):
            with _autocast_context(device, amp_dtype, enabled=(use_amp and amp_enabled)):
                g, z_l, h_l, H_gw = model.encode(
                    gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err
                )
                feat_g, feat_o = model.get_contrastive_embeddings(g, z_l)

                # Compute credible level if dual fusion
                if need_cred:
                    from ALBEF_train import compute_credible_level
                    cred_level = compute_credible_level(gw_m, opt_coords)
                else:
                    cred_level = None

                # Positive pairs (matched GW-optical)
                dt_pos = torch.zeros((batch_size,), device=device, dtype=torch.float32)
                logits_pos = model.fusion_logits(
                    g, h_l, z_l=z_l, H_gw=H_gw, cred_level=cred_level,
                    gw_s=gw_s, gw_m=gw_m, opt_coords=opt_coords,
                    dt_days=dt_pos,
                )

                # Negative pairs (shift optical by 1 so each GW pairs with wrong optical)
                shift = 1
                h_l_neg = torch.roll(h_l, shifts=shift, dims=0)
                z_l_neg = torch.roll(z_l, shifts=shift, dims=0)
                opt_coords_neg = torch.roll(opt_coords, shifts=shift, dims=0)
                # Recompute credible level for mismatched pair (current GW + rolled optical coords)
                if need_cred:
                    cred_level_neg = compute_credible_level(gw_m, opt_coords_neg)
                else:
                    cred_level_neg = None
                logits_neg = model.fusion_logits(
                    g, h_l_neg, z_l=z_l_neg, H_gw=H_gw, cred_level=cred_level_neg,
                    gw_s=gw_s, gw_m=gw_m, opt_coords=opt_coords_neg,
                    dt_days=(
                        compute_time_delta_days(
                            torch.roll(batch_event_time_mjd, shifts=shift, dims=0),
                            batch_event_time_mjd,
                        )
                        if batch_event_time_mjd is not None
                        else None
                    ),
                )

                # Similarity matrix for retrieval (aligned with training ITC/SupCon path)
                sim_g2o = compute_g2o_similarity_with_time_compat(
                    model,
                    feat_g,
                    feat_o,
                    gw_event_time_mjd=batch_event_time_mjd,
                    opt_event_time_mjd=batch_event_time_mjd,
                )
            return g, z_l, h_l, feat_g, feat_o, logits_pos, logits_neg, sim_g2o

        g, z_l, h_l, feat_g, feat_o, logits_pos, logits_neg, sim_g2o = _forward_pass(use_amp=True)
        non_finite = _collect_non_finite({
            "feat_g": feat_g,
            "feat_o": feat_o,
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "sim_g2o": sim_g2o,
        })

        # AMP on very large batches can occasionally produce non-finite values.
        # Retry once in FP32 for this batch before continuing.
        if non_finite and device.type == "cuda" and amp_enabled:
            bad_str = ", ".join(f"{k}={v}" for k, v in non_finite.items())
            print(
                "WARNING: Non-finite values detected in AMP forward pass "
                f"(batch_size={batch_size}): {bad_str}. Retrying this batch in FP32."
            )
            g, z_l, h_l, feat_g, feat_o, logits_pos, logits_neg, sim_g2o = _forward_pass(use_amp=False)
            non_finite = _collect_non_finite({
                "feat_g": feat_g,
                "feat_o": feat_o,
                "logits_pos": logits_pos,
                "logits_neg": logits_neg,
                "sim_g2o": sim_g2o,
            })
            if non_finite:
                bad_str = ", ".join(f"{k}={v}" for k, v in non_finite.items())
                print(
                    "WARNING: Non-finite values remain after FP32 retry: "
                    f"{bad_str}. Downstream metrics/plots will filter invalid rows."
                )

        all_feat_g.append(feat_g.float().cpu())
        all_feat_o.append(feat_o.float().cpu())
        all_gw_indices.append(gw_indices.cpu())
        # Concatenate pos and neg logits + labels
        all_logits.append(logits_pos.float().cpu())
        all_logits.append(logits_neg.float().cpu())
        all_labels.append(torch.ones(batch_size, dtype=torch.long))
        all_labels.append(torch.zeros(batch_size, dtype=torch.long))
        all_sim_blocks.append(sim_g2o.float().cpu())
        all_gw_idx_blocks.append(gw_indices.cpu())
        all_h_l_cls.append(h_l.float().cpu())
        all_z_l_cls.append(z_l.float().cpu())
        all_opt_coords.append(opt_coords.float().cpu())
        all_opt_t_raw.append(opt_t.float().cpu())
        all_opt_v_raw.append(opt_v.float().cpu())
        all_opt_mask_raw.append(opt_mask.float().cpu())
        all_opt_err_raw.append(opt_err.float().cpu())
        if batch_sources is not None:
            all_gw_source_labels.extend(batch_sources)

    return {
        "feat_g": torch.cat(all_feat_g),
        "feat_o": torch.cat(all_feat_o),
        "gw_indices": torch.cat(all_gw_indices),
        "logits": torch.cat(all_logits),
        "labels": torch.cat(all_labels),
        "sim_blocks": all_sim_blocks,
        "gw_idx_blocks": all_gw_idx_blocks,
        "h_l_cls": torch.cat(all_h_l_cls),
        "z_l_cls": torch.cat(all_z_l_cls),
        "opt_coords": torch.cat(all_opt_coords),
        "opt_t_raw": torch.cat(all_opt_t_raw),
        "opt_v_raw": torch.cat(all_opt_v_raw),
        "opt_mask_raw": torch.cat(all_opt_mask_raw),
        "opt_err_raw": torch.cat(all_opt_err_raw),
        "dual_fusion": dual,
        "gw_source_labels": np.asarray(all_gw_source_labels, dtype=object) if all_gw_source_labels else None,
    }


@torch.no_grad()
def extract_triplet_logits(model, loader, device, model_args, neg_optical_data,
                           neg_gw_indices,
                           gw_source_types=None,
                           shuffle_gw=False, shuffle_seed=42,
                           amp_dtype=torch.float32, amp_enabled=False,
                           neg_offset_policy: Optional[EvalNegativeTimeOffsetPolicy] = None,
                           gw_event_time_mjd_table=None,
                           hardneg_windows_days: Optional[List[float]] = None,
                           hardneg_min_candidates: int = 1,
                           hardneg_semi_hard: bool = True,
                           hardneg_semi_hard_margin: float = 0.2,
                           hardneg_fallback_mode: str = "inbatch_semihard"):
    """Extract logits for four types of sample pairs.
    
    1. Positive pairs (GW, KN): matched GW-KN optical pairs
    2. Optical negatives (GW, nonKN): GW paired with non-KN transients
    3. GW negatives (GW_has_kn0, KN): negative-GW paired with KN optical
    4. Semi-hard negatives (GW_wrong, KN): mismatched GW paired with KN optical
       - Uses similarity-based semi-hard negative mining: for each optical,
         select the most similar wrong GW that is still less similar than
         the correct GW (below positive similarity).
    
    Args:
        model: Trained ALBEF model
        loader: DataLoader for test data
        device: Torch device
        model_args: Model configuration dict
        neg_optical_data: Dict with negative optical samples from external file
        neg_gw_indices: Numpy array of has_kn=0 GW indices from test HDF5
        gw_source_types: Optional list of source_type labels indexed by GW ID
        shuffle_gw: If True, randomly shuffle GW within each batch (ablation test)
        shuffle_seed: Random seed for GW shuffling
        hardneg_semi_hard: Whether to select semi-hard negatives
        hardneg_semi_hard_margin: Semi-hard margin
        hardneg_fallback_mode: Fallback mode when windowed mining has insufficient candidates
        
    Returns:
        dict with logits for all pair types and source labels
    """
    logits_positive = []    # (GW, matched KN)
    logits_optical_neg = [] # (GW, non-KN transient)
    logits_gw_neg = []      # (GW_has_kn0, KN)
    logits_hard_neg = []    # (wrong GW, KN) - semi-hard
    cred_positive = []
    cred_optical_neg = []
    cred_gw_neg = []
    cred_hard_neg = []
    optical_neg_types = []  # Type of non-KN transient
    source_positive = []    # source_type for positive pairs
    source_optical_neg = [] # source_type for optical negatives
    source_gw_neg = []      # source_type for GW negatives
    source_hard_neg = []    # source_type for semi-hard negatives
    dt_stats = _init_dt_stats()
    dt_positive_all = []
    dt_optical_all = []
    dt_gw_all = []
    dt_hard_all = []
    
    n_ref = model_args["n_ref"]
    ref_start = model_args["ref_start"]
    ref_end = model_args["ref_end"]
    need_cred = _model_requires_cred_level(model)

    # RNG for GW shuffling
    if shuffle_gw:
        shuffle_rng = torch.Generator(device='cpu')
        shuffle_rng.manual_seed(shuffle_seed)

    # Pre-process negative optical data if available
    has_neg_optical = neg_optical_data is not None
    has_neg_gw = neg_gw_indices is not None and len(neg_gw_indices) > 0
    neg_idx = 0
    gw_neg_rng = np.random.default_rng(shuffle_seed)
    
    if has_neg_optical:
        neg_values = neg_optical_data['values'].to(device)
        neg_times = neg_optical_data['times'].to(device)
        neg_masks = neg_optical_data['masks'].to(device)
        neg_errors = neg_optical_data['errors'].to(device)
        neg_coords = neg_optical_data['coordinates'].to(device)
        neg_zero_time_mjd_cls_base = None
        if 'zero_time_mjd_cls_base' in neg_optical_data:
            neg_zero_time_mjd_cls_base = neg_optical_data['zero_time_mjd_cls_base'].to(device).to(torch.float32)
        neg_types_list = neg_optical_data.get('types', ['unknown'] * len(neg_values))
        total_neg = len(neg_values)

    gw_file = None
    gw_scalars_ds = None
    gw_skymaps_ds = None
    if has_neg_gw:
        dataset_obj = getattr(loader, "dataset", None)
        test_h5_path = getattr(dataset_obj, "h5_path", None)
        if test_h5_path is None:
            print("WARNING: Cannot locate test HDF5 path from loader.dataset. GW negatives disabled.")
            has_neg_gw = False
        elif not os.path.exists(test_h5_path):
            print(f"WARNING: Test HDF5 path not found for GW negatives: {test_h5_path}")
            has_neg_gw = False
        else:
            neg_gw_indices = np.asarray(neg_gw_indices, dtype=np.int64)
            gw_file = h5py.File(test_h5_path, 'r')
            gw_scalars_ds = gw_file['events/gw_data/scalars']
            gw_skymaps_ds = gw_file['events/gw_data/skymaps']

    try:
        for batch_data in tqdm(loader, desc="Extracting triplet logits"):
            # Unpack - get positive KN data
            if len(batch_data) >= 16:
                (
                    gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                    _neg_t, _neg_v, _neg_mask, _neg_err, _neg_coords,
                    opt_zero_time_mjd_base, _neg_zero_time_mjd_base, _neg_zero_time_mjd_cls_base,
                ) = batch_data[:16]
            elif len(batch_data) >= 15:
                (
                    gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                    _neg_t, _neg_v, _neg_mask, _neg_err, _neg_coords,
                    opt_zero_time_mjd_base, _neg_zero_time_mjd_base,
                ) = batch_data[:15]
            elif len(batch_data) >= 13:
                (
                    gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices,
                    _neg_t, _neg_v, _neg_mask, _neg_err, _neg_coords,
                ) = batch_data[:13]
                opt_zero_time_mjd_base = None
            elif len(batch_data) >= 9:
                gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices, opt_zero_time_mjd_base = batch_data[:9]
            elif len(batch_data) >= 8:
                gw_s, gw_m, opt_t, opt_v, opt_mask, opt_err, opt_coords, gw_indices = batch_data[:8]
                opt_zero_time_mjd_base = None
            else:
                continue

            gw_s = gw_s.to(device)
            gw_m = gw_m.to(device)
            opt_t = opt_t.to(device)
            opt_v = opt_v.to(device)
            opt_mask = opt_mask.to(device)
            opt_err = opt_err.to(device)
            opt_coords = opt_coords.to(device)
            gw_indices_dev = gw_indices.to(device).long()
            batch_event_time_mjd = None
            if gw_event_time_mjd_table is not None:
                batch_event_time_mjd = gw_event_time_mjd_table[gw_indices_dev]
            if opt_zero_time_mjd_base is not None:
                opt_zero_time_mjd_base = opt_zero_time_mjd_base.to(device).to(torch.float32)
            gw_indices_cpu = gw_indices_dev.detach().cpu().numpy().tolist()
            if gw_source_types is not None:
                batch_sources = [_lookup_source_type(gw_idx, gw_source_types) for gw_idx in gw_indices_cpu]
            else:
                batch_sources = None

            batch_size = gw_s.size(0)

            # GW shuffle ablation: randomly permute GW within batch
            if shuffle_gw:
                perm = torch.randperm(batch_size, generator=shuffle_rng).to(gw_s.device)
                gw_s = gw_s[perm]
                gw_m = gw_m[perm]
                gw_indices_anchor = gw_indices_dev[perm]
                if batch_event_time_mjd is not None:
                    batch_event_time_mjd_anchor = batch_event_time_mjd[perm]
                else:
                    batch_event_time_mjd_anchor = None
            else:
                gw_indices_anchor = gw_indices_dev
                batch_event_time_mjd_anchor = batch_event_time_mjd

            ref_time = build_ref_time(batch_size, n_ref, ref_start, ref_end, device, opt_t.dtype)

            with _autocast_context(device, amp_dtype, enabled=amp_enabled):
                # Encode GW and KN optical
                g, z_l, h_l, H_gw = model.encode(
                    gw_s, gw_m, opt_coords, opt_t, opt_v, ref_time, opt_mask, opt_err
                )

                if need_cred:
                    from ALBEF_train import compute_credible_level
                    _cred = compute_credible_level(gw_m, opt_coords)
                else:
                    _cred = None

                # 1. Positive pairs: matched GW-KN
                dt_pos = None
                if opt_zero_time_mjd_base is not None and batch_event_time_mjd_anchor is not None:
                    dt_pos = compute_time_delta_days(opt_zero_time_mjd_base, batch_event_time_mjd_anchor)
                    _accumulate_dt_stats(dt_stats, "positive", dt_pos)
                    dt_positive_all.append(dt_pos.detach().cpu())
                logits_pos = model.fusion_logits(
                    g, h_l, z_l=z_l, H_gw=H_gw, cred_level=_cred,
                    gw_s=gw_s, gw_m=gw_m, opt_coords=opt_coords,
                    dt_days=dt_pos,
                )
                logits_positive.append(logits_pos.float().cpu())
                if _cred is not None:
                    cred_positive.append(_cred.detach().float().cpu())
                if batch_sources is not None:
                    source_positive.extend(batch_sources)

                # 2. Semi-hard negatives: use similarity-based semi-hard negative mining
                feat_g, feat_o = model.get_contrastive_embeddings(g, z_l)
                sim_g2o = compute_g2o_similarity_with_time_compat(
                    model,
                    feat_g,
                    feat_o,
                    gw_event_time_mjd=batch_event_time_mjd_anchor,
                    opt_event_time_mjd=batch_event_time_mjd_anchor,
                )

                if batch_event_time_mjd_anchor is not None:
                    from ALBEF_train import sample_inbatch_hard_negatives_with_time
                    hard_window_days = (
                        [float(v) for v in hardneg_windows_days]
                        if hardneg_windows_days
                        else [30.0, 60.0, 120.0]
                    )
                    semi_hard_opt_idx, _, _ = sample_inbatch_hard_negatives_with_time(
                        sim_g2o=sim_g2o,
                        gw_indices=gw_indices_anchor,
                        batch_event_time_mjd=batch_event_time_mjd_anchor,
                        window_days=hard_window_days,
                        min_candidates=int(hardneg_min_candidates),
                        semi_hard=bool(hardneg_semi_hard),
                        semi_hard_margin=float(hardneg_semi_hard_margin),
                        fallback_mode=str(hardneg_fallback_mode),
                    )
                else:
                    if hardneg_semi_hard:
                        semi_hard_opt_idx = model.sample_semi_hard_negatives(
                            sim_g2o, gw_indices_anchor, margin=float(hardneg_semi_hard_margin)
                        )
                    else:
                        semi_hard_opt_idx = model.sample_hard_negatives(sim_g2o, gw_indices_anchor)

                coords_hard = opt_coords[semi_hard_opt_idx]
                candidate_gw_time_hard = (
                    batch_event_time_mjd_anchor[semi_hard_opt_idx]
                    if batch_event_time_mjd_anchor is not None
                    else None
                )
                z_l_hard, h_l_hard, dt_hard = encode_hard_negative_with_time_shift(
                    model,
                    opt_coords=coords_hard,
                    opt_t=opt_t[semi_hard_opt_idx],
                    opt_v=opt_v[semi_hard_opt_idx],
                    opt_mask=opt_mask[semi_hard_opt_idx],
                    opt_err=opt_err[semi_hard_opt_idx],
                    opt_ref_t=ref_time,
                    candidate_gw_time_mjd=candidate_gw_time_hard,
                    anchor_gw_time_mjd=batch_event_time_mjd_anchor,
                    scale_divisor=float(model_args.get("neg_offset_scale_days_divisor", 100.0)),
                )
                _cred_hard = (
                    compute_credible_level(gw_m, coords_hard)
                    if need_cred else None
                )
                if batch_event_time_mjd_anchor is not None:
                    _accumulate_dt_stats(dt_stats, "hard_negative", dt_hard)
                    dt_hard_all.append(dt_hard.detach().cpu())
                logits_hard = model.fusion_logits(
                    g, h_l_hard, z_l=z_l_hard, H_gw=H_gw, cred_level=_cred_hard,
                    gw_s=gw_s, gw_m=gw_m, opt_coords=coords_hard,
                    dt_days=dt_hard,
                )
                logits_hard_neg.append(logits_hard.float().cpu())
                if _cred_hard is not None:
                    cred_hard_neg.append(_cred_hard.detach().float().cpu())
                if batch_sources is not None:
                    semi_hard_idx_cpu = semi_hard_opt_idx.detach().cpu().numpy().tolist()
                    for hard_idx in semi_hard_idx_cpu:
                        hard_idx = int(hard_idx)
                        if 0 <= hard_idx < len(batch_sources):
                            source_hard_neg.append(batch_sources[hard_idx])
                        else:
                            source_hard_neg.append("unknown")

                # 3. GW negatives: has_kn=0 GW paired with KN optical
                if has_neg_gw:
                    sampled_neg_gw = gw_neg_rng.choice(neg_gw_indices, size=batch_size, replace=True)
                    gw_s_neg_np = np.stack([gw_scalars_ds[int(i)] for i in sampled_neg_gw], axis=0)
                    gw_m_neg_np = np.stack([gw_skymaps_ds[int(i)] for i in sampled_neg_gw], axis=0)

                    gw_s_neg = torch.from_numpy(gw_s_neg_np).to(device)
                    gw_m_neg = torch.from_numpy(gw_m_neg_np).to(device)

                    g_gw_neg, z_l_gw_neg, h_l_gw_neg, H_gw_neg = model.encode(
                        gw_s_neg, gw_m_neg, opt_coords, opt_t, opt_v, ref_time, opt_mask, opt_err
                    )
                    _cred_gw_neg = (
                        compute_credible_level(gw_m_neg, opt_coords)
                        if need_cred else None
                    )
                    dt_gw_neg = None
                    if opt_zero_time_mjd_base is not None and gw_event_time_mjd_table is not None:
                        sampled_neg_gw_tensor = torch.from_numpy(sampled_neg_gw).to(device=device, dtype=torch.long)
                        sampled_neg_gw_time = gw_event_time_mjd_table[sampled_neg_gw_tensor]
                        dt_gw_neg = compute_time_delta_days(opt_zero_time_mjd_base, sampled_neg_gw_time)
                        _accumulate_dt_stats(dt_stats, "gw_negative", dt_gw_neg)
                        dt_gw_all.append(dt_gw_neg.detach().cpu())
                    logits_gw = model.fusion_logits(
                        g_gw_neg, h_l_gw_neg, z_l=z_l_gw_neg, H_gw=H_gw_neg, cred_level=_cred_gw_neg,
                        gw_s=gw_s_neg, gw_m=gw_m_neg, opt_coords=opt_coords,
                        dt_days=dt_gw_neg,
                    )
                    logits_gw_neg.append(logits_gw.float().cpu())
                    if _cred_gw_neg is not None:
                        cred_gw_neg.append(_cred_gw_neg.detach().float().cpu())
                    if gw_source_types is not None:
                        source_gw_neg.extend(
                            _lookup_source_type(gw_idx, gw_source_types)
                            for gw_idx in sampled_neg_gw.tolist()
                        )

                # 4. Optical negatives: correct GW paired with non-KN transients
                if has_neg_optical:
                    batch_neg_indices = []
                    for _ in range(batch_size):
                        idx = neg_idx % total_neg
                        batch_neg_indices.append(idx)
                        neg_idx += 1
                    batch_neg_types = [neg_types_list[idx] for idx in batch_neg_indices]

                    neg_v_batch = neg_values[batch_neg_indices]
                    neg_t_batch = neg_times[batch_neg_indices]
                    neg_m_batch = neg_masks[batch_neg_indices]
                    neg_e_batch = neg_errors[batch_neg_indices]
                    neg_c_batch = neg_coords[batch_neg_indices]

                    ref_time_neg = build_ref_time(batch_size, n_ref, ref_start, ref_end,
                                                  device, neg_t_batch.dtype)
                    _cred_neg = (
                        compute_credible_level(gw_m, neg_c_batch)
                        if need_cred else None
                    )
                    if neg_offset_policy is not None and neg_offset_policy.enabled:
                        delta_np = neg_offset_policy.sample_offsets(batch_size)
                        delta_days = torch.from_numpy(delta_np).to(device=device, dtype=torch.float32)
                        shifted_neg_t_batch = apply_time_offsets(
                            neg_t_batch, neg_m_batch, delta_days, neg_offset_policy.scale_divisor
                        )
                        _, z_l_neg_i, h_l_neg_i, _ = model.encode(
                            gw_s, gw_m, neg_c_batch, shifted_neg_t_batch, neg_v_batch,
                            ref_time_neg, neg_m_batch, neg_e_batch
                        )
                        dt_optical = None
                        if neg_zero_time_mjd_cls_base is not None and batch_event_time_mjd_anchor is not None:
                            dt_optical = compute_time_delta_days(
                                neg_zero_time_mjd_cls_base[batch_neg_indices] - delta_days,
                                batch_event_time_mjd_anchor,
                            )
                            _accumulate_dt_stats(dt_stats, "optical_negative", dt_optical)
                            dt_optical_all.append(dt_optical.detach().cpu())
                        logits_optical = model.fusion_logits(
                            g, h_l_neg_i, z_l=z_l_neg_i, H_gw=H_gw, cred_level=_cred_neg,
                            gw_s=gw_s, gw_m=gw_m, opt_coords=neg_c_batch,
                            dt_days=dt_optical,
                        )
                    else:
                        _, z_l_neg, h_l_neg, _ = model.encode(
                            gw_s, gw_m, neg_c_batch, neg_t_batch, neg_v_batch,
                            ref_time_neg, neg_m_batch, neg_e_batch
                        )
                        dt_optical = None
                        if neg_zero_time_mjd_cls_base is not None and batch_event_time_mjd_anchor is not None:
                            dt_optical = compute_time_delta_days(
                                neg_zero_time_mjd_cls_base[batch_neg_indices], batch_event_time_mjd_anchor
                            )
                            _accumulate_dt_stats(dt_stats, "optical_negative", dt_optical)
                            dt_optical_all.append(dt_optical.detach().cpu())
                        logits_optical = model.fusion_logits(
                            g, h_l_neg, z_l=z_l_neg, H_gw=H_gw, cred_level=_cred_neg,
                            gw_s=gw_s, gw_m=gw_m, opt_coords=neg_c_batch,
                            dt_days=dt_optical,
                        )
                    logits_optical_neg.append(logits_optical.float().cpu())
                    if _cred_neg is not None:
                        cred_optical_neg.append(_cred_neg.detach().float().cpu())
                    optical_neg_types.extend(batch_neg_types)
                    if batch_sources is not None:
                        source_optical_neg.extend(batch_sources)
    finally:
        if gw_file is not None:
            gw_file.close()

    finalized_dt_stats = _finalize_dt_stats(dt_stats)
    dt_has_any = any(int(v.get("count", 0)) > 0 for v in finalized_dt_stats.values())
    result = {
        "logits_positive": torch.cat(logits_positive) if logits_positive else None,
        "logits_hard_neg": torch.cat(logits_hard_neg) if logits_hard_neg else None,
        "logits_gw_neg": torch.cat(logits_gw_neg) if logits_gw_neg else None,
        "cred_positive": torch.cat(cred_positive) if cred_positive else None,
        "cred_hard_neg": torch.cat(cred_hard_neg) if cred_hard_neg else None,
        "cred_gw_neg": torch.cat(cred_gw_neg) if cred_gw_neg else None,
        "source_positive": source_positive,
        "source_optical_neg": source_optical_neg,
        "source_gw_neg": source_gw_neg,
        "source_hard_neg": source_hard_neg,
        "time_delta_meta": {
            "enabled": bool(dt_has_any),
            "stats": finalized_dt_stats,
            "note": "Computed from available time fields.",
        },
        "dt_positive_days": torch.cat(dt_positive_all) if dt_positive_all else None,
        "dt_optical_negative_days": torch.cat(dt_optical_all) if dt_optical_all else None,
        "dt_gw_negative_days": torch.cat(dt_gw_all) if dt_gw_all else None,
        "dt_hard_negative_days": torch.cat(dt_hard_all) if dt_hard_all else None,
    }

    if has_neg_optical and logits_optical_neg:
        optical_logits = torch.cat(logits_optical_neg)
        result["logits_optical_neg"] = optical_logits
        result["cred_optical_neg"] = torch.cat(cred_optical_neg) if cred_optical_neg else None
        result["optical_neg_types"] = optical_neg_types
    else:
        result["logits_optical_neg"] = None
        result["cred_optical_neg"] = None
        result["optical_neg_types"] = []

    return result


def evaluate_retrieval_batch_mode(embeddings, ks=(1, 5, 10)):
    """Per-batch retrieval metrics (same as training eval)."""
    results_accum = []
    for sim, gi in zip(embeddings["sim_blocks"], embeddings["gw_idx_blocks"]):
        r = compute_retrieval_metrics(sim, gi, ks=ks)
        results_accum.append(r)

    avg = {}
    if results_accum:
        for key in results_accum[0]:
            avg[key] = sum(d[key] for d in results_accum) / len(results_accum)
    return avg


@torch.no_grad()
def _build_gallery_query_cache(model, unique_gw_ids, test_data_path, device,
                               amp_dtype=torch.float32, amp_enabled=False):
    """Cache GW-side classification features once per GW event for gallery eval."""
    if test_data_path is None or not os.path.exists(test_data_path):
        raise FileNotFoundError(f"test_data_path not found for gallery logits eval: {test_data_path}")

    core_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    dual = _is_dual_fusion_model(model)
    cache = {}

    with h5py.File(test_data_path, 'r') as f:
        gw_scalars_ds = f['events/gw_data/scalars']
        gw_skymaps_ds = f['events/gw_data/skymaps']

        for gw_id in tqdm(unique_gw_ids, desc="Caching GW query features"):
            gw_id = int(gw_id)
            gw_s = torch.from_numpy(gw_scalars_ds[gw_id]).unsqueeze(0).to(device)
            gw_m = torch.from_numpy(gw_skymaps_ds[gw_id]).unsqueeze(0).to(device)

            with _autocast_context(device, amp_dtype, enabled=amp_enabled):
                g, H_gw = core_model.gw_encoder(gw_s, gw_m)
                feature_dropout = getattr(core_model, "feature_dropout", None)
                if feature_dropout is not None and getattr(feature_dropout, "p", 0.0) > 0:
                    g = feature_dropout(g)
                    H_gw = feature_dropout(H_gw)

            item = {
                "g": g.float().cpu().squeeze(0),
                "gw_s": gw_s.float().cpu().squeeze(0),
            }
            if dual:
                item["H_gw"] = H_gw.float().cpu().squeeze(0)
                item["gw_m"] = gw_m.float().cpu().squeeze(0)
            cache[gw_id] = item

    return cache


def _compute_credible_level_single_gw(gw_m_single, opt_coords):
    """Credible level for one GW skymap against multiple optical coordinates."""
    import math

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

    pix_xyz = gw_m_single[:3, :]  # [3, 19200]
    dot = torch.matmul(opt_xyz, pix_xyz)  # [B, 19200]
    nearest_idx = dot.argmax(dim=-1)  # [B]

    dA = gw_m_single[3, :]  # [19200]
    dP = gw_m_single[4, :]  # [19200]
    dA = torch.nan_to_num(dA.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    dP = torch.nan_to_num(dP.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    density = dP / dA.clamp_min(torch.finfo(dP.dtype).eps)
    density_at_opt = density[nearest_idx]  # [B]
    total_probability = dP.sum().clamp_min(torch.finfo(dP.dtype).eps)
    cred_level = torch.where(
        density.unsqueeze(0) >= density_at_opt.unsqueeze(-1),
        dP.unsqueeze(0),
        torch.zeros_like(dP).unsqueeze(0),
    ).sum(dim=-1) / total_probability
    return cred_level.unsqueeze(-1)  # [B, 1]


@torch.no_grad()
def _score_gallery_candidates_with_logits(model, query_cache, candidate_indices,
                                          embeddings, device, dual,
                                          query_gw_id=None,
                                          gw_event_time_mjd_table=None,
                                          reencode_time_shift=False,
                                          scale_divisor=100.0,
                                          n_ref=64,
                                          ref_start=-0.3,
                                          ref_end=0.6,
                                          logits_chunk_size=1024,
                                          amp_dtype=torch.float32, amp_enabled=False):
    """Score (GW query, optical candidate) pairs with fusion logits."""
    if len(candidate_indices) == 0:
        return np.array([], dtype=np.float32)
    need_cred = _model_requires_cred_level(model)

    h_l_all = embeddings["h_l_cls"]
    z_l_all = embeddings["z_l_cls"]
    opt_coords_all = embeddings["opt_coords"]
    gw_idx_all = embeddings["gw_indices"]
    opt_t_all = embeddings.get("opt_t_raw")
    opt_v_all = embeddings.get("opt_v_raw")
    opt_mask_all = embeddings.get("opt_mask_raw")
    opt_err_all = embeddings.get("opt_err_raw")

    g_query = query_cache["g"].to(device)
    gw_s_query = query_cache["gw_s"].to(device)
    if dual:
        H_query = query_cache["H_gw"].to(device)
        gw_m_query = query_cache["gw_m"].to(device)

    scores = []
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    can_reencode_with_shift = bool(
        reencode_time_shift
        and gw_event_time_mjd_table is not None
        and query_gw_id is not None
        and opt_t_all is not None
        and opt_v_all is not None
        and opt_mask_all is not None
        and opt_err_all is not None
    )
    if reencode_time_shift and not can_reencode_with_shift:
        print(
            "WARNING: Gallery distractor time-shift re-encoding requested but unavailable "
            "(missing gw_event_time_mjd_table/query_gw_id/raw optical cache). Fallback to cached optical features."
        )
    scale_divisor = float(scale_divisor)
    if scale_divisor <= 0:
        raise ValueError("scale_divisor must be > 0 for gallery time-shift re-encoding.")

    for start in range(0, len(candidate_indices), logits_chunk_size):
        chunk_np = candidate_indices[start:start + logits_chunk_size]
        chunk_idx = torch.from_numpy(chunk_np).long()

        opt_coords_chunk = opt_coords_all.index_select(0, chunk_idx).to(device)
        dt_chunk = None
        cand_gw_idx_chunk = None
        if gw_event_time_mjd_table is not None and query_gw_id is not None:
            cand_gw_idx_chunk = gw_idx_all.index_select(0, chunk_idx).to(device=device, dtype=torch.long)
            query_gw_idx_chunk = torch.full_like(cand_gw_idx_chunk, int(query_gw_id))
            cand_gw_time = gw_event_time_mjd_table[cand_gw_idx_chunk]
            query_gw_time = gw_event_time_mjd_table[query_gw_idx_chunk]
            dt_chunk = compute_time_delta_days(cand_gw_time, query_gw_time)
        if can_reencode_with_shift:
            opt_t_chunk = opt_t_all.index_select(0, chunk_idx).to(device)
            opt_v_chunk = opt_v_all.index_select(0, chunk_idx).to(device)
            opt_mask_chunk = opt_mask_all.index_select(0, chunk_idx).to(device)
            opt_err_chunk = opt_err_all.index_select(0, chunk_idx).to(device)
            delta_days = dt_chunk
            shifted_opt_t = apply_time_offsets(
                opt_t_chunk, opt_mask_chunk, delta_days, scale_divisor
            )
            ref_time_chunk = build_ref_time(
                shifted_opt_t.size(0), int(n_ref), float(ref_start), float(ref_end),
                device, shifted_opt_t.dtype
            )
            with _autocast_context(device, amp_dtype, enabled=amp_enabled):
                z_chunk_shift, h_chunk_shift = model.encode_optical(
                    opt_coords_chunk, shifted_opt_t, opt_v_chunk, ref_time_chunk, opt_mask_chunk, opt_err_chunk
                )
            h_chunk = h_chunk_shift
            z_chunk = z_chunk_shift if dual else None
        else:
            h_chunk = h_l_all.index_select(0, chunk_idx).to(device)
            z_chunk = z_l_all.index_select(0, chunk_idx).to(device) if dual else None

        batch_size = h_chunk.size(0)
        g_chunk = g_query.unsqueeze(0).expand(batch_size, -1)

        if dual:
            H_chunk = H_query.unsqueeze(0).expand(batch_size, -1, -1)
            cred_chunk = _compute_credible_level_single_gw(gw_m_query, opt_coords_chunk) if need_cred else None
            gw_s_chunk = gw_s_query.unsqueeze(0).expand(batch_size, -1)
            gw_m_chunk = gw_m_query.unsqueeze(0).expand(batch_size, -1, -1) if need_cred else None
        else:
            H_chunk = None
            cred_chunk = None
            gw_s_chunk = None
            gw_m_chunk = None

        with _autocast_context(device, amp_dtype, enabled=amp_enabled):
            logits = model.fusion_logits(
                g_chunk, h_chunk, z_l=z_chunk, H_gw=H_chunk, cred_level=cred_chunk,
                gw_s=gw_s_chunk, gw_m=gw_m_chunk, opt_coords=opt_coords_chunk,
                dt_days=dt_chunk,
            )
        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        scores.append(probs.cpu())

    return torch.cat(scores).numpy()


@torch.no_grad()
def evaluate_retrieval_gallery_mode(model, embeddings, gallery_sizes, test_data_path,
                                    device, n_trials=10, seed=42,
                                    model_args=None,
                                    gw_event_time_mjd_table=None,
                                    amp_dtype=torch.float32, amp_enabled=False,
                                    gw_source_types=None):
    """
    Gallery retrieval with classification branch logits.

    For each GW query event, build candidate pool of size N
    (1 correct optical + N-1 distractors), score each pair by fusion-head
    positive probability, then compute Recall@K and MRR.

    If gw_source_types is provided, also returns per-source metrics under
    the "by_source" key of the result dict.
    """
    gw_indices = embeddings["gw_indices"]
    unique_gw = torch.unique(gw_indices).cpu().tolist()
    rng = np.random.default_rng(seed)
    dual = bool(embeddings.get("dual_fusion", _is_dual_fusion_model(model)))

    if "h_l_cls" not in embeddings or "z_l_cls" not in embeddings:
        raise KeyError("Missing classification features in embeddings for gallery logits eval.")
    gallery_shift_enable = bool(gw_event_time_mjd_table is not None)
    if gallery_shift_enable:
        print("Gallery retrieval: enabling candidate timeline shift re-encoding for distractors.")
    else:
        print("Gallery retrieval: event_time_mjd unavailable, using unshifted cached optical features.")
    model_args = model_args or {}
    n_ref = int(model_args.get("n_ref", 64))
    ref_start = float(model_args.get("ref_start", -0.3))
    ref_end = float(model_args.get("ref_end", 0.6))
    scale_divisor = float(model_args.get("neg_offset_scale_days_divisor", 100.0))

    # Build per-GW source label map for per-source accumulation
    gw_source_map: Dict[int, str] = {}
    if gw_source_types is not None:
        for gw_id in unique_gw:
            gw_source_map[int(gw_id)] = _lookup_source_type(int(gw_id), gw_source_types)
        source_labels = sorted(set(gw_source_map.values()))
        print(f"Gallery retrieval: tracking per-source metrics for {source_labels}")

    print("Building GW query cache for logits-based gallery retrieval...")
    gw_query_cache = _build_gallery_query_cache(
        model, unique_gw, test_data_path, device,
        amp_dtype=amp_dtype, amp_enabled=amp_enabled
    )

    results = {}
    # per_source_accum[source][N] = {"recalls": {1:[], 5:[], 10:[]}, "mrrs": []}
    per_source_accum: Dict[str, Dict[int, Dict]] = {}

    for N in gallery_sizes:
        recalls = {1: [], 5: [], 10: []}
        mrrs = []

        for _ in range(n_trials):
            for gw_id in unique_gw:
                gw_mask = gw_indices == gw_id
                other_mask = gw_indices != gw_id

                if int(gw_mask.sum().item()) == 0 or int(other_mask.sum().item()) == 0:
                    continue

                gw_idxs = torch.where(gw_mask)[0]
                correct_idx = int(gw_idxs[rng.integers(len(gw_idxs))].item())

                other_idxs = torch.where(other_mask)[0].cpu().numpy()
                n_distract = min(N - 1, len(other_idxs))
                if n_distract > 0:
                    distract_idxs = rng.choice(other_idxs, size=n_distract, replace=False)
                else:
                    distract_idxs = np.array([], dtype=np.int64)

                gallery_indices = np.concatenate(
                    ([correct_idx], np.asarray(distract_idxs, dtype=np.int64))
                )
                probs = _score_gallery_candidates_with_logits(
                    model, gw_query_cache[int(gw_id)], gallery_indices,
                    embeddings, device, dual,
                    query_gw_id=int(gw_id),
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    reencode_time_shift=gallery_shift_enable,
                    scale_divisor=scale_divisor,
                    n_ref=n_ref,
                    ref_start=ref_start,
                    ref_end=ref_end,
                    amp_dtype=amp_dtype, amp_enabled=amp_enabled
                )
                ranked = np.argsort(-probs)

                # Correct optical is inserted at gallery index 0.
                correct_rank = int(np.where(ranked == 0)[0][0])

                for k in recalls:
                    recalls[k].append(1.0 if correct_rank < k else 0.0)
                mrrs.append(1.0 / (correct_rank + 1))

                # Per-source accumulation
                if gw_source_map:
                    src = gw_source_map.get(int(gw_id), "unknown")
                    if src not in per_source_accum:
                        per_source_accum[src] = {}
                    if N not in per_source_accum[src]:
                        per_source_accum[src][N] = {
                            "recalls": {1: [], 5: [], 10: []},
                            "mrrs": [],
                        }
                    src_acc = per_source_accum[src][N]
                    for k in src_acc["recalls"]:
                        src_acc["recalls"][k].append(1.0 if correct_rank < k else 0.0)
                    src_acc["mrrs"].append(1.0 / (correct_rank + 1))

        for k in recalls:
            key = f"gallery_{N}_recall_at_{k}"
            results[key] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        results[f"gallery_{N}_mrr"] = float(np.mean(mrrs)) if mrrs else 0.0

    # Collapse per-source accumulators into flat metric dicts
    if per_source_accum:
        by_source: Dict[str, Dict] = {}
        for src, size_acc in per_source_accum.items():
            src_metrics: Dict[str, float] = {}
            for N, acc in size_acc.items():
                for k, vals in acc["recalls"].items():
                    src_metrics[f"gallery_{N}_recall_at_{k}"] = float(np.mean(vals)) if vals else 0.0
                src_metrics[f"gallery_{N}_mrr"] = float(np.mean(acc["mrrs"])) if acc["mrrs"] else 0.0
            by_source[src] = src_metrics
        results["by_source"] = by_source

    return results


def evaluate_classification_triplet(triplet_logits, report_dt_bins=False, dt_bin_edges=None, report_dt_macro=False):
    """Classification metrics using all pair types from triplet extraction.

    Positive: matched (GW, KN) pairs → label 1
    Negative:
      - optical negatives (GW, nonKN)
      - GW negatives (GW_has_kn0, KN)
      - semi-hard negatives (GW_wrong, KN)
      → label 0
    """
    all_probs = []
    all_labels = []
    probs_pos = None
    probs_optical = None
    probs_gw = None
    probs_hard = None
    all_abs_dt = []

    def _append_dt_or_nan(dt_tensor, expected_len):
        if expected_len <= 0:
            return
        if dt_tensor is not None and len(dt_tensor) == expected_len:
            all_abs_dt.append(dt_tensor.abs().to(torch.float32))
        else:
            all_abs_dt.append(torch.full((expected_len,), float("nan"), dtype=torch.float32))

    # Positives
    if triplet_logits.get("logits_positive") is not None:
        probs_pos = torch.softmax(triplet_logits["logits_positive"].float(), dim=1)[:, 1]
        all_probs.append(probs_pos)
        all_labels.append(torch.ones(len(probs_pos), dtype=torch.long))
        dt_pos = triplet_logits.get("dt_positive_days")
        _append_dt_or_nan(dt_pos, len(probs_pos))

    optical_logits = triplet_logits.get("logits_optical_neg")
    if optical_logits is not None:
        probs_optical = torch.softmax(optical_logits.float(), dim=1)[:, 1]
        all_probs.append(probs_optical)
        all_labels.append(torch.zeros(len(probs_optical), dtype=torch.long))
        dt_opt = triplet_logits.get("dt_optical_negative_days")
        _append_dt_or_nan(dt_opt, len(probs_optical))

    # GW negatives (GW_has_kn0, KN)
    if triplet_logits.get("logits_gw_neg") is not None:
        probs_gw = torch.softmax(triplet_logits["logits_gw_neg"].float(), dim=1)[:, 1]
        all_probs.append(probs_gw)
        all_labels.append(torch.zeros(len(probs_gw), dtype=torch.long))
        dt_gw = triplet_logits.get("dt_gw_negative_days")
        _append_dt_or_nan(dt_gw, len(probs_gw))

    # Semi-hard negatives (GW_wrong, KN)
    if triplet_logits.get("logits_hard_neg") is not None:
        probs_hard = torch.softmax(triplet_logits["logits_hard_neg"].float(), dim=1)[:, 1]
        all_probs.append(probs_hard)
        all_labels.append(torch.zeros(len(probs_hard), dtype=torch.long))
        dt_h = triplet_logits.get("dt_hard_negative_days")
        _append_dt_or_nan(dt_h, len(probs_hard))

    if not all_probs:
        return {}

    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)

    n_pos = (labels == 1).sum().item()
    n_optical = len(probs_optical) if probs_optical is not None else 0
    n_gw = len(probs_gw) if probs_gw is not None else 0
    n_hard = len(probs_hard) if probs_hard is not None else 0
    print(
        "  Triplet classification: "
        f"{n_pos} pos + {n_optical} optical_neg + {n_gw} gw_neg + {n_hard} semi_hard_neg = {len(probs)} total"
    )

    metrics = compute_classification_metrics(probs, labels)
    if report_dt_bins and dt_bin_edges is not None and all_abs_dt:
        abs_dt = torch.cat(all_abs_dt)
        coverage_n = int(torch.isfinite(abs_dt).sum().item())
        metrics["dt_bins_coverage"] = {
            "n_total": int(len(abs_dt)),
            "n_with_dt": coverage_n,
            "coverage_ratio": float(coverage_n / float(len(abs_dt))) if len(abs_dt) > 0 else 0.0,
        }
        metrics["dt_bins"] = compute_dt_bin_metrics(probs, labels, abs_dt, dt_bin_edges)
        if report_dt_macro:
            metrics["dt_macro"] = compute_dt_macro_metrics(metrics["dt_bins"])
    return metrics


def _summarize_score_tensor(scores):
    if scores is None:
        return None
    vals = scores.detach().float().cpu().numpy()
    if vals.size == 0:
        return None
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "n": int(vals.size),
    }


def _metric_delta(current_metrics, baseline_metrics, keys=("auroc", "auprc", "f1_optimal", "ece")):
    out = {}
    if not baseline_metrics:
        return out
    for key in keys:
        if key in current_metrics and key in baseline_metrics:
            out[f"delta_{key}"] = float(current_metrics[key] - baseline_metrics[key])
    return out


def summarize_triplet_prob_distributions(triplet_logits):
    optical_logits = triplet_logits.get("logits_optical_neg")
    return {
        "positive": _summarize_score_tensor(
            torch.softmax(triplet_logits["logits_positive"].float(), dim=1)[:, 1]
        ) if triplet_logits.get("logits_positive") is not None else None,
        "optical_negative": _summarize_score_tensor(
            torch.softmax(optical_logits.float(), dim=1)[:, 1]
        ) if optical_logits is not None else None,
        "gw_negative": _summarize_score_tensor(
            torch.softmax(triplet_logits["logits_gw_neg"].float(), dim=1)[:, 1]
        ) if triplet_logits.get("logits_gw_neg") is not None else None,
        "hard_negative": _summarize_score_tensor(
            torch.softmax(triplet_logits["logits_hard_neg"].float(), dim=1)[:, 1]
        ) if triplet_logits.get("logits_hard_neg") is not None else None,
    }


def evaluate_classification_triplet_by_source(triplet_logits):
    """Per-source classification metrics from triplet pairs (bns/nsbh)."""
    source_keys = (
        "source_positive",
        "source_optical_neg",
        "source_gw_neg",
        "source_hard_neg",
    )
    has_source = any(len(triplet_logits.get(k, [])) > 0 for k in source_keys)
    if not has_source:
        print("WARNING: source_type labels unavailable. Skipping classification_by_source.")
        return {}

    per_source = {}

    def ensure_bucket(source):
        source = _normalize_source_type(source)
        if source not in per_source:
            per_source[source] = {
                "probs": [],
                "labels": [],
                "pair_type_counts": {
                    "n_total": 0,
                    "n_positive": 0,
                    "n_optical_neg": 0,
                    "n_gw_neg": 0,
                    "n_hard_neg": 0,
                },
            }
        return per_source[source]

    def add_pair_type(logits, sources, label, count_key, pair_name):
        if logits is None:
            return

        probs = torch.softmax(logits.float(), dim=1)[:, 1]
        if sources is None:
            print(f"WARNING: Missing source labels for {pair_name}. Skipping this pair type.")
            return

        if len(sources) != len(probs):
            n = min(len(sources), len(probs))
            print(
                f"WARNING: source label length mismatch for {pair_name}: "
                f"{len(sources)} labels vs {len(probs)} logits. Truncating to {n}."
            )
            if n <= 0:
                return
            probs = probs[:n]
            sources = sources[:n]

        src_np = np.array([_normalize_source_type(s) for s in sources], dtype=object)
        for src in np.unique(src_np):
            mask_np = src_np == src
            mask = torch.from_numpy(mask_np)
            src_probs = probs[mask]
            if src_probs.numel() == 0:
                continue
            bucket = ensure_bucket(src)
            bucket["probs"].append(src_probs)
            bucket["labels"].append(
                torch.full((src_probs.numel(),), label, dtype=torch.long)
            )
            bucket["pair_type_counts"][count_key] += int(src_probs.numel())

    add_pair_type(
        triplet_logits.get("logits_positive"),
        triplet_logits.get("source_positive"),
        label=1,
        count_key="n_positive",
        pair_name="positive",
    )

    optical_logits = triplet_logits.get("logits_optical_neg")
    add_pair_type(
        optical_logits,
        triplet_logits.get("source_optical_neg"),
        label=0,
        count_key="n_optical_neg",
        pair_name="optical_neg",
    )

    add_pair_type(
        triplet_logits.get("logits_gw_neg"),
        triplet_logits.get("source_gw_neg"),
        label=0,
        count_key="n_gw_neg",
        pair_name="gw_neg",
    )

    add_pair_type(
        triplet_logits.get("logits_hard_neg"),
        triplet_logits.get("source_hard_neg"),
        label=0,
        count_key="n_hard_neg",
        pair_name="hard_neg",
    )

    results = {}
    for source in sorted(per_source):
        bucket = per_source[source]
        if not bucket["probs"]:
            continue

        probs = torch.cat(bucket["probs"])
        labels = torch.cat(bucket["labels"])
        counts = bucket["pair_type_counts"]
        counts["n_total"] = (
            counts["n_positive"]
            + counts["n_optical_neg"]
            + counts["n_gw_neg"]
            + counts["n_hard_neg"]
        )

        metrics = compute_classification_metrics(probs, labels)
        tp = float(metrics.get("tp", 0))
        fp = float(metrics.get("fp", 0))
        fn = float(metrics.get("fn", 0))
        precision_conf = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_conf = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_conf = (
            2.0 * precision_conf * recall_conf / (precision_conf + recall_conf)
            if (precision_conf + recall_conf) > 0 else 0.0
        )
        metrics["precision_confusion"] = precision_conf
        metrics["recall_confusion"] = recall_conf
        metrics["f1_confusion"] = f1_conf

        results[source] = {
            "metrics": metrics,
            "pair_type_counts": counts,
        }

    if results:
        summary = ", ".join(
            f"{src}:n={vals['pair_type_counts']['n_total']}" for src, vals in results.items()
        )
        print(f"  Triplet classification by source: {summary}")

    return results


def evaluate_embeddings(embeddings):
    """Embedding quality metrics."""
    feat_g = embeddings["feat_g"]
    feat_o = embeddings["feat_o"]
    gw_indices = embeddings["gw_indices"]
    feat_g, feat_o, gw_indices, n_drop = _filter_finite_embedding_rows(
        feat_g, feat_o, gw_indices
    )
    if n_drop > 0:
        print(f"WARNING: Dropping {n_drop} non-finite embedding rows before quality metrics.")
    if feat_g.size(0) < 2:
        print("WARNING: Not enough finite embedding rows for quality metrics; returning zeros.")
        return {
            "alignment": 0.0,
            "uniformity_gw": 0.0,
            "uniformity_opt": 0.0,
            "inter_modal_gap": 0.0,
            "intra_sim": 0.0,
            "inter_sim": 0.0,
        }
    return compute_embedding_metrics(
        feat_g,
        feat_o,
        gw_indices,
    )


def generate_plots(embeddings, results, output_dir, triplet_logits=None):
    """Generate evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Build probs/labels for ROC/PR/calibration ---
    # Prefer triplet logits (consistent with results["classification"]);
    # fall back to roll-by-1 embeddings if triplet_logits unavailable.
    if triplet_logits is not None:
        all_probs, all_labels = [], []
        pos = triplet_logits.get("logits_positive")
        if pos is not None:
            all_probs.append(torch.softmax(pos.float(), dim=1)[:, 1])
            all_labels.append(torch.ones(pos.size(0), dtype=torch.long))
        for key in ("logits_optical_neg", "logits_gw_neg", "logits_hard_neg"):
            neg = triplet_logits.get(key)
            if neg is not None:
                all_probs.append(torch.softmax(neg.float(), dim=1)[:, 1])
                all_labels.append(torch.zeros(neg.size(0), dtype=torch.long))
        if all_probs:
            probs = torch.cat(all_probs).numpy()
            labels_np = torch.cat(all_labels).numpy()
        else:
            probs = np.array([], dtype=np.float32)
            labels_np = np.array([], dtype=np.int64)
    else:
        logits = embeddings["logits"]
        labels = embeddings["labels"]
        probs = torch.softmax(logits.float(), dim=1)[:, 1].numpy()
        labels_np = labels.numpy()

    finite_cls = np.isfinite(probs) & np.isfinite(labels_np)
    n_bad_cls = int((~finite_cls).sum())
    if n_bad_cls > 0:
        print(f"WARNING: Dropping {n_bad_cls} non-finite classification samples before plotting.")
        probs = probs[finite_cls]
        labels_np = labels_np[finite_cls]
    if len(probs) == 0:
        print("WARNING: No finite classification samples available. Skipping ROC/PR/calibration plots.")
        probs = np.array([], dtype=np.float32)
        labels_np = np.array([], dtype=np.int64)

    sorted_idx = np.argsort(-probs) if len(probs) > 0 else np.array([], dtype=np.int64)
    sorted_labels = labels_np[sorted_idx] if len(sorted_idx) > 0 else np.array([], dtype=np.int64)
    n_pos = sorted_labels.sum() if len(sorted_labels) > 0 else 0
    n_neg = len(sorted_labels) - n_pos

    if n_pos > 0 and n_neg > 0:
        tpr = np.cumsum(sorted_labels) / n_pos
        fpr = np.cumsum(1 - sorted_labels) / n_neg
        tpr = np.concatenate([[0], tpr])
        fpr = np.concatenate([[0], fpr])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        auroc = results.get("classification", {}).get("auroc", 0)
        ax.set_title(f"ROC Curve (AUROC={auroc:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # --- 2. Precision-Recall Curve ---
    if n_pos > 0:
        tp_cum = np.cumsum(sorted_labels)
        ranks = np.arange(1, len(sorted_labels) + 1)
        precision = tp_cum / ranks
        recall = tp_cum / n_pos

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(recall, precision, lw=2)
        auprc = results.get("classification", {}).get("auprc", 0)
        ax.set_title(f"Precision-Recall Curve (AUPRC={auprc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        fig.savefig(os.path.join(output_dir, "pr_curve.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # --- 3. Calibration Plot ---
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= low) & (probs < high) if i < n_bins - 1 else (probs >= low) & (probs <= high)
        if mask.sum() > 0:
            bin_accs.append(labels_np[mask].mean())
            bin_confs.append(probs[mask].mean())
            bin_counts.append(mask.sum())

    if bin_accs:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, label="Model")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
        ece = results.get("classification", {}).get("ece", 0)
        ax.set_title(f"Calibration (ECE={ece:.4f})")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.legend()
        fig.savefig(os.path.join(output_dir, "calibration.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # --- 4. Gallery-mode Recall@1 vs Pool Size ---
    gallery_results = results.get("retrieval_gallery", {})
    if gallery_results:
        sizes = sorted(set(
            int(k.split("_")[1]) for k in gallery_results if "recall_at_1" in k
        ))
        r1_values = [gallery_results.get(f"gallery_{s}_recall_at_1", 0)
                     for s in sizes]

        if sizes:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(sizes, r1_values, "o-", lw=2)
            ax.set_xlabel("Gallery Size (number of candidates)")
            ax.set_ylabel("Recall@1")
            ax.set_title("Recall@1 vs Candidate Pool Size")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            fig.savefig(os.path.join(output_dir, "recall_vs_gallery.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # --- 5. t-SNE of embeddings ---
    try:
        from sklearn.manifold import TSNE
        n_total = len(embeddings["feat_g"])
        n_sample = min(2000, n_total)
        if n_sample < 2:
            raise ValueError("Not enough samples for t-SNE.")

        perm = np.random.RandomState(42).permutation(n_total)[:n_sample]
        fg = embeddings["feat_g"][perm].numpy()
        fo = embeddings["feat_o"][perm].numpy()
        gi = embeddings["gw_indices"][perm].numpy()
        source_labels = None
        source_all = embeddings.get("gw_source_labels")
        if source_all is not None and len(source_all) == n_total:
            source_labels = np.asarray(source_all, dtype=object)[perm]

        finite_embed = np.isfinite(fg).all(axis=1) & np.isfinite(fo).all(axis=1)
        n_drop_embed = int((~finite_embed).sum())
        if n_drop_embed > 0:
            print(f"WARNING: Dropping {n_drop_embed} non-finite embedding samples before t-SNE.")
            fg = fg[finite_embed]
            fo = fo[finite_embed]
            gi = gi[finite_embed]
            if source_labels is not None:
                source_labels = source_labels[finite_embed]
            n_sample = len(fg)

        if n_sample < 5:
            raise ValueError("Not enough finite samples for t-SNE.")

        combined = np.concatenate([fg, fo])
        if not np.isfinite(combined).all():
            raise ValueError("Combined t-SNE input still contains non-finite values.")

        perplexity = min(30, max(5, combined.shape[0] - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = tsne.fit_transform(combined)

        fig, ax = plt.subplots(figsize=(10, 8))
        # Color by GW event, shape by modality.
        # Prefer a balanced BNS/NSBH subset to avoid index-order bias.
        unique_events = np.unique(gi)
        max_events_plot = min(20, len(unique_events))
        selected_events = np.array(unique_events[:max_events_plot])
        norm_sources = None

        if source_labels is not None:
            norm_sources = np.array([_normalize_source_type(s) for s in source_labels], dtype=object)
            event_to_source = {}
            for ev in unique_events:
                ev_idx = np.where(gi == ev)[0]
                if len(ev_idx) > 0:
                    event_to_source[int(ev)] = norm_sources[ev_idx[0]]

            bns_events = [int(ev) for ev in unique_events if event_to_source.get(int(ev)) == "bns"]
            nsbh_events = [int(ev) for ev in unique_events if event_to_source.get(int(ev)) == "nsbh"]
            other_events = [
                int(ev) for ev in unique_events
                if event_to_source.get(int(ev)) not in {"bns", "nsbh"}
            ]
            rng_events = np.random.RandomState(42)
            bns_events = rng_events.permutation(bns_events).tolist()
            nsbh_events = rng_events.permutation(nsbh_events).tolist()
            other_events = rng_events.permutation(other_events).tolist()

            target_bns = max_events_plot // 2
            target_nsbh = max_events_plot - target_bns
            selected = []
            selected.extend(bns_events[:min(target_bns, len(bns_events))])
            selected.extend(nsbh_events[:min(target_nsbh, len(nsbh_events))])

            if len(selected) < max_events_plot:
                remaining_pool = (
                    bns_events[min(target_bns, len(bns_events)):] +
                    nsbh_events[min(target_nsbh, len(nsbh_events)):] +
                    other_events
                )
                need = max_events_plot - len(selected)
                selected.extend(remaining_pool[:need])

            if selected:
                selected_events = np.array(rng_events.permutation(selected).tolist(), dtype=unique_events.dtype)
                n_bns_sel = int(sum(event_to_source.get(int(ev), "unknown") == "bns" for ev in selected_events))
                n_nsbh_sel = int(sum(event_to_source.get(int(ev), "unknown") == "nsbh" for ev in selected_events))
                print(
                    f"t-SNE event subset (first plot): n={len(selected_events)} "
                    f"[bns={n_bns_sel}, nsbh={n_nsbh_sel}]"
                )

        colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(selected_events))))

        for i, ev in enumerate(selected_events):
            mask = gi == ev
            c = colors[i % len(colors)]
            idx_gw = np.where(mask)[0]
            idx_opt = np.where(mask)[0] + n_sample
            ax.scatter(coords[idx_gw, 0], coords[idx_gw, 1],
                       c=[c], marker="^", s=40, alpha=0.7)
            ax.scatter(coords[idx_opt, 0], coords[idx_opt, 1],
                       c=[c], marker="o", s=30, alpha=0.7)

        ax.set_title("t-SNE Embedding Space (^=GW, o=Optical)")
        fig.savefig(os.path.join(output_dir, "tsne_embeddings.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

        # --- 6. t-SNE of GW embeddings by source_type (BNS/NSBH) ---
        if source_labels is not None:
            if norm_sources is None:
                norm_sources = np.array([_normalize_source_type(s) for s in source_labels], dtype=object)
            unique_sources = sorted(np.unique(norm_sources))
            if len(unique_sources) > 0:
                fig, ax = plt.subplots(figsize=(8, 7))
                source_palette = {
                    "bns": "#1f77b4",
                    "nsbh": "#d62728",
                    "unknown": "#7f7f7f",
                }
                fallback_colors = plt.cm.Set2(np.linspace(0, 1, max(3, len(unique_sources))))
                for i, src in enumerate(unique_sources):
                    mask = norm_sources == src
                    idx = np.where(mask)[0]
                    if len(idx) == 0:
                        continue
                    color = source_palette.get(src, fallback_colors[i % len(fallback_colors)])
                    ax.scatter(
                        coords[idx, 0], coords[idx, 1],
                        c=[color], marker="o", s=36, alpha=0.85,
                        label=f"{src} (n={len(idx)})"
                    )

                ax.set_title("t-SNE of GW Embeddings by Source Type")
                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")
                ax.legend(loc="best", fontsize=10)
                ax.grid(True, alpha=0.2)
                fig.savefig(os.path.join(output_dir, "tsne_gw_by_source.png"),
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
        else:
            print("source labels unavailable in embeddings, skipping source-type t-SNE plot.")
    except ImportError:
        print("sklearn not available, skipping t-SNE plot")
    except ValueError as e:
        print(f"Skipping t-SNE plot: {e}")

    print(f"Plots saved to {output_dir}/")


def generate_logits_distribution_plot(triplet_logits, output_dir):
    """Generate logits distribution plots for four pair types.

    Plot types:
    1. Positive (GW, KN)
    2. Optical negatives (GW, nonKN)
    3. GW negatives (GW_has_kn0, KN)
    4. Semi-hard negatives (GW_wrong, KN)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("matplotlib not available, skipping logits distribution plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract probabilities (class 1 = match probability)
    probs_pos = None
    probs_optical = None
    probs_gw = None
    probs_hard = None
    
    if triplet_logits.get("logits_positive") is not None:
        probs_pos = torch.softmax(triplet_logits["logits_positive"].float(), dim=1)[:, 1].numpy()
    optical_logits = triplet_logits.get("logits_optical_neg")
    if optical_logits is not None:
        probs_optical = torch.softmax(optical_logits.float(), dim=1)[:, 1].numpy()
    if triplet_logits.get("logits_gw_neg") is not None:
        probs_gw = torch.softmax(triplet_logits["logits_gw_neg"].float(), dim=1)[:, 1].numpy()
    if triplet_logits.get("logits_hard_neg") is not None:
        probs_hard = torch.softmax(triplet_logits["logits_hard_neg"].float(), dim=1)[:, 1].numpy()
    
    # --- 1. Main distribution plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 51)
    alpha = 0.6
    
    if probs_pos is not None:
        ax.hist(probs_pos, bins=bins, alpha=alpha, label=f'Positive (GW, KN) n={len(probs_pos)}', 
                edgecolor='#2ecc71', linewidth=3, histtype='step')
    if probs_optical is not None:
        ax.hist(probs_optical, bins=bins, alpha=alpha, label=f'Optical Negatives (GW, nonKN) n={len(probs_optical)}',
                edgecolor='#3498db', linewidth=3, histtype='step')
    if probs_gw is not None:
        ax.hist(probs_gw, bins=bins, alpha=alpha, label=f'GW Negatives (GW_has_kn0, KN) n={len(probs_gw)}',
                edgecolor='#f39c12', linewidth=3, histtype='step')
    if probs_hard is not None:
        ax.hist(probs_hard, bins=bins, alpha=alpha, label=f'Semi-Hard Negatives (GW_wrong, KN) n={len(probs_hard)}',
                edgecolor='#e74c3c', linewidth=3, histtype='step')
    
    ax.set_xlabel('Match Probability (Softmax Output)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Classification Head Logits Distribution\nby Sample Pair Type', fontsize=14)
    ax.legend(loc='upper center', fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add vertical line at 0.5 threshold
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold=0.5')
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "logits_distribution_triplet.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    
    # --- 2. KDE density plot for better visualization ---
    try:
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_range = np.linspace(0, 1, 200)
        
        if probs_pos is not None and len(probs_pos) > 1:
            kde_pos = stats.gaussian_kde(probs_pos, bw_method=0.05)
            ax.plot(x_range, kde_pos(x_range), color='#27ae60', linewidth=2.5,
                   label=f'Positive (GW, KN)')
            
        if probs_optical is not None and len(probs_optical) > 1:
            kde_optical = stats.gaussian_kde(probs_optical, bw_method=0.05)
            ax.plot(x_range, kde_optical(x_range), color='#2980b9', linewidth=2.5,
                   label='Optical Negatives (GW, nonKN)')

        if probs_gw is not None and len(probs_gw) > 1:
            kde_gw = stats.gaussian_kde(probs_gw, bw_method=0.05)
            ax.plot(x_range, kde_gw(x_range), color='#d68910', linewidth=2.5,
                   label='GW Negatives (GW_has_kn0, KN)')
            
        if probs_hard is not None and len(probs_hard) > 1:
            kde_hard = stats.gaussian_kde(probs_hard, bw_method=0.05)
            ax.plot(x_range, kde_hard(x_range), color='#c0392b', linewidth=2.5,
                   label='Semi-Hard Negatives (GW_wrong, KN)')
        
        ax.set_xlabel('Match Probability (Softmax Output)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Classification Head Output Distribution (KDE)\nby Sample Pair Type', fontsize=14)
        ax.legend(loc='upper center', fontsize=10)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "logits_distribution_kde.png"), dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print("scipy not available, skipping KDE plot")
    
    # Print summary statistics
    print("\n--- Logits Distribution Summary ---")
    if probs_pos is not None:
        print(f"  Positive (GW, KN):     mean={np.mean(probs_pos):.4f}  std={np.std(probs_pos):.4f}  "
              f"median={np.median(probs_pos):.4f}  n={len(probs_pos)}")
    if probs_optical is not None:
        print(f"  Optical Negatives (GW, nonKN): mean={np.mean(probs_optical):.4f}  std={np.std(probs_optical):.4f}  "
              f"median={np.median(probs_optical):.4f}  n={len(probs_optical)}")
    if probs_gw is not None:
        print(f"  GW Negatives (GW_has_kn0, KN): mean={np.mean(probs_gw):.4f}  std={np.std(probs_gw):.4f}  "
              f"median={np.median(probs_gw):.4f}  n={len(probs_gw)}")
    if probs_hard is not None:
        print(f"  Semi-Hard Negatives (GW_wrong, KN): mean={np.mean(probs_hard):.4f}  std={np.std(probs_hard):.4f}  "
              f"median={np.median(probs_hard):.4f}  n={len(probs_hard)}")
    
    print(f"Logits distribution plots saved to {output_dir}/")


def generate_gw_shuffle_comparison_plot(triplet_logits_normal, triplet_logits_shuffle, output_dir):
    """Generate comparison plot for GW-shuffle ablation test.
    
    Compares logits distributions before and after shuffling GW inputs.
    If distributions are similar after shuffle, the model doesn't use GW info effectively.
    
    Args:
        triplet_logits_normal: Dict with logits from normal forward pass
        triplet_logits_shuffle: Dict with logits from GW-shuffled forward pass
        output_dir: Directory to save plots
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats
    except ImportError:
        print("matplotlib/scipy not available, skipping GW-shuffle comparison plot")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract probabilities
    def get_probs(logits_dict, key):
        if logits_dict.get(key) is not None:
            return torch.softmax(logits_dict[key].float(), dim=1)[:, 1].numpy()
        return None

    def get_optical_neg_probs(logits_dict):
        optical_logits = logits_dict.get("logits_optical_neg")
        if optical_logits is None:
            return None
        return torch.softmax(optical_logits.float(), dim=1)[:, 1].numpy()

    probs_pos_normal = get_probs(triplet_logits_normal, "logits_positive")
    probs_optical_normal = get_optical_neg_probs(triplet_logits_normal)
    probs_gw_normal = get_probs(triplet_logits_normal, "logits_gw_neg")
    probs_hard_normal = get_probs(triplet_logits_normal, "logits_hard_neg")

    probs_pos_shuffle = get_probs(triplet_logits_shuffle, "logits_positive")
    probs_optical_shuffle = get_optical_neg_probs(triplet_logits_shuffle)
    probs_gw_shuffle = get_probs(triplet_logits_shuffle, "logits_gw_neg")
    probs_hard_shuffle = get_probs(triplet_logits_shuffle, "logits_hard_neg")

    # --- Combined KDE comparison plot ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    x_range = np.linspace(0, 1, 200)
    
    pair_types = [
        ('Positive (GW, KN)', probs_pos_normal, probs_pos_shuffle, '#27ae60', '#2ecc71'),
        ('Optical Negatives (GW, nonKN)', probs_optical_normal, probs_optical_shuffle, '#2980b9', '#3498db'),
        ('GW Negatives (GW_has_kn0, KN)', probs_gw_normal, probs_gw_shuffle, '#d68910', '#f39c12'),
        ('Semi-Hard Negatives (GW_wrong, KN)', probs_hard_normal, probs_hard_shuffle, '#c0392b', '#e74c3c'),
    ]
    
    for ax, (title, probs_n, probs_s, color_n, color_s) in zip(axes, pair_types):
        if probs_n is not None and len(probs_n) > 1:
            kde_n = stats.gaussian_kde(probs_n, bw_method=0.05)
            ax.plot(x_range, kde_n(x_range), color=color_n, linewidth=2.5,
                   label=f'Normal (μ={np.mean(probs_n):.3f})')
        
        if probs_s is not None and len(probs_s) > 1:
            kde_s = stats.gaussian_kde(probs_s, bw_method=0.05)
            ax.plot(x_range, kde_s(x_range), color=color_s, linewidth=2.5, linestyle='--',
                   label=f'GW-Shuffle (μ={np.mean(probs_s):.3f})')
        
        ax.set_xlabel('Match Probability', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='black', linestyle=':', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.suptitle('GW-Shuffle Ablation Test: Effect of Randomizing GW Input', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gw_shuffle_comparison.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    
    # --- All distributions on one plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if probs_pos_normal is not None and len(probs_pos_normal) > 1:
        kde = stats.gaussian_kde(probs_pos_normal, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#27ae60', linewidth=2.5,
               label='Positive - Normal')
    if probs_pos_shuffle is not None and len(probs_pos_shuffle) > 1:
        kde = stats.gaussian_kde(probs_pos_shuffle, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#27ae60', linewidth=2.5, linestyle='--',
               label='Positive - Shuffle')
    
    if probs_optical_normal is not None and len(probs_optical_normal) > 1:
        kde = stats.gaussian_kde(probs_optical_normal, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#2980b9', linewidth=2.5,
               label='Optical Negatives - Normal')
    if probs_optical_shuffle is not None and len(probs_optical_shuffle) > 1:
        kde = stats.gaussian_kde(probs_optical_shuffle, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#2980b9', linewidth=2.5, linestyle='--',
               label='Optical Negatives - Shuffle')

    if probs_gw_normal is not None and len(probs_gw_normal) > 1:
        kde = stats.gaussian_kde(probs_gw_normal, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#d68910', linewidth=2.5,
               label='GW Negatives - Normal')
    if probs_gw_shuffle is not None and len(probs_gw_shuffle) > 1:
        kde = stats.gaussian_kde(probs_gw_shuffle, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#d68910', linewidth=2.5, linestyle='--',
               label='GW Negatives - Shuffle')
    
    if probs_hard_normal is not None and len(probs_hard_normal) > 1:
        kde = stats.gaussian_kde(probs_hard_normal, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#c0392b', linewidth=2.5,
               label='Semi-Hard Negatives - Normal')
    if probs_hard_shuffle is not None and len(probs_hard_shuffle) > 1:
        kde = stats.gaussian_kde(probs_hard_shuffle, bw_method=0.05)
        ax.plot(x_range, kde(x_range), color='#c0392b', linewidth=2.5, linestyle='--',
               label='Semi-Hard Negatives - Shuffle')
    
    ax.set_xlabel('Match Probability (Softmax Output)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('GW-Shuffle Ablation: All Distributions Comparison\n(Solid=Normal, Dashed=GW-Shuffled)', fontsize=14)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "gw_shuffle_all_distributions.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    
    # Print comparison statistics
    print("\n" + "=" * 70)
    print("GW-SHUFFLE ABLATION TEST RESULTS")
    print("=" * 70)
    print("\nIf distributions are similar after shuffle, model doesn't use GW info.")
    print("If Positive (green) drops significantly, model IS using GW info.\n")
    
    def print_comparison(name, probs_n, probs_s):
        if probs_n is not None and probs_s is not None:
            delta_mean = np.mean(probs_s) - np.mean(probs_n)
            delta_std = np.std(probs_s) - np.std(probs_n)
            # KS test for distribution difference
            ks_stat, ks_pval = stats.ks_2samp(probs_n, probs_s)
            print(f"  {name}:")
            print(f"    Normal:  mean={np.mean(probs_n):.4f}  std={np.std(probs_n):.4f}")
            print(f"    Shuffle: mean={np.mean(probs_s):.4f}  std={np.std(probs_s):.4f}")
            print(f"    Δmean={delta_mean:+.4f}  Δstd={delta_std:+.4f}")
            print(f"    KS-test: stat={ks_stat:.4f}  p-value={ks_pval:.2e}")
            if ks_pval < 0.05:
                print(f"    → Distributions are SIGNIFICANTLY DIFFERENT (p<0.05)")
            else:
                print(f"    → Distributions are NOT significantly different (p≥0.05)")
            print()
    
    print_comparison("Positive (GW, KN)", probs_pos_normal, probs_pos_shuffle)
    print_comparison("Optical Negatives (GW, nonKN)", probs_optical_normal, probs_optical_shuffle)
    print_comparison("GW Negatives (GW_has_kn0, KN)", probs_gw_normal, probs_gw_shuffle)
    print_comparison("Semi-Hard Negatives (GW_wrong, KN)", probs_hard_normal, probs_hard_shuffle)
    
    print("=" * 70)
    print(f"GW-shuffle comparison plots saved to {output_dir}/")


def save_results(results, output_dir):
    """Save results as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    results_clean = convert(results)
    path = os.path.join(output_dir, "eval_results.json")
    with open(path, "w") as f:
        json.dump(results_clean, f, indent=2)
    print(f"Results saved to {path}")


def print_summary(results):
    """Print a concise summary of all results."""
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    # Retrieval (batch mode)
    ret = results.get("retrieval_batch", {})
    if ret:
        print("\n--- Retrieval (Batch Mode) ---")
        print(f"  GW→Opt  R@1={ret.get('g2o_recall_at_1', 0):.4f}  "
              f"R@5={ret.get('g2o_recall_at_5', 0):.4f}  "
              f"R@10={ret.get('g2o_recall_at_10', 0):.4f}  "
              f"MRR={ret.get('g2o_mrr', 0):.4f}  "
              f"mAP={ret.get('g2o_map', 0):.4f}")
        print(f"  Opt→GW  R@1={ret.get('o2g_recall_at_1', 0):.4f}  "
              f"R@5={ret.get('o2g_recall_at_5', 0):.4f}  "
              f"MRR={ret.get('o2g_mrr', 0):.4f}")

    # Gallery mode
    gal = results.get("retrieval_gallery", {})
    if gal:
        print("\n--- Retrieval (Gallery Mode) ---")
        sizes = sorted(set(
            int(k.split("_")[1]) for k in gal if "recall_at_1" in k
        ))
        for s in sizes:
            r1 = gal.get(f"gallery_{s}_recall_at_1", 0)
            r5 = gal.get(f"gallery_{s}_recall_at_5", 0)
            mrr = gal.get(f"gallery_{s}_mrr", 0)
            print(f"  Pool={s:5d}  R@1={r1:.4f}  R@5={r5:.4f}  MRR={mrr:.4f}")

    td_meta = results.get("meta", {}).get("time_delta", {})
    dt_dist = td_meta.get("distributions", {}) if isinstance(td_meta, dict) else {}
    if dt_dist:
        print("\n--- Time-Delta Distributions ---")
        for key in ("positive", "optical_negative", "gw_negative", "hard_negative"):
            entry = dt_dist.get(key, {})
            if not entry:
                continue
            q = entry.get("quantiles_days", {})
            abs_q = entry.get("abs", {}).get("quantiles_days", {})
            print(
                f"  {key}: n={entry.get('count',0)} "
                f"mean={entry.get('mean_days',0):.3f} "
                f"std={entry.get('std_days',0):.3f} "
                f"p50={q.get('p50',0):.3f} "
                f"| abs p50={abs_q.get('p50',0):.3f} "
                f"p90={abs_q.get('p90',0):.3f} "
                f"p95={abs_q.get('p95',0):.3f}"
            )

    # Classification
    cls = results.get("classification", {})
    if cls:
        print("\n--- Classification ---")
        print(f"  AUROC={cls.get('auroc', 0):.4f}  "
              f"AUPRC={cls.get('auprc', 0):.4f}  "
              f"F1={cls.get('f1_optimal', 0):.4f} (t={cls.get('f1_threshold', 0):.2f})  "
              f"ECE={cls.get('ece', 0):.4f}")
        tp = cls.get('tp', 0)
        fp = cls.get('fp', 0)
        tn = cls.get('tn', 0)
        fn = cls.get('fn', 0)
        print(f"  Confusion (t=0.5): TP={tp} FP={fp} TN={tn} FN={fn}")
        dt_bins = cls.get("dt_bins", {})
        dt_cov = cls.get("dt_bins_coverage", {})
        if dt_cov:
            print(
                "  |dt| coverage: "
                f"{dt_cov.get('n_with_dt',0)}/{dt_cov.get('n_total',0)} "
                f"({dt_cov.get('coverage_ratio',0.0):.2%})"
            )
        if dt_bins:
            print("  |dt| bins:")
            for b in dt_bins.get("bins", []):
                if int(b.get("n", 0)) <= 0:
                    continue
                if "auroc" in b and "auprc" in b:
                    print(
                        f"    {b.get('range','?')}: n={b.get('n',0)} "
                        f"(pos={b.get('n_pos',0)}, neg={b.get('n_neg',0)}) "
                        f"AUROC={b.get('auroc',0):.4f} AUPRC={b.get('auprc',0):.4f}"
                    )
                else:
                    print(
                        f"    {b.get('range','?')}: n={b.get('n',0)} "
                        f"(pos={b.get('n_pos',0)}, neg={b.get('n_neg',0)}) "
                        f"Precision={b.get('precision',0):.4f} "
                        f"Recall={b.get('recall',0):.4f} "
                        f"F1={b.get('f1_confusion',0):.4f}"
                    )
        dt_macro = cls.get("dt_macro", {})
        if dt_macro:
            print(
                "  |dt| macro: "
                f"AUROC={dt_macro.get('auroc',0):.4f} "
                f"AUPRC={dt_macro.get('auprc',0):.4f} "
                f"F1={dt_macro.get('f1_optimal',0):.4f} "
                f"(bins_used={dt_macro.get('bins_used',0)}/{dt_macro.get('bins_total',0)})"
            )

    cls_shuffle = results.get("ablation", {}).get("gw_shuffle", {}).get("classification", {})
    if cls_shuffle:
        print("\n--- Classification (GW-Shuffle) ---")
        print(f"  AUROC={cls_shuffle.get('auroc', 0):.4f}  "
              f"AUPRC={cls_shuffle.get('auprc', 0):.4f}  "
              f"F1={cls_shuffle.get('f1_optimal', 0):.4f} (t={cls_shuffle.get('f1_threshold', 0):.2f})  "
              f"ECE={cls_shuffle.get('ece', 0):.4f}")

    cls_by_src = results.get("classification_by_source", {})
    if cls_by_src:
        print("\n--- Classification by Source ---")
        for src in sorted(cls_by_src):
            src_block = cls_by_src.get(src, {})
            src_metrics = src_block.get("metrics", {})
            src_counts = src_block.get("pair_type_counts", {})
            print(f"  [{src}] AUROC={src_metrics.get('auroc', 0):.4f}  "
                  f"AUPRC={src_metrics.get('auprc', 0):.4f}  "
                  f"F1={src_metrics.get('f1_optimal', 0):.4f} "
                  f"(t={src_metrics.get('f1_threshold', 0):.2f})  "
                  f"ECE={src_metrics.get('ece', 0):.4f}")
            print(f"    Confusion@0.5: Precision={src_metrics.get('precision_confusion', 0):.4f}  "
                  f"Recall={src_metrics.get('recall_confusion', 0):.4f}  "
                  f"F1={src_metrics.get('f1_confusion', 0):.4f}")
            print(f"    n_total={src_counts.get('n_total', 0)} "
                  f"(pos={src_counts.get('n_positive', 0)}, "
                  f"optical_neg={src_counts.get('n_optical_neg', 0)}, "
                  f"gw_neg={src_counts.get('n_gw_neg', 0)}, "
                  f"hard_neg={src_counts.get('n_hard_neg', 0)})")

    # Embedding
    emb = results.get("embedding", {})
    if emb:
        print("\n--- Embedding Quality ---")
        print(f"  Alignment={emb.get('alignment', 0):.4f}  "
              f"Uniformity(GW)={emb.get('uniformity_gw', 0):.4f}  "
              f"Uniformity(Opt)={emb.get('uniformity_opt', 0):.4f}")
        print(f"  Inter-modal gap={emb.get('inter_modal_gap', 0):.4f}  "
              f"Intra-sim={emb.get('intra_sim', 0):.4f}  "
              f"Inter-sim={emb.get('inter_sim', 0):.4f}")

    print("=" * 70)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gallery_sizes = [int(s) for s in args.gallery_sizes.split(",")]
    amp_dtype, amp_enabled = _resolve_eval_amp(args.amp_dtype, device)
    if device.type == "cuda":
        if amp_enabled:
            print(f"Evaluation precision: AMP enabled ({_amp_dtype_name(amp_dtype)})")
        else:
            print("Evaluation precision: fp32 (AMP disabled)")
    else:
        print("Evaluation precision: fp32 (CPU)")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model, model_args, saved_args = load_model(args, device)
    runtime_model_args = dict(model_args)
    dataset_window_start, dataset_window_end = _read_root_time_window_attrs(args.test_data_path)
    runtime_model_args["dataset_window_metadata"] = {
        "positive": _read_root_time_window_attrs(args.test_data_path),
        "negative": _read_root_time_window_attrs(args.neg_data_path),
    }
    runtime_model_args["effective_input_window_metadata"] = build_effective_input_window_metadata(
        float(runtime_model_args.get("ref_start", -0.3)),
        float(runtime_model_args.get("ref_end", 0.6)),
        runtime_input_window_start=float(runtime_model_args.get("ref_start", -0.3)),
        runtime_input_window_end=float(runtime_model_args.get("ref_end", 0.6)),
        dataset_window_start=dataset_window_start,
        dataset_window_end=dataset_window_end,
    )
    nonkn_cls_base_field = str(model_args.get("nonkn_cls_base_field", "zero_time_mjd_cls_base"))
    neg_time_offset_enable = bool(
        choose_value(
            args.neg_time_offset_enable, saved_args, "neg_time_offset_enable", default=False
        )
    )
    neg_offset_dist_npz = choose_value(
        args.neg_offset_dist_npz, saved_args, "neg_offset_dist_npz", default=None
    )
    neg_offset_dist_key = str(
        choose_value(args.neg_offset_dist_key, saved_args, "neg_offset_dist_key", default="delta_days_combined")
    )
    neg_offset_eval_mode = str(
        choose_value(args.neg_offset_eval_mode, saved_args, "neg_offset_eval_mode", default="quantile_ensemble")
    )
    neg_offset_eval_quantiles = str(
        choose_value(
            args.neg_offset_eval_quantiles,
            saved_args,
            "neg_offset_eval_quantiles",
            default="0.1,0.3,0.5,0.7,0.9",
        )
    )
    neg_offset_scale_days_divisor = float(
        choose_value(
            args.neg_offset_scale_days_divisor,
            saved_args,
            "neg_offset_scale_days_divisor",
            default=100.0,
        )
    )
    report_dt_bins = bool(
        choose_value(args.report_dt_bins, saved_args, "report_dt_bins", default=True)
    )
    report_dt_macro = bool(
        choose_value(args.report_dt_macro, saved_args, "report_dt_macro", default=True)
    )
    dt_bin_edges_text = str(
        choose_value(
            args.dt_bin_edges,
            saved_args,
            "dt_bin_edges",
            default="0,30,90,180,365,730,1460,inf",
        )
    )
    dt_bin_edges = parse_dt_bin_edges(dt_bin_edges_text)
    hardneg_semi_hard = bool(choose_value(None, saved_args, "semi_hard", default=True))
    hardneg_semi_hard_margin = float(choose_value(None, saved_args, "semi_hard_margin", default=0.2))
    hardneg_fallback_mode = str(
        choose_value(None, saved_args, "hardneg_fallback_mode", default="inbatch_semihard")
    )
    hardneg_windows_text = str(
        choose_value(
            None,
            saved_args,
            "hardneg_time_window_days",
            default="30,60,120",
        )
    )
    hardneg_windows_days = parse_day_windows(hardneg_windows_text)
    hardneg_min_candidates = int(
        choose_value(
            None,
            saved_args,
            "hardneg_min_candidates",
            default=4,
        )
    )
    neg_offset_policy = EvalNegativeTimeOffsetPolicy(
        enabled=neg_time_offset_enable,
        dist_npz=neg_offset_dist_npz,
        dist_key=neg_offset_dist_key,
        eval_mode=neg_offset_eval_mode,
        eval_quantiles=neg_offset_eval_quantiles,
        scale_divisor=neg_offset_scale_days_divisor,
        seed=int(choose_value(None, saved_args, "seed", default=42)),
        bank_size=int(choose_value(None, saved_args, "neg_offset_bank_size", default=1000000)),
    )
    if neg_offset_policy.enabled:
        print("Negative time-offset policy enabled:")
    else:
        print("Negative time-offset policy disabled.")
    print(json.dumps(neg_offset_policy.info, indent=2))
    print(
        "Hard-negative mining config (from training args): "
        f"semi_hard={hardneg_semi_hard}, margin={hardneg_semi_hard_margin}, "
        f"fallback_mode={hardneg_fallback_mode}, windows={hardneg_windows_days}, "
        f"min_candidates={hardneg_min_candidates}"
    )
    gw_event_time_mjd_table = load_gw_event_time_mjd_table(
        args.test_data_path, device, required=False
    )
    if gw_event_time_mjd_table is None:
        print(
            "WARNING: 'events/gw_data/event_time_mjd' missing in eval dataset; "
            "hard-negative time-shift re-encoding will fall back to unshifted optical timelines."
        )
    need_zero_time_mjd_for_cls = bool(runtime_model_args.get("use_time_delta_cls_feature", False))

    # Build test dataloader
    loader, dataset = build_test_dataloader(
        args, saved_args, return_zero_time_mjd=need_zero_time_mjd_for_cls,
        nonkn_cls_base_field=nonkn_cls_base_field,
    )
    print(f"Test set: {len(loader)} batches")
    gw_source_types = load_gw_source_types(args.test_data_path)

    # Extract all embeddings (retry with single-worker if multiprocessing is blocked)
    try:
        embeddings = extract_all_embeddings(
            model, loader, device, runtime_model_args,
            gw_source_types=gw_source_types,
            amp_dtype=amp_dtype, amp_enabled=amp_enabled,
            gw_event_time_mjd_table=gw_event_time_mjd_table,
        )
    except PermissionError as e:
        if args.num_workers > 0:
            print("WARNING: DataLoader multiprocessing failed (PermissionError). "
                  "Retrying with num_workers=0.")
            args.num_workers = 0
            loader, dataset = build_test_dataloader(
                args, saved_args, return_zero_time_mjd=need_zero_time_mjd_for_cls,
                nonkn_cls_base_field=nonkn_cls_base_field,
            )
            embeddings = extract_all_embeddings(
                model, loader, device, runtime_model_args,
                gw_source_types=gw_source_types,
                amp_dtype=amp_dtype, amp_enabled=amp_enabled,
                gw_event_time_mjd_table=gw_event_time_mjd_table,
            )
        else:
            raise
    n_samples = len(embeddings["feat_g"])
    n_unique_gw = len(torch.unique(embeddings["gw_indices"]))
    print(f"Extracted {n_samples} samples from {n_unique_gw} unique GW events")

    # Compute all metrics
    results = {}
    results["meta"] = {
        "time_offset": neg_offset_policy.info,
        "time_delta": {},
        "dt_bins": {
            "enabled": bool(report_dt_bins),
            "macro_enabled": bool(report_dt_macro),
            "edges_days": [float(e) if np.isfinite(e) else "inf" for e in dt_bin_edges],
        },
        "hard_negative_mining": {
            "semi_hard": bool(hardneg_semi_hard),
            "semi_hard_margin": float(hardneg_semi_hard_margin),
            "fallback_mode": str(hardneg_fallback_mode),
            "windows_days": [float(v) for v in hardneg_windows_days],
            "min_candidates": int(hardneg_min_candidates),
        },
        "dataset_window_metadata": runtime_model_args.get("dataset_window_metadata"),
        "effective_input_window_metadata": runtime_model_args.get("effective_input_window_metadata"),
    }
    print("\nComputing batch-mode retrieval metrics...")
    results["retrieval_batch"] = evaluate_retrieval_batch_mode(embeddings)

    print("Computing gallery-mode retrieval metrics...")
    _gallery_result = evaluate_retrieval_gallery_mode(
        model, embeddings, gallery_sizes, args.test_data_path,
        device=device, n_trials=args.gallery_trials,
        model_args=runtime_model_args,
        gw_event_time_mjd_table=gw_event_time_mjd_table,
        amp_dtype=amp_dtype, amp_enabled=amp_enabled,
        gw_source_types=gw_source_types,
    )
    _gallery_by_source = _gallery_result.pop("by_source", {})
    results["retrieval_gallery"] = _gallery_result
    if _gallery_by_source:
        results["retrieval_gallery_by_source"] = _gallery_by_source

    print("Computing embedding quality metrics...")
    results["embedding"] = evaluate_embeddings(embeddings)

    # Load negative optical samples for triplet logits analysis
    neg_optical_data = None
    if args.neg_data_path and os.path.exists(args.neg_data_path):
        neg_optical_data = load_negative_optical_samples(
            args.neg_data_path, 
            args.neg_group, 
            n_samples=args.n_neg_samples,
            require_zero_time_mjd_base=False,
            require_zero_time_mjd_cls_base=False,
            nonkn_cls_base_field=nonkn_cls_base_field,
            runtime_input_window_start=float(runtime_model_args.get("ref_start", -0.3)),
            runtime_input_window_end=float(runtime_model_args.get("ref_end", 0.6)),
        )
    neg_gw_indices = load_negative_gw_indices(args.test_data_path)

    def _extract_triplets_with_retry(*, shuffle_gw=False):
        loader_local, _ = build_test_dataloader(
            args, saved_args, return_zero_time_mjd=need_zero_time_mjd_for_cls,
            nonkn_cls_base_field=nonkn_cls_base_field,
        )
        try:
            return extract_triplet_logits(
                model, loader_local, device, runtime_model_args, neg_optical_data, neg_gw_indices,
                gw_source_types=gw_source_types,
                shuffle_gw=shuffle_gw, shuffle_seed=42,
                amp_dtype=amp_dtype, amp_enabled=amp_enabled,
                neg_offset_policy=neg_offset_policy,
                gw_event_time_mjd_table=gw_event_time_mjd_table,
                hardneg_windows_days=hardneg_windows_days,
                hardneg_min_candidates=hardneg_min_candidates,
                hardneg_semi_hard=hardneg_semi_hard,
                hardneg_semi_hard_margin=hardneg_semi_hard_margin,
                hardneg_fallback_mode=hardneg_fallback_mode,
            )
        except PermissionError:
            if args.num_workers <= 0:
                raise
            print("WARNING: DataLoader multiprocessing failed (PermissionError). Retrying with num_workers=0.")
            args.num_workers = 0
            loader_local, _ = build_test_dataloader(
                args, saved_args, return_zero_time_mjd=need_zero_time_mjd_for_cls,
                nonkn_cls_base_field=nonkn_cls_base_field,
            )
            return extract_triplet_logits(
                model, loader_local, device, runtime_model_args, neg_optical_data, neg_gw_indices,
                gw_source_types=gw_source_types,
                shuffle_gw=shuffle_gw, shuffle_seed=42,
                amp_dtype=amp_dtype, amp_enabled=amp_enabled,
                neg_offset_policy=neg_offset_policy,
                gw_event_time_mjd_table=gw_event_time_mjd_table,
                hardneg_windows_days=hardneg_windows_days,
                hardneg_min_candidates=hardneg_min_candidates,
                hardneg_semi_hard=hardneg_semi_hard,
                hardneg_semi_hard_margin=hardneg_semi_hard_margin,
                hardneg_fallback_mode=hardneg_fallback_mode,
            )

    # Extract triplet logits for distribution analysis
    print("\nExtracting triplet logits for distribution analysis...")
    triplet_logits = _extract_triplets_with_retry(shuffle_gw=False)

    print("\nExtracting triplet logits with GW-SHUFFLE (ablation test)...")
    triplet_logits_shuffle = _extract_triplets_with_retry(shuffle_gw=True)

    # Recompute classification from triplet logits (pos + optical_neg + gw_neg + semi_hard_neg)
    if triplet_logits is not None:
        td_meta = triplet_logits.get("time_delta_meta")
        if td_meta is not None:
            results["meta"]["time_delta"].update(td_meta)
        dt_distributions = {}
        dt_pos = triplet_logits.get("dt_positive_days")
        dt_opt = triplet_logits.get("dt_optical_negative_days")
        dt_gw = triplet_logits.get("dt_gw_negative_days")
        dt_hard = triplet_logits.get("dt_hard_negative_days")
        for key, tensor in (
            ("positive", dt_pos),
            ("optical_negative", dt_opt),
            ("gw_negative", dt_gw),
            ("hard_negative", dt_hard),
        ):
            summary = _summarize_dt_distribution(tensor)
            if summary:
                dt_distributions[key] = summary
        if dt_distributions:
            results["meta"]["time_delta"]["distributions"] = dt_distributions
        print("\nRecomputing classification metrics from triplet pairs...")
        triplet_cls = evaluate_classification_triplet(
            triplet_logits,
            report_dt_bins=report_dt_bins,
            dt_bin_edges=dt_bin_edges,
            report_dt_macro=report_dt_macro,
        )
        if triplet_cls:
            results["classification"] = triplet_cls
        triplet_cls_by_source = evaluate_classification_triplet_by_source(triplet_logits)
        if triplet_cls_by_source:
            results["classification_by_source"] = triplet_cls_by_source
        results.setdefault("ablation", {})
        if triplet_logits_shuffle is not None:
            shuffle_cls = evaluate_classification_triplet(
                triplet_logits_shuffle,
                report_dt_bins=report_dt_bins,
                dt_bin_edges=dt_bin_edges,
                report_dt_macro=report_dt_macro,
            )
            shuffle_src = evaluate_classification_triplet_by_source(triplet_logits_shuffle)
            results["ablation"]["gw_shuffle"] = {
                "classification": shuffle_cls,
                "classification_by_source": shuffle_src if shuffle_src else {},
                "distribution_summary": summarize_triplet_prob_distributions(triplet_logits_shuffle),
                "delta_vs_baseline": _metric_delta(shuffle_cls or {}, triplet_cls or {}),
            }

    # Save and display
    save_results(results, args.output_dir)
    print_summary(results)

    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(embeddings, results, args.output_dir, triplet_logits=triplet_logits)
        
        # Generate triplet logits distribution plot
        if triplet_logits is not None:
            print("\nGenerating logits distribution plots...")
            generate_logits_distribution_plot(triplet_logits, args.output_dir)
        
        # Generate GW-shuffle comparison plot
        if triplet_logits is not None and triplet_logits_shuffle is not None:
            print("\nGenerating GW-shuffle comparison plots...")
            generate_gw_shuffle_comparison_plot(
                triplet_logits, triplet_logits_shuffle, args.output_dir
            )


if __name__ == "__main__":
    "python test_evaluate.py --checkpoint ${BASE_DIR}/data/model/checkpoints/supcon_v1/ALBEF/albef_best.pth --test_data_path ${BASE_DIR}/data/LSST_KN_BNS/combined_dataset_with_neg_gw.h5 --output_dir eval_results"
    main()
