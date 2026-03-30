import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

nb = new_notebook()

nb.cells.append(new_markdown_cell("# Attention Distribution for Typical Lightcurves (Optical Only Model)"))

code1 = '''import os
_BASE = os.environ.get('BASE_DIR', '/fred/oz016/bgao_kn')

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from contextlib import nullcontext

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from torch.amp import autocast

BANDS = ("u", "g", "r", "i", "z", "Y")
BAND_COLORS = {
    "u": "#3B82F6",
    "g": "#10B981",
    "r": "#EF4444",
    "i": "#F59E0B",
    "z": "#8B5CF6",
    "Y": "#6B7280",
}
BAND_COLORS_LC = {k.lower(): v for k, v in BAND_COLORS.items()}

BASE = Path(f"{_BASE}/gw-kn-multimodal")
MODEL_PY = BASE / 'Model' / 'model.py'
POS_H5 = Path(f"{_BASE}/data/Optical_Only_dataset/combined_dataset_test.h5")
NEG_H5 = Path(f"{_BASE}/data/Optical_Only_dataset/Tutorial_negative_dataset.h5")
NEG_GROUP = "Tutorial/optical_data"

OPTICAL_ONLY_CKPT = Path(f'{_BASE}/data/model/checkpoints_optical/optical_only/optical_only_kn_v16/optical_only_best.pth')
OFFSET_NPZ = Path(f'{_BASE}/data/Optical_Only_dataset/delta_days_distribution.npz')

OUTDIR = BASE / "figures" / "typical_lightcurve_attention"
OUTDIR.mkdir(parents=True, exist_ok=True)

ASINH_MAG_FACTOR = 2.5 / np.log(10.0)
PSFFLUX_ZP = 31.4
TIME_SCALE_DAYS = 100.0

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 150,
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)

print(f"Device: {device}")
'''
nb.cells.append(new_code_cell(code1))

code2 = '''# Model utils
def load_module(name: str, path: Path):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def namespace_to_dict(obj) -> dict:
    if isinstance(obj, dict): return dict(obj)
    if hasattr(obj, '__dict__'): return dict(vars(obj))
    return {}

def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

model_mod = load_module('gw_kn_model_compare_notebook', MODEL_PY)

def load_optical_only_bundle(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = namespace_to_dict(ckpt.get('args', {}))
    state_dict = clean_state_dict(ckpt['model_state_dict'])
    state_dict = model_mod.migrate_time_embed_state_dict(state_dict)
    has_universal_aux = (
        bool(saved_args.get('universal_train_enable', False))
        or any(key.startswith('projection_head.') for key in state_dict)
        or any(key.startswith('adv_head_n_det.') for key in state_dict)
    )
    head_hidden_dim = saved_args.get('head_hidden_dim', None)
    enc_dim = int(saved_args.get('enc_dim', 128))
    model = model_mod.OpticalKNClassifier(
        optical_input_dim=6,
        ref_time_dim=int(saved_args.get('ref_dim', 64)),
        enc_dim=enc_dim,
        num_heads=int(saved_args.get('num_heads', 4)),
        k_dim=int(saved_args.get('k_dim', 64)),
        opt_dropout=float(saved_args.get('opt_dropout', 0.1)),
        feature_dropout=float(saved_args.get('feature_dropout', 0.05)),
        head_hidden_dim=head_hidden_dim,
        head_dropout=float(saved_args.get('head_dropout', 0.2)),
        universal_aux_enable=bool(has_universal_aux),
        proj_dim=64,
        adv_hidden_dim=int(head_hidden_dim) if head_hidden_dim is not None else enc_dim,
        n_det_bucket_classes=5,
        n_bands_bucket_classes=4,
        t_span_bucket_classes=5,
        grl_lambda=float(saved_args.get('grl_lambda', 1.0)),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return {
        'name': 'Optical-only',
        'ckpt_path': ckpt_path,
        'model': model,
        'curve_encoder': model.optical_encoder,
        'n_ref': int(saved_args.get('n_ref', 64)),
        'ref_start': float(saved_args.get('ref_start', -0.3)),
        'ref_end': float(saved_args.get('ref_end', 0.6)),
        'offset_scale_days_divisor': float(saved_args.get('offset_scale_days_divisor', TIME_SCALE_DAYS)),
    }

OPTICAL_ONLY = load_optical_only_bundle(OPTICAL_ONLY_CKPT, device)

def load_offset_days(npz_path: Path, key: str = 'delta_days_combined', quantiles = (0.1, 0.3, 0.5, 0.7, 0.9)) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as npz:
        values = np.asarray(npz[key], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    return np.quantile(values, np.asarray(quantiles, dtype=np.float64))

OFFSET_DAYS = load_offset_days(OFFSET_NPZ)
'''
nb.cells.append(new_code_cell(code2))

code3 = '''# Typical lightcurve selection logic
@dataclass
class DatasetConfig:
    class_name: str
    h5_path: Path
    group: str
    unique_mode: Optional[str]

@dataclass
class SelectionResult:
    class_name: str
    sample_index: int
    source_label: str
    n_obs: int
    n_det_snr5: int
    n_bands: int
    typical_score: float

def decode_scalar(value) -> str:
    if isinstance(value, (bytes, np.bytes_)): return value.decode("utf-8", errors="ignore")
    return str(value)

def robust_scales(features: np.ndarray) -> np.ndarray:
    med = np.nanmedian(features, axis=0)
    mad = np.nanmedian(np.abs(features - med), axis=0)
    std = np.nanstd(features, axis=0)
    return np.where(mad > 1.0e-6, mad, np.where(std > 1.0e-6, std, 1.0)).astype(np.float64, copy=False)

def robust_scores(features: np.ndarray) -> np.ndarray:
    center = np.nanmedian(features, axis=0)
    scale = robust_scales(features)
    z = (features - center) / scale
    return np.sqrt(np.sum(np.square(z), axis=1, dtype=np.float64), dtype=np.float64)

def compute_lc_summary_for_indices(grp: h5py.Group, sample_indices: np.ndarray) -> np.ndarray:
    sorted_pos = np.argsort(sample_indices)
    sorted_indices = np.asarray(sample_indices[sorted_pos], dtype=np.int64)
    values = np.asarray(grp["values"][sorted_indices], dtype=np.float32)
    errors = np.asarray(grp["errors"][sorted_indices], dtype=np.float32)
    masks = np.asarray(grp["masks"][sorted_indices], dtype=np.float32) > 0.0
    counts = masks.sum(axis=(1, 2)).astype(np.float32)
    counts_safe = np.where(counts > 0.0, counts, 1.0)
    values_masked = np.where(masks, values, 0.0)
    errors_masked = np.where(masks, errors, 0.0)
    mean_lupt = values_masked.sum(axis=(1, 2), dtype=np.float64) / counts_safe
    sq_mean = np.square(values_masked, dtype=np.float64).sum(axis=(1, 2), dtype=np.float64) / counts_safe
    std_lupt = np.sqrt(np.maximum(0.0, sq_mean - np.square(mean_lupt, dtype=np.float64)))
    amp_lupt = (
        np.max(np.where(masks, values, -np.inf), axis=(1, 2)).astype(np.float64)
        - np.min(np.where(masks, values, np.inf), axis=(1, 2)).astype(np.float64)
    )
    mean_err = errors_masked.sum(axis=(1, 2), dtype=np.float64) / counts_safe
    summary_sorted = np.column_stack([mean_lupt, std_lupt, amp_lupt, mean_err]).astype(np.float64, copy=False)
    summary = np.empty_like(summary_sorted)
    summary[sorted_pos] = summary_sorted
    return summary

def compute_observation_and_detection_counts(h5f: h5py.File, grp: h5py.Group, sample_index: int) -> tuple[int, int]:
    values = np.asarray(grp["values"][sample_index], dtype=np.float64)
    errors = np.asarray(grp["errors"][sample_index], dtype=np.float64)
    masks = np.asarray(grp["masks"][sample_index], dtype=np.float32) > 0.0
    n_obs = int(np.count_nonzero(masks))
    if n_obs < 1: return 0, 0
    lupt_b_njy = np.asarray(h5f.attrs["lupt_b_njy"], dtype=np.float64).reshape(-1)
    psfflux_zp = float(h5f.attrs["psfflux_zp"])
    snr_threshold = float(h5f.attrs.get("snr_threshold", 5.0))
    band_grid = np.broadcast_to(np.arange(values.shape[1], dtype=np.int64), values.shape)
    valid = masks & np.isfinite(values) & np.isfinite(errors) & (errors > 0.0)
    if not np.any(valid): return n_obs, 0
    m_lupt = values[valid]
    sigma_lupt = errors[valid]
    band_idx = band_grid[valid]
    b_valid = lupt_b_njy[band_idx]
    asinh_arg = (psfflux_zp - m_lupt) / ASINH_MAG_FACTOR - np.log(b_valid)
    f_psf = 2.0 * b_valid * np.sinh(asinh_arg)
    sigma_psf = sigma_lupt * np.sqrt((f_psf * f_psf) + (2.0 * b_valid) ** 2) / ASINH_MAG_FACTOR
    snr = np.full(f_psf.shape, -np.inf, dtype=np.float64)
    finite = np.isfinite(f_psf) & np.isfinite(sigma_psf) & (sigma_psf > 0.0)
    snr[finite] = f_psf[finite] / sigma_psf[finite]
    return n_obs, int(np.count_nonzero(snr > snr_threshold))

def select_typical_subset(*, h5f: h5py.File, grp: h5py.Group, cfg: DatasetConfig, candidate_indices: np.ndarray, max_results: int) -> List[SelectionResult]:
    t_span = np.asarray(grp["meta_t_span"][:], dtype=np.float64)
    n_obs_all = np.asarray(grp["meta_n_det"][:], dtype=np.float64)
    n_bands_all = np.asarray(grp["meta_n_bands"][:], dtype=np.float64)
    
    meta_features = np.column_stack([n_obs_all[candidate_indices], n_bands_all[candidate_indices], t_span[candidate_indices]])
    meta_scores = robust_scores(meta_features)
    shortlist_local = np.argsort(meta_scores)[:max(50, max_results*10)]
    shortlist = candidate_indices[shortlist_local]
    
    lc_summary = compute_lc_summary_for_indices(grp, shortlist)
    combined = np.column_stack([meta_features[shortlist_local], lc_summary])
    combined_scores = robust_scores(combined)
    order = np.argsort(combined_scores)
    
    results = []
    seen = set()
    for pos in order:
        idx = int(shortlist[pos])
        uniq_val = int(grp[cfg.unique_mode][idx]) if cfg.unique_mode in grp else idx
        if uniq_val in seen: continue
        seen.add(uniq_val)
        
        n_obs, n_det = compute_observation_and_detection_counts(h5f, grp, idx)
        src_lb = decode_scalar(grp["types"][idx]) if "types" in grp else "KN"
        results.append(SelectionResult(cfg.class_name, idx, src_lb, n_obs, n_det, int(n_bands_all[idx]), float(combined_scores[pos])))
        if len(results) >= max_results: break
    return results

def get_examples():
    configs = {
        "positive": DatasetConfig("positive", POS_H5, "events/optical_data", "parent_event_idx"),
        "negative": DatasetConfig("negative", NEG_H5, NEG_GROUP, None),
    }
    examples = []
    for cls_name, cfg in configs.items():
        with h5py.File(cfg.h5_path, "r") as f:
            grp = f[cfg.group]
            if cls_name == "positive":
                rng = np.random.default_rng(42)
                pool = rng.choice(grp["values"].shape[0], min(grp["values"].shape[0], 2000), replace=False)
                res = select_typical_subset(h5f=f, grp=grp, cfg=cfg, candidate_indices=pool, max_results=3)
                examples.extend(res)
            else:
                types = np.asarray([decode_scalar(x) for x in grp["types"][:]])
                # pick 1 typical for each of top 3 types
                uniq_types, counts = np.unique(types, return_counts=True)
                top_types = uniq_types[np.argsort(-counts)][:3]
                for typ in top_types:
                    t_idx = np.flatnonzero(types == typ)
                    pool = np.random.default_rng(42).choice(t_idx, min(len(t_idx), 2000), replace=False)
                    res = select_typical_subset(h5f=f, grp=grp, cfg=cfg, candidate_indices=pool, max_results=1)
                    if res: res[0].source_label = typ; examples.append(res[0])
    return examples, configs

examples, configs = get_examples()
pd.DataFrame([vars(e) for e in examples])
'''
nb.cells.append(new_code_cell(code3))

code4 = '''# Attention logic
def build_ref_time(batch_size: int, n_ref: int, ref_start: float, ref_end: float, device: torch.device, dtype: torch.dtype):
    ref = torch.linspace(float(ref_start), float(ref_end), int(n_ref), dtype=dtype, device=device)
    return ref.unsqueeze(0).repeat(batch_size, 1)

def apply_time_offsets(opt_t: torch.Tensor, opt_mask: torch.Tensor, delta_days: torch.Tensor, scale_divisor: float = TIME_SCALE_DAYS) -> torch.Tensor:
    shift = (delta_days.to(device=opt_t.device, dtype=opt_t.dtype) / float(scale_divisor)).unsqueeze(1)
    valid_slots = (opt_mask.sum(dim=-1) > 0).to(dtype=opt_t.dtype)
    return opt_t + shift * valid_slots

def compute_time_attention(bundle: dict, t_scaled, values, masks, errors, offset_days: np.ndarray) -> dict:
    curve_encoder = bundle['curve_encoder']
    curve_encoder.eval()

    opt_t = torch.from_numpy(t_scaled[None, :]).to(device=device, dtype=torch.float32)
    opt_v = torch.from_numpy(values[None, :, :]).to(device=device, dtype=torch.float32)
    opt_mask = torch.from_numpy(masks[None, :, :]).to(device=device, dtype=torch.float32)
    opt_err = torch.from_numpy(errors[None, :, :]).to(device=device, dtype=torch.float32)
    ref_time = build_ref_time(1, bundle['n_ref'], bundle['ref_start'], bundle['ref_end'], device, opt_t.dtype)

    slot_valid_t = (opt_mask.sum(dim=-1) > 0)
    band_present = (opt_mask.sum(dim=1) > 0)
    
    shifted_t_stack = []
    attn_norm_stack = []

    for off_days in np.asarray(offset_days, dtype=np.float64).tolist():
        delta_days = torch.full((1,), float(off_days), device=device, dtype=torch.float32)
        shifted_t = apply_time_offsets(opt_t, opt_mask, delta_days=delta_days, scale_divisor=bundle['offset_scale_days_divisor'])
        ctx = autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type == 'cuda')) if device.type == 'cuda' else nullcontext()
        with torch.no_grad(), ctx:
            _, _, attn_weights = curve_encoder(shifted_t, opt_v, ref_time, opt_mask, errors_obs=opt_err, return_attn=True)
            
        attn_ref = attn_weights.float()[:, :, 1:, :, :]
        attn_head_mean = attn_ref.mean(dim=1)
        band_present_f = band_present[:, None, None, :].to(dtype=attn_head_mean.dtype)
        attn_band_sum = (attn_head_mean * band_present_f).sum(dim=-1)
        slot_valid_f = slot_valid_t[:, None, :].to(dtype=attn_band_sum.dtype)
        attn_band_sum = attn_band_sum * slot_valid_f
        row_sum = attn_band_sum.sum(dim=-1, keepdim=True)
        attn_norm = torch.where(row_sum > 0, attn_band_sum / row_sum, torch.zeros_like(attn_band_sum))
        
        shifted_t_stack.append(shifted_t.detach().cpu().numpy()[0])
        attn_norm_stack.append(attn_norm.detach().cpu().numpy()[0])

    shifted_t_stack = np.asarray(shifted_t_stack, dtype=np.float64)
    attn_norm_stack = np.asarray(attn_norm_stack, dtype=np.float64)
    
    attn_mean = attn_norm_stack.mean(axis=0)
    shifted_mean = shifted_t_stack.mean(axis=0)
    
    slot_valid_np = masks.sum(axis=1) > 0
    slot_band_idx = np.argmax(masks[slot_valid_np], axis=1)
    band_order = ['u','g','r','i','z','y']
    slot_band_labels = np.asarray([band_order[int(i)] for i in slot_band_idx], dtype=object)
    
    heat = attn_mean[:, slot_valid_np]
    obs_time_mean_days = shifted_mean[slot_valid_np] * TIME_SCALE_DAYS
    ref_axis_days = ref_time[0].detach().cpu().numpy().astype(np.float64) * TIME_SCALE_DAYS
    
    return {
        'heat': heat,
        'ref_axis_days': ref_axis_days,
        'obs_time_mean_days': obs_time_mean_days,
        'slot_band_labels': slot_band_labels,
    }

def centers_to_edges(centers: np.ndarray, default_half_width: float = 0.5) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 1:
        return np.asarray([centers[0] - default_half_width, centers[0] + default_half_width], dtype=np.float64)
    mids = 0.5 * (centers[:-1] + centers[1:])
    first = centers[0] - (mids[0] - centers[0])
    last = centers[-1] + (centers[-1] - mids[-1])
    return np.concatenate([[first], mids, [last]])
'''
nb.cells.append(new_code_cell(code4))

code5 = '''# Plotting setup and execution for each example
def plot_luptitude_and_attention(ex: SelectionResult, h5_path: Path, group: str):
    with h5py.File(h5_path, "r") as f:
        grp = f[group]
        idx = ex.sample_index
        t_scaled = np.asarray(grp["times"][idx], dtype=np.float32)
        values = np.asarray(grp["values"][idx], dtype=np.float32)
        errors = np.asarray(grp["errors"][idx], dtype=np.float32)
        masks = np.asarray(grp["masks"][idx], dtype=np.float32)

    attn_res = compute_time_attention(OPTICAL_ONLY, t_scaled, values, masks, errors, OFFSET_DAYS)
    
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5], hspace=0.3)
    ax_lupt = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    
    times = t_scaled * TIME_SCALE_DAYS
    for band_idx, band_name in enumerate(BANDS):
        valid = masks[:, band_idx] > 0
        if not np.any(valid): continue
        x = times[valid]
        y = values[valid, band_idx]
        yerr = errors[valid, band_idx]
        sort_idx = np.argsort(x)
        c = BAND_COLORS[band_name]
        ax_lupt.errorbar(x[sort_idx], y[sort_idx], yerr=yerr[sort_idx],
                         fmt="o", ms=4, capsize=2, color=c, label=band_name)

    ax_lupt.invert_yaxis()
    ax_lupt.axvline(0.0, color="#111827", lw=0.9, ls="--", alpha=0.7)
    ax_lupt.grid(True, alpha=0.22, lw=0.6)
    ax_lupt.set_title(f"{ex.source_label} | idx={ex.sample_index} | Ndet={ex.n_det_snr5} Nbands={ex.n_bands}")
    ax_lupt.set_ylabel("Luptitude")
    ax_lupt.legend(loc="upper right", ncol=6, frameon=False)
    
    heat = attn_res['heat']
    vmax = max(float(np.nanpercentile(heat, 99)), 1.0e-6)
    obs_edges = np.arange(0.5, heat.shape[1] + 1.5, 1.0)
    ref_edges = centers_to_edges(attn_res['ref_axis_days'], default_half_width=0.5 * (TIME_SCALE_DAYS / max(1, len(attn_res['ref_axis_days'])-1)))
    
    mesh = ax_heat.pcolormesh(obs_edges, ref_edges, heat, shading='auto', cmap='magma', vmin=0.0, vmax=vmax)
    obs_point_idx = np.arange(1, heat.shape[1] + 1)
    ax_heat.set_xticks(obs_point_idx)
    ax_heat.set_xticklabels([f'{i}-{b}\n{t:+.1f}d' for i, b, t in zip(obs_point_idx, attn_res['slot_band_labels'], attn_res['obs_time_mean_days'])], fontsize=8)
    ax_heat.set_xlim(0.5, heat.shape[1] + 0.5)
    ax_heat.set_xlabel("Observation Point + Band")
    ax_heat.set_ylabel("Reference Time [days]")
    ax_heat.grid(which='major', axis='y', alpha=0.12, linestyle='--', linewidth=0.7)
    
    cb = fig.colorbar(mesh, ax=ax_heat, fraction=0.03, pad=0.02)
    cb.set_label("Attention Weight")
    
    out_path = OUTDIR / f"attention_{ex.source_label}_idx{ex.sample_index}.png"
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return str(out_path)

output_files = []
for ex in examples:
    cfg = configs[ex.class_name]
    output_files.append(plot_luptitude_and_attention(ex, cfg.h5_path, cfg.group))

print("Saved attention plots:")
for f in output_files: print(f)
'''
nb.cells.append(new_code_cell(code5))

with open('/fred/oz016/bgao_kn/gw-kn-multimodal/plots_scripts/plot-example-luptitude-attention.ipynb', 'w') as f:
    nbf.write(nb, f)
