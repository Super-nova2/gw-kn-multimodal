#!/usr/bin/env python3
"""
Build NEW optical-only datasets (HDF5) with unified time-zero at first detection.

This script is intentionally independent from existing ALBEF preprocessing scripts.
It does not modify existing code paths; it creates new H5 artifacts for optical-only.

First-detection rule:
1) Prefer PHOTFLAG != 0
2) Fallback to SNR > 5 (FLUXCAL / FLUXCALERR)
3) If still no detection in a realization, drop that realization

Saved time vector:
    times = (MJD - t0) / 100
where t0 is first-detection MJD (+ optional fixed offset in days).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from astropy.io import fits
from tqdm import tqdm


NUM_BANDS = 6
MAX_LC_LENGTH = 200
LUPT_BAND_ORDER = ("u", "g", "r", "i", "z", "Y")
ASINH_MAG_FACTOR = 2.5 / np.log(10.0)
DEFAULT_TIME_WINDOW_START = -0.3
DEFAULT_TIME_WINDOW_END = 0.6
DEFAULT_DENSITY_BINS_N_DET = (3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 80.0, 200.0)
DEFAULT_DENSITY_BINS_N_BANDS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
DEFAULT_DENSITY_BINS_T_SPAN = (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
DEFAULT_ZERO_TIME_WINDOW_START = 6100.0
DEFAULT_ZERO_TIME_WINDOW_END = 64500.0

BAND_TO_INDEX = {
    "LSST-u": 0,
    "LSST-g": 1,
    "LSST-r": 2,
    "LSST-i": 3,
    "LSST-z": 4,
    "LSST-Y": 5,
    "u": 0,
    "g": 1,
    "r": 2,
    "i": 3,
    "z": 4,
    "Y": 5,
}

MJD_EXPLODE_PATTERN = re.compile(r"MJD_EXPLODE:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")


def parse_bool_arg(text, *, name: str) -> bool:
    if isinstance(text, bool):
        return bool(text)
    raw = str(text).strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean-like value, got: {text!r}")


def parse_bin_edges(text: Optional[str], default_vals: Sequence[float], *, name: str) -> np.ndarray:
    if text is None:
        vals = np.asarray(default_vals, dtype=np.float64)
    else:
        parts = [p.strip() for p in str(text).split(",")]
        vals = np.asarray([float(p) for p in parts if p != ""], dtype=np.float64)
    if vals.size == 0:
        raise ValueError(f"{name} must contain at least one numeric value.")
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"{name} must contain finite numeric values.")
    vals = np.unique(np.sort(vals))
    return vals.astype(np.float64, copy=False)


def deterministic_uniform_sample(low: float, high: float, key: str) -> float:
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise ValueError(f"Invalid sampling window: [{low}, {high}]")
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    u64 = int.from_bytes(digest, byteorder="big", signed=False)
    u = float(u64) / float((1 << 64) - 1)
    return float(low + (high - low) * u)


def compute_meta_features_from_formatted(
    mask_mat: np.ndarray,
    time_vec: np.ndarray,
) -> Tuple[int, int, float, int]:
    mask = np.asarray(mask_mat, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError(f"mask_mat must be 2D [T,B], got shape={mask.shape}")
    if mask.shape[1] != NUM_BANDS:
        raise ValueError(f"mask_mat second dim must be {NUM_BANDS}, got {mask.shape[1]}")

    mask_bin = mask > 0
    n_det = int(mask_bin.sum())
    band_hits = mask_bin.sum(axis=0) > 0
    n_bands = int(band_hits.sum())
    single_band_id = int(np.argmax(band_hits)) if n_bands == 1 else -1

    slot_valid = mask_bin.sum(axis=1) > 0
    if np.any(slot_valid):
        tv = np.asarray(time_vec, dtype=np.float32)[slot_valid]
        t_span = float(np.max(tv) - np.min(tv))
    else:
        t_span = 0.0

    return n_det, n_bands, t_span, single_band_id


def compute_detection_meta_from_formatted(
    mask_mat: np.ndarray,
    time_vec: np.ndarray,
    slot_is_detection: np.ndarray,
) -> Tuple[int, int, int, float, int]:
    n_obs, n_bands, t_span, single_band_id = compute_meta_features_from_formatted(mask_mat, time_vec)
    det_vec = np.asarray(slot_is_detection, dtype=np.float32).reshape(-1)
    if det_vec.shape[0] != np.asarray(mask_mat).shape[0]:
        raise ValueError("slot_is_detection length must match formatted time dimension.")
    n_det_snr5 = int(np.sum(det_vec > 0))
    return int(n_obs), int(n_det_snr5), int(n_bands), float(t_span), int(single_band_id)


def _bin_indices_1d(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def _flat_bin_index(
    idx_det: np.ndarray,
    idx_band: np.ndarray,
    idx_span: np.ndarray,
    n_det_bins: int,
    n_band_bins: int,
    n_span_bins: int,
) -> np.ndarray:
    _ = n_det_bins  # kept for readability at call site
    return ((idx_det * n_band_bins) + idx_band) * n_span_bins + idx_span


def joint_hist_from_meta_arrays(
    n_det: np.ndarray,
    n_bands: np.ndarray,
    t_span: np.ndarray,
    edges_n_det: np.ndarray,
    edges_n_bands: np.ndarray,
    edges_t_span: np.ndarray,
) -> np.ndarray:
    n_det_bins = int(edges_n_det.size + 1)
    n_band_bins = int(edges_n_bands.size + 1)
    n_span_bins = int(edges_t_span.size + 1)
    n_joint = n_det_bins * n_band_bins * n_span_bins

    det_idx = _bin_indices_1d(np.asarray(n_det, dtype=np.float64), edges_n_det)
    band_idx = _bin_indices_1d(np.asarray(n_bands, dtype=np.float64), edges_n_bands)
    span_idx = _bin_indices_1d(np.asarray(t_span, dtype=np.float64), edges_t_span)
    flat = _flat_bin_index(det_idx, band_idx, span_idx, n_det_bins, n_band_bins, n_span_bins)
    hist = np.bincount(flat, minlength=n_joint)
    return hist.astype(np.int64, copy=False)


def load_positive_density_histogram(
    pos_h5_path: Path,
    edges_n_det: np.ndarray,
    edges_n_bands: np.ndarray,
    edges_t_span: np.ndarray,
    chunk_size: int = 8192,
) -> np.ndarray:
    grp = "events/optical_data"
    with h5py.File(pos_h5_path, "r") as f:
        if grp not in f:
            raise KeyError(f"Missing group '{grp}' in {pos_h5_path}")
        g = f[grp]
        n_total = int(g["values"].shape[0])
        n_det_hist: List[np.ndarray] = []
        n_band_hist: List[np.ndarray] = []
        t_span_hist: List[np.ndarray] = []

        has_meta = (
            "meta_n_det_snr5" in g
            and "meta_n_bands" in g
            and "meta_t_span" in g
        )
        if has_meta:
            for s in range(0, n_total, int(chunk_size)):
                e = min(s + int(chunk_size), n_total)
                n_det_hist.append(np.asarray(g["meta_n_det_snr5"][s:e], dtype=np.float64).reshape(-1))
                n_band_hist.append(np.asarray(g["meta_n_bands"][s:e], dtype=np.float64).reshape(-1))
                t_span_hist.append(np.asarray(g["meta_t_span"][s:e], dtype=np.float64).reshape(-1))
        else:
            ds_masks = g["masks"]
            ds_times = g["times"]
            ds_slot_is_detection = g["slot_is_detection"] if "slot_is_detection" in g else None
            for s in range(0, n_total, int(chunk_size)):
                e = min(s + int(chunk_size), n_total)
                masks = np.asarray(ds_masks[s:e], dtype=np.float32)
                times = np.asarray(ds_times[s:e], dtype=np.float32)
                slot_det = (
                    np.asarray(ds_slot_is_detection[s:e], dtype=np.float32)
                    if ds_slot_is_detection is not None
                    else None
                )
                n_det_buf = []
                n_band_buf = []
                t_span_buf = []
                for i in range(masks.shape[0]):
                    if slot_det is not None:
                        _, n_det_i, n_bands_i, t_span_i, _ = compute_detection_meta_from_formatted(
                            masks[i], times[i], slot_det[i]
                        )
                    else:
                        n_det_i, n_bands_i, t_span_i, _ = compute_meta_features_from_formatted(
                            masks[i], times[i]
                        )
                    n_det_buf.append(float(n_det_i))
                    n_band_buf.append(float(n_bands_i))
                    t_span_buf.append(float(t_span_i))
                n_det_hist.append(np.asarray(n_det_buf, dtype=np.float64))
                n_band_hist.append(np.asarray(n_band_buf, dtype=np.float64))
                t_span_hist.append(np.asarray(t_span_buf, dtype=np.float64))

    if not n_det_hist:
        raise ValueError(f"No positive optical samples found in {pos_h5_path}")
    n_det_arr = np.concatenate(n_det_hist, axis=0)
    n_band_arr = np.concatenate(n_band_hist, axis=0)
    t_span_arr = np.concatenate(t_span_hist, axis=0)
    return joint_hist_from_meta_arrays(
        n_det=n_det_arr,
        n_bands=n_band_arr,
        t_span=t_span_arr,
        edges_n_det=edges_n_det,
        edges_n_bands=edges_n_bands,
        edges_t_span=edges_t_span,
    )


def build_target_quota_from_pos_distribution(
    pos_hist: np.ndarray,
    neg_candidate_hist: np.ndarray,
) -> np.ndarray:
    pos = np.asarray(pos_hist, dtype=np.float64).reshape(-1)
    neg = np.asarray(neg_candidate_hist, dtype=np.int64).reshape(-1)
    if pos.shape != neg.shape:
        raise ValueError("pos_hist and neg_candidate_hist must have identical flattened shape.")

    pos_total = float(pos.sum())
    neg_total = int(neg.sum())
    if pos_total <= 0 or neg_total <= 0:
        return np.zeros_like(neg, dtype=np.int64)

    probs = pos / pos_total
    raw = probs * float(neg_total)
    quota = np.floor(raw).astype(np.int64)
    frac = raw - quota.astype(np.float64)
    residual = int(neg_total - int(quota.sum()))
    if residual > 0:
        order = np.argsort(frac)[::-1]
        for idx in order[:residual]:
            quota[idx] += 1

    quota = np.minimum(quota, neg)
    return quota.astype(np.int64, copy=False)


def read_mjd_explode(readme_path: Path) -> float:
    text = readme_path.read_text(encoding="utf-8", errors="ignore")
    m = MJD_EXPLODE_PATTERN.search(text)
    if m is None:
        raise ValueError(f"MJD_EXPLODE not found in README: {readme_path}")
    return float(m.group(1))


def band_index(band_raw) -> Optional[int]:
    band = str(band_raw).strip()
    if band in BAND_TO_INDEX:
        return BAND_TO_INDEX[band]
    if band.startswith("LSST-"):
        tail = band.split("-", 1)[1]
        return BAND_TO_INDEX.get(tail)
    return None


def first_detection_index(
    flux: np.ndarray,
    fluxerr: np.ndarray,
    photflag: Optional[np.ndarray],
    snr_threshold: float,
) -> Tuple[Optional[int], Optional[str]]:
    det_mask = build_detection_mask(
        flux=flux,
        fluxerr=fluxerr,
        photflag=photflag,
        snr_threshold=snr_threshold,
    )
    if np.any(det_mask):
        if photflag is not None and np.any(photflag.astype(np.int64) != 0):
            return int(np.argmax(det_mask)), "photflag"
        return int(np.argmax(det_mask)), "snr"
    return None, None


def build_detection_mask(
    flux: np.ndarray,
    fluxerr: np.ndarray,
    photflag: Optional[np.ndarray],
    snr_threshold: float,
) -> np.ndarray:
    if flux.size == 0:
        return np.zeros((0,), dtype=bool)

    if photflag is not None:
        det_mask = photflag.astype(np.int64) != 0
        if np.any(det_mask):
            return np.asarray(det_mask, dtype=bool)

    valid = np.isfinite(fluxerr) & (fluxerr > 0)
    if np.any(valid):
        snr = np.full(flux.shape, -np.inf, dtype=np.float64)
        snr[valid] = flux[valid] / fluxerr[valid]
        det_mask = snr > float(snr_threshold)
        return np.asarray(det_mask, dtype=bool)

    return np.zeros(flux.shape, dtype=bool)


def parse_lupt_m5_mag(text: str) -> np.ndarray:
    raw = str(text).strip()
    if raw == "":
        raise ValueError(
            "--lupt_m5_mag is required and must contain 6 comma-separated finite values in order u,g,r,i,z,Y."
        )
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != NUM_BANDS:
        raise ValueError(
            f"--lupt_m5_mag must provide exactly {NUM_BANDS} values in order u,g,r,i,z,Y; got {len(parts)}."
        )
    try:
        vals = np.asarray([float(p) for p in parts], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("--lupt_m5_mag contains non-numeric values.") from exc
    if not np.all(np.isfinite(vals)):
        raise ValueError("--lupt_m5_mag values must be finite.")
    return vals


def build_luptitude_params(
    fluxcal_zp: float,
    psfflux_zp: float,
    lupt_k: float,
    lupt_m5_mag: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    if not np.isfinite(fluxcal_zp) or not np.isfinite(psfflux_zp):
        raise ValueError("fluxcal_zp and psfflux_zp must be finite.")
    if not np.isfinite(lupt_k) or lupt_k <= 0:
        raise ValueError("lupt_k must be finite and > 0.")
    if lupt_m5_mag.shape != (NUM_BANDS,):
        raise ValueError(
            f"lupt_m5_mag must contain exactly {NUM_BANDS} values in order u,g,r,i,z,Y."
        )
    if not np.all(np.isfinite(lupt_m5_mag)):
        raise ValueError("lupt_m5_mag values must be finite.")

    fluxcal_to_psfflux_factor = 10.0 ** (0.4 * (float(psfflux_zp) - float(fluxcal_zp)))
    if not np.isfinite(fluxcal_to_psfflux_factor) or fluxcal_to_psfflux_factor <= 0:
        raise ValueError(
            f"Invalid FLUXCAL->psfFlux conversion factor computed from fluxcal_zp={fluxcal_zp}, psfflux_zp={psfflux_zp}."
        )

    lupt_f5sigma_njy = 10.0 ** ((float(psfflux_zp) - lupt_m5_mag.astype(np.float64, copy=False)) / 2.5)
    if np.any(lupt_f5sigma_njy <= 0) or not np.all(np.isfinite(lupt_f5sigma_njy)):
        raise ValueError("Derived lupt_f5sigma_njy values must be finite and > 0.")
    lupt_b_njy = float(lupt_k) * (lupt_f5sigma_njy / 5.0)
    if np.any(lupt_b_njy <= 0) or not np.all(np.isfinite(lupt_b_njy)):
        raise ValueError("Derived lupt_b_njy values must be finite and > 0.")
    return float(fluxcal_to_psfflux_factor), lupt_f5sigma_njy, lupt_b_njy


def transform_fluxcal_to_luptitude(
    mjd: np.ndarray,
    fluxcal: np.ndarray,
    fluxcalerr: np.ndarray,
    flt: np.ndarray,
    fluxcal_to_psfflux_factor: float,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lupt_b_arr = np.asarray(lupt_b_njy, dtype=np.float64)
    if lupt_b_arr.shape != (NUM_BANDS,):
        raise ValueError(
            f"lupt_b_njy must contain {NUM_BANDS} values in order u,g,r,i,z,Y."
        )

    band_idx_list: List[int] = []
    for band_raw in flt:
        idx = band_index(band_raw)
        band_idx_list.append(-1 if idx is None else int(idx))
    band_idx = np.asarray(band_idx_list, dtype=np.int64)
    base_valid = (
        np.isfinite(mjd)
        & np.isfinite(fluxcal)
        & np.isfinite(fluxcalerr)
        & (fluxcalerr > 0)
        & (band_idx >= 0)
    )
    if not np.any(base_valid):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=flt.dtype),
            np.asarray([], dtype=np.int64),
        )

    idx_valid = np.nonzero(base_valid)[0]
    band_valid = band_idx[idx_valid]
    b_valid = lupt_b_arr[band_valid]

    f_psf = fluxcal[idx_valid] * float(fluxcal_to_psfflux_factor)
    sigma_psf = np.abs(fluxcalerr[idx_valid]) * float(fluxcal_to_psfflux_factor)
    m_lupt = float(psfflux_zp) - ASINH_MAG_FACTOR * (
        np.arcsinh(f_psf / (2.0 * b_valid)) + np.log(b_valid)
    )
    sigma_lupt = ASINH_MAG_FACTOR * sigma_psf / np.sqrt((f_psf * f_psf) + (2.0 * b_valid) ** 2)

    finite_valid = np.isfinite(m_lupt) & np.isfinite(sigma_lupt) & (sigma_lupt > 0)
    if not np.any(finite_valid):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=flt.dtype),
            np.asarray([], dtype=np.int64),
        )

    keep = idx_valid[finite_valid]
    return (
        np.asarray(mjd[keep], dtype=np.float64),
        np.asarray(m_lupt[finite_valid], dtype=np.float64),
        np.asarray(sigma_lupt[finite_valid], dtype=np.float64),
        np.asarray(flt[keep]),
        np.asarray(keep, dtype=np.int64),
    )


def format_realization(
    mjd: np.ndarray,
    flux: np.ndarray,
    fluxerr: np.ndarray,
    flt: np.ndarray,
    is_detection_obs: np.ndarray,
    ra: float,
    dec: float,
    t0_mjd: float,
    enforce_time_window: bool,
    time_window_start: float,
    time_window_end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rel_times = (mjd - float(t0_mjd)) / 100.0
    det_obs = np.asarray(is_detection_obs, dtype=np.float32).reshape(-1)
    if det_obs.shape[0] != rel_times.shape[0]:
        raise ValueError("is_detection_obs length must match mjd/flux length after luptitude transform.")

    if bool(enforce_time_window):
        in_window = (rel_times >= float(time_window_start)) & (rel_times <= float(time_window_end))
        if not np.any(in_window):
            raise ValueError("realization has no points inside configured time window")
        rel_times = rel_times[in_window]
        flux = flux[in_window]
        fluxerr = fluxerr[in_window]
        flt = flt[in_window]
        det_obs = det_obs[in_window]

    if len(rel_times) > MAX_LC_LENGTH:
        keep = np.argsort(np.abs(rel_times))[:MAX_LC_LENGTH]
        keep = np.sort(keep)
        rel_times = rel_times[keep]
        flux = flux[keep]
        fluxerr = fluxerr[keep]
        flt = flt[keep]
        det_obs = det_obs[keep]

    seq_len = min(len(rel_times), MAX_LC_LENGTH)
    val_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    err_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    mask_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    time_vec = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)
    slot_is_detection = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)

    for t in range(seq_len):
        b_idx = band_index(flt[t])
        if b_idx is None:
            continue
        val_mat[t, b_idx] = flux[t]
        err_mat[t, b_idx] = fluxerr[t]
        mask_mat[t, b_idx] = 1.0
        time_vec[t] = rel_times[t]
        slot_is_detection[t] = float(det_obs[t] > 0)

    if not np.any(mask_mat > 0):
        raise ValueError("realization has no valid band after formatting")

    coords = np.array([ra, dec], dtype=np.float32)
    return val_mat, err_mat, mask_mat, time_vec, coords, slot_is_detection, float(t0_mjd)


def iter_event_dirs(base_dir: Path, sim_name: str) -> List[Path]:
    return sorted([p for p in base_dir.glob(f"{sim_name}_*") if p.is_dir()])


def parse_kn_event(
    event_dir: Path,
    sim_name: str,
    snr_threshold: float,
    min_nobs: int,
    fixed_offset_days: float,
    enforce_time_window: bool,
    time_window_start: float,
    time_window_end: float,
    fluxcal_to_psfflux_factor: float,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]], Dict[str, int]]:
    prefix = event_dir.name
    head_path = event_dir / f"{prefix}_HEAD.FITS"
    phot_path = event_dir / f"{prefix}_PHOT.FITS"
    readme_path = event_dir / f"{prefix}.README"

    stats = {
        "n_realizations_total": 0,
        "n_realizations_kept": 0,
        "drop_nobs": 0,
        "drop_no_detection": 0,
        "drop_empty_or_invalid": 0,
    }
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []

    if not (head_path.exists() and phot_path.exists() and readme_path.exists()):
        return out, stats

    try:
        _ = read_mjd_explode(readme_path)  # validated only; t0 comes from first detection
        with fits.open(head_path, memmap=False) as hdul_head, fits.open(
            phot_path, memmap=False
        ) as hdul_phot:
            data_head = hdul_head[1].data
            data_phot = hdul_phot[1].data

            ptrobs_min = data_head["PTROBS_MIN"]
            ptrobs_max = data_head["PTROBS_MAX"]
            mjd_all = data_phot["MJD"]
            flux_all = data_phot["FLUXCAL"]
            fluxerr_all = data_phot["FLUXCALERR"]
            flt_all = data_phot["BAND"]
            photflag_all = data_phot["PHOTFLAG"] if "PHOTFLAG" in data_phot.columns.names else None

            stats["n_realizations_total"] = int(len(data_head))
            for i in range(len(data_head)):
                nobs = int(data_head["NOBS"][i])
                if nobs < int(min_nobs):
                    stats["drop_nobs"] += 1
                    continue

                start_idx = int(ptrobs_min[i]) - 1
                end_idx = int(ptrobs_max[i])
                if start_idx < 0 or end_idx <= start_idx or end_idx > len(mjd_all):
                    stats["drop_empty_or_invalid"] += 1
                    continue

                lc_mjd = np.asarray(mjd_all[start_idx:end_idx], dtype=np.float64)
                lc_flux = np.asarray(flux_all[start_idx:end_idx], dtype=np.float64)
                lc_fluxerr = np.asarray(fluxerr_all[start_idx:end_idx], dtype=np.float64)
                lc_flt = np.asarray(flt_all[start_idx:end_idx])
                lc_photflag = (
                    np.asarray(photflag_all[start_idx:end_idx], dtype=np.int64)
                    if photflag_all is not None
                    else None
                )

                if lc_mjd.size == 0:
                    stats["drop_empty_or_invalid"] += 1
                    continue

                det_idx, _ = first_detection_index(
                    flux=lc_flux,
                    fluxerr=lc_fluxerr,
                    photflag=lc_photflag,
                    snr_threshold=snr_threshold,
                )
                if det_idx is None:
                    stats["drop_no_detection"] += 1
                    continue
                det_mask_obs = build_detection_mask(
                    flux=lc_flux,
                    fluxerr=lc_fluxerr,
                    photflag=lc_photflag,
                    snr_threshold=snr_threshold,
                )

                _ = fixed_offset_days
                t0_key = f"{event_dir}:{i}"
                t0_mjd = deterministic_uniform_sample(
                    low=DEFAULT_ZERO_TIME_WINDOW_START,
                    high=DEFAULT_ZERO_TIME_WINDOW_END,
                    key=t0_key,
                )
                lc_mjd_lupt, lc_lupt, lc_lupt_err, lc_flt_lupt, lc_keep_idx = transform_fluxcal_to_luptitude(
                    mjd=lc_mjd,
                    fluxcal=lc_flux,
                    fluxcalerr=lc_fluxerr,
                    flt=lc_flt,
                    fluxcal_to_psfflux_factor=fluxcal_to_psfflux_factor,
                    psfflux_zp=psfflux_zp,
                    lupt_b_njy=lupt_b_njy,
                )
                if lc_mjd_lupt.size == 0:
                    stats["drop_empty_or_invalid"] += 1
                    continue
                det_mask_lupt = np.asarray(det_mask_obs[lc_keep_idx], dtype=np.float32)

                try:
                    ra = float(data_head["RA"][i])
                    dec = float(data_head["DEC"][i])
                    out.append(
                        format_realization(
                            mjd=lc_mjd_lupt,
                            flux=lc_lupt,
                            fluxerr=lc_lupt_err,
                            flt=lc_flt_lupt,
                            is_detection_obs=det_mask_lupt,
                            ra=ra,
                            dec=dec,
                            t0_mjd=t0_mjd,
                            enforce_time_window=bool(enforce_time_window),
                            time_window_start=float(time_window_start),
                            time_window_end=float(time_window_end),
                        )
                    )
                    stats["n_realizations_kept"] += 1
                except Exception:
                    stats["drop_empty_or_invalid"] += 1
                    continue
    except Exception:
        return out, stats

    return out, stats


def _parse_kn_event_worker(
    event_dir_str: str,
    sim_name: str,
    snr_threshold: float,
    min_nobs: int,
    fixed_offset_days: float,
    enforce_time_window: bool,
    time_window_start: float,
    time_window_end: float,
    fluxcal_to_psfflux_factor: float,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]], Dict[str, int]]:
    event_dir = Path(event_dir_str)
    lcs, stats = parse_kn_event(
        event_dir=event_dir,
        sim_name=sim_name,
        snr_threshold=snr_threshold,
        min_nobs=min_nobs,
        fixed_offset_days=fixed_offset_days,
        enforce_time_window=bool(enforce_time_window),
        time_window_start=float(time_window_start),
        time_window_end=float(time_window_end),
        fluxcal_to_psfflux_factor=fluxcal_to_psfflux_factor,
        psfflux_zp=psfflux_zp,
        lupt_b_njy=lupt_b_njy,
    )
    return event_dir.name, lcs, stats


def infer_transient_type(folder_name: str) -> Optional[str]:
    name = folder_name.lower()
    if "kn" in name:
        return None
    if "agn" in name:
        return "AGN"
    if "tde" in name:
        return "TDE"
    if "ulens" in name:
        return "uLens"
    if "dwarf-nova" in name:
        return "dwarf-nova"
    if "sn" in name or "slsn" in name or "pisn" in name:
        return "SN"
    return None


def iter_negative_head_files(sim_root: Path) -> List[Path]:
    files = list(sim_root.rglob("*_HEAD.FITS")) + list(sim_root.rglob("*_HEAD.FITS.gz"))
    return sorted(set(files))


def find_pair_phot_file(head_path: Path) -> Optional[Path]:
    name = head_path.name
    if name.endswith("_HEAD.FITS"):
        p = head_path.with_name(name.replace("_HEAD.FITS", "_PHOT.FITS"))
        return p if p.exists() else None
    if name.endswith("_HEAD.FITS.gz"):
        p = head_path.with_name(name.replace("_HEAD.FITS.gz", "_PHOT.FITS.gz"))
        return p if p.exists() else None
    return None


def parse_negative_file(
    head_path: Path,
    phot_path: Path,
    transient_type: str,
    snr_threshold: float,
    min_nobs: int,
    fixed_offset_days: float,
    enforce_time_window: bool,
    time_window_start: float,
    time_window_end: float,
    fluxcal_to_psfflux_factor: float,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]], Dict[str, int]]:
    stats = {
        "n_realizations_total": 0,
        "n_realizations_kept": 0,
        "drop_nobs": 0,
        "drop_no_detection": 0,
        "drop_empty_or_invalid": 0,
    }
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = []

    try:
        with fits.open(head_path, memmap=False) as hdul_head, fits.open(
            phot_path, memmap=False
        ) as hdul_phot:
            data_head = hdul_head[1].data
            data_phot = hdul_phot[1].data

            ptrobs_min = data_head["PTROBS_MIN"]
            ptrobs_max = data_head["PTROBS_MAX"]
            mjd_all = data_phot["MJD"]
            flux_all = data_phot["FLUXCAL"]
            fluxerr_all = data_phot["FLUXCALERR"]
            flt_all = data_phot["BAND"]
            photflag_all = data_phot["PHOTFLAG"] if "PHOTFLAG" in data_phot.columns.names else None

            stats["n_realizations_total"] = int(len(data_head))
            for i in range(len(data_head)):
                nobs = int(data_head["NOBS"][i])
                if nobs < int(min_nobs):
                    stats["drop_nobs"] += 1
                    continue

                start_idx = int(ptrobs_min[i]) - 1
                end_idx = int(ptrobs_max[i])
                if start_idx < 0 or end_idx <= start_idx or end_idx > len(mjd_all):
                    stats["drop_empty_or_invalid"] += 1
                    continue

                lc_mjd = np.asarray(mjd_all[start_idx:end_idx], dtype=np.float64)
                lc_flux = np.asarray(flux_all[start_idx:end_idx], dtype=np.float64)
                lc_fluxerr = np.asarray(fluxerr_all[start_idx:end_idx], dtype=np.float64)
                lc_flt = np.asarray(flt_all[start_idx:end_idx])
                lc_photflag = (
                    np.asarray(photflag_all[start_idx:end_idx], dtype=np.int64)
                    if photflag_all is not None
                    else None
                )

                if lc_mjd.size == 0:
                    stats["drop_empty_or_invalid"] += 1
                    continue

                det_idx, _ = first_detection_index(
                    flux=lc_flux,
                    fluxerr=lc_fluxerr,
                    photflag=lc_photflag,
                    snr_threshold=snr_threshold,
                )
                if det_idx is None:
                    stats["drop_no_detection"] += 1
                    continue
                det_mask_obs = build_detection_mask(
                    flux=lc_flux,
                    fluxerr=lc_fluxerr,
                    photflag=lc_photflag,
                    snr_threshold=snr_threshold,
                )

                _ = fixed_offset_days
                t0_key = f"{head_path}:{i}"
                t0_mjd = deterministic_uniform_sample(
                    low=DEFAULT_ZERO_TIME_WINDOW_START,
                    high=DEFAULT_ZERO_TIME_WINDOW_END,
                    key=t0_key,
                )
                lc_mjd_lupt, lc_lupt, lc_lupt_err, lc_flt_lupt, lc_keep_idx = transform_fluxcal_to_luptitude(
                    mjd=lc_mjd,
                    fluxcal=lc_flux,
                    fluxcalerr=lc_fluxerr,
                    flt=lc_flt,
                    fluxcal_to_psfflux_factor=fluxcal_to_psfflux_factor,
                    psfflux_zp=psfflux_zp,
                    lupt_b_njy=lupt_b_njy,
                )
                if lc_mjd_lupt.size == 0:
                    stats["drop_empty_or_invalid"] += 1
                    continue
                det_mask_lupt = np.asarray(det_mask_obs[lc_keep_idx], dtype=np.float32)

                try:
                    ra = float(data_head["RA"][i])
                    dec = float(data_head["DEC"][i])
                    out.append(
                        format_realization(
                            mjd=lc_mjd_lupt,
                            flux=lc_lupt,
                            fluxerr=lc_lupt_err,
                            flt=lc_flt_lupt,
                            is_detection_obs=det_mask_lupt,
                            ra=ra,
                            dec=dec,
                            t0_mjd=t0_mjd,
                            enforce_time_window=bool(enforce_time_window),
                            time_window_start=float(time_window_start),
                            time_window_end=float(time_window_end),
                        )
                    )
                    stats["n_realizations_kept"] += 1
                except Exception:
                    stats["drop_empty_or_invalid"] += 1
                    continue
    except Exception:
        return out, stats

    return out, stats


def _parse_negative_file_worker(
    head_path_str: str,
    phot_path_str: str,
    transient_type: str,
    snr_threshold: float,
    min_nobs: int,
    fixed_offset_days: float,
    enforce_time_window: bool,
    time_window_start: float,
    time_window_end: float,
    fluxcal_to_psfflux_factor: float,
    psfflux_zp: float,
    lupt_b_njy: Sequence[float],
) -> Tuple[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]], Dict[str, int]]:
    lcs, stats = parse_negative_file(
        head_path=Path(head_path_str),
        phot_path=Path(phot_path_str),
        transient_type=transient_type,
        snr_threshold=snr_threshold,
        min_nobs=min_nobs,
        fixed_offset_days=fixed_offset_days,
        enforce_time_window=bool(enforce_time_window),
        time_window_start=float(time_window_start),
        time_window_end=float(time_window_end),
        fluxcal_to_psfflux_factor=fluxcal_to_psfflux_factor,
        psfflux_zp=psfflux_zp,
        lupt_b_njy=lupt_b_njy,
    )
    return transient_type, lcs, stats


def _parallel_map_with_fallback(
    fn,
    iterables: Optional[Tuple[Iterable, ...]] = None,
    worker_count: int = 1,
    chunksize: int = 1,
    total: int = 0,
    desc: str = "",
    arg_rows: Optional[List[Tuple]] = None,
):
    # Accept both call styles:
    # 1) iterables=(col1, col2, ...)
    # 2) arg_rows=[(a1,b1,...), (a2,b2,...), ...]
    if arg_rows is not None:
        if len(arg_rows) == 0:
            return
        cols = list(zip(*arg_rows))
        iterables = tuple([list(col) for col in cols])

    if iterables is None:
        raise TypeError("Either iterables or arg_rows must be provided.")

    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            mapped = executor.map(fn, *iterables, chunksize=chunksize)
            for item in tqdm(mapped, total=total, desc=desc):
                yield item
    except (PermissionError, OSError, RuntimeError) as exc:
        print(
            f"[WARN] ProcessPool unavailable ({exc}); "
            f"fallback to ThreadPoolExecutor with {worker_count} workers."
        )
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            mapped = executor.map(fn, *iterables)
            for item in tqdm(mapped, total=total, desc=desc):
                yield item


def _create_optical_group(
    grp,
    chunk_size: int,
    write_meta_features: bool,
):
    # Keep legacy fields for backward compatibility with existing H5 files.
    # Optical-only training/evaluation consumes values/errors/masks/times only.
    ds_values = grp.create_dataset(
        "values",
        (0, MAX_LC_LENGTH, NUM_BANDS),
        maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
        dtype="f4",
        chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS),
    )
    ds_errors = grp.create_dataset(
        "errors",
        (0, MAX_LC_LENGTH, NUM_BANDS),
        maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
        dtype="f4",
        chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS),
    )
    ds_masks = grp.create_dataset(
        "masks",
        (0, MAX_LC_LENGTH, NUM_BANDS),
        maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
        dtype="f4",
        chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS),
    )
    ds_times = grp.create_dataset(
        "times",
        (0, MAX_LC_LENGTH),
        maxshape=(None, MAX_LC_LENGTH),
        dtype="f4",
        chunks=(chunk_size, MAX_LC_LENGTH),
    )
    ds_slot_is_detection = grp.create_dataset(
        "slot_is_detection",
        (0, MAX_LC_LENGTH),
        maxshape=(None, MAX_LC_LENGTH),
        dtype="f4",
        chunks=(chunk_size, MAX_LC_LENGTH),
    )
    ds_zero_time_mjd_base = grp.create_dataset(
        "zero_time_mjd_base",
        (0,),
        maxshape=(None,),
        dtype="f8",
        chunks=(chunk_size,),
    )
    ds_coords = grp.create_dataset(
        "coordinates",
        (0, 2),
        maxshape=(None, 2),
        dtype="f4",
        chunks=(chunk_size, 2),
    )
    ds_meta_n_det = None
    ds_meta_n_obs = None
    ds_meta_n_det_snr5 = None
    ds_meta_n_bands = None
    ds_meta_t_span = None
    ds_meta_single_band_id = None
    if bool(write_meta_features):
        ds_meta_n_obs = grp.create_dataset(
            "meta_n_obs",
            (0,),
            maxshape=(None,),
            dtype="i2",
            chunks=(chunk_size,),
        )
        ds_meta_n_det = grp.create_dataset(
            "meta_n_det",
            (0,),
            maxshape=(None,),
            dtype="i2",
            chunks=(chunk_size,),
        )
        ds_meta_n_det_snr5 = grp.create_dataset(
            "meta_n_det_snr5",
            (0,),
            maxshape=(None,),
            dtype="i2",
            chunks=(chunk_size,),
        )
        ds_meta_n_bands = grp.create_dataset(
            "meta_n_bands",
            (0,),
            maxshape=(None,),
            dtype="i1",
            chunks=(chunk_size,),
        )
        ds_meta_t_span = grp.create_dataset(
            "meta_t_span",
            (0,),
            maxshape=(None,),
            dtype="f4",
            chunks=(chunk_size,),
        )
        ds_meta_single_band_id = grp.create_dataset(
            "meta_single_band_id",
            (0,),
            maxshape=(None,),
            dtype="i1",
            chunks=(chunk_size,),
        )
    return (
        ds_values,
        ds_errors,
        ds_masks,
        ds_times,
        ds_slot_is_detection,
        ds_zero_time_mjd_base,
        ds_coords,
        ds_meta_n_obs,
        ds_meta_n_det,
        ds_meta_n_det_snr5,
        ds_meta_n_bands,
        ds_meta_t_span,
        ds_meta_single_band_id,
    )


def write_luptitude_metadata_attrs(
    h5_obj,
    fluxcal_zp: float,
    psfflux_zp: float,
    fluxcal_to_psfflux_factor: float,
    lupt_k: float,
    lupt_m5_mag: np.ndarray,
    lupt_f5sigma_njy: np.ndarray,
    lupt_b_njy: np.ndarray,
) -> None:
    h5_obj.attrs["photometry_representation"] = "luptitude"
    h5_obj.attrs["flux_input_column"] = "FLUXCAL"
    h5_obj.attrs["fluxerr_input_column"] = "FLUXCALERR"
    h5_obj.attrs["fluxcal_zp"] = float(fluxcal_zp)
    h5_obj.attrs["psfflux_zp"] = float(psfflux_zp)
    h5_obj.attrs["fluxcal_to_psfflux_factor"] = float(fluxcal_to_psfflux_factor)
    h5_obj.attrs["lupt_k"] = float(lupt_k)
    h5_obj.attrs["lupt_band_order"] = ",".join(LUPT_BAND_ORDER)
    h5_obj.attrs["lupt_m5_mag"] = np.asarray(lupt_m5_mag, dtype=np.float64)
    h5_obj.attrs["lupt_f5sigma_njy"] = np.asarray(lupt_f5sigma_njy, dtype=np.float64)
    h5_obj.attrs["lupt_b_njy"] = np.asarray(lupt_b_njy, dtype=np.float64)
    h5_obj.attrs["values_semantics"] = "luptitude"
    h5_obj.attrs["errors_semantics"] = "luptitude_sigma"


def create_positive_h5(
    output_h5: Path,
    bns_sim_root: Path,
    bns_sim_name: str,
    nsbh_sim_root: Path,
    nsbh_sim_name: str,
    snr_threshold: float,
    min_nobs: int,
    fixed_offset_days: float,
    fluxcal_zp: float,
    psfflux_zp: float,
    lupt_k: float,
    lupt_m5_mag: np.ndarray,
    lupt_f5sigma_njy: np.ndarray,
    fluxcal_to_psfflux_factor: float,
    lupt_b_njy: np.ndarray,
    max_events_per_source: Optional[int],
    max_lcs_per_event: Optional[int],
    buffer_limit: int,
    num_workers: int,
    enforce_time_window: bool,
    time_window_start: float,
    time_window_end: float,
    write_meta_features: bool,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    dt_str = h5py.string_dtype(encoding="utf-8")
    chunk_size = 1024

    with h5py.File(output_h5, "w") as f:
        gw_grp = f.create_group("events/gw_data")
        ds_gw_ids = gw_grp.create_dataset("ids", (0,), maxshape=(None,), dtype=dt_str, chunks=(chunk_size,))
        ds_gw_source = gw_grp.create_dataset(
            "source_type", (0,), maxshape=(None,), dtype=dt_str, chunks=(chunk_size,)
        )

        opt_grp = f.create_group("events/optical_data")
        (
            ds_values,
            ds_errors,
            ds_masks,
            ds_times,
            ds_slot_is_detection,
            ds_zero_time_mjd_base,
            ds_coords,
            ds_meta_n_obs,
            ds_meta_n_det,
            ds_meta_n_det_snr5,
            ds_meta_n_bands,
            ds_meta_t_span,
            ds_meta_single_band_id,
        ) = _create_optical_group(opt_grp, chunk_size, write_meta_features=bool(write_meta_features))
        ds_parent = opt_grp.create_dataset(
            "parent_gw_idx",
            (0,),
            maxshape=(None,),
            dtype="i4",
            chunks=(chunk_size,),
        )

        gw_count = 0
        optical_count = 0
        b_vals: List[np.ndarray] = []
        b_errs: List[np.ndarray] = []
        b_masks: List[np.ndarray] = []
        b_times: List[np.ndarray] = []
        b_slot_is_detection: List[np.ndarray] = []
        b_zero_time_mjd_base: List[float] = []
        b_coords: List[np.ndarray] = []
        b_parent: List[int] = []
        b_meta_n_obs: List[int] = []
        b_meta_n_det: List[int] = []
        b_meta_n_det_snr5: List[int] = []
        b_meta_n_bands: List[int] = []
        b_meta_t_span: List[float] = []
        b_meta_single_band_id: List[int] = []

        stats = {
            "bns_events_total": 0,
            "nsbh_events_total": 0,
            "bns_events_written": 0,
            "nsbh_events_written": 0,
            "drop_lcs_over_event_cap": 0,
            "drop_nobs": 0,
            "drop_no_detection": 0,
            "drop_empty_or_invalid": 0,
        }

        def flush() -> None:
            nonlocal optical_count
            if not b_vals:
                return
            n_new = len(b_vals)
            cur = optical_count
            new_size = cur + n_new
            ds_values.resize(new_size, axis=0)
            ds_errors.resize(new_size, axis=0)
            ds_masks.resize(new_size, axis=0)
            ds_times.resize(new_size, axis=0)
            ds_slot_is_detection.resize(new_size, axis=0)
            ds_zero_time_mjd_base.resize(new_size, axis=0)
            ds_coords.resize(new_size, axis=0)
            ds_parent.resize(new_size, axis=0)
            if ds_meta_n_det is not None:
                ds_meta_n_obs.resize(new_size, axis=0)
                ds_meta_n_det.resize(new_size, axis=0)
                ds_meta_n_det_snr5.resize(new_size, axis=0)
                ds_meta_n_bands.resize(new_size, axis=0)
                ds_meta_t_span.resize(new_size, axis=0)
                ds_meta_single_band_id.resize(new_size, axis=0)
            ds_values[cur:new_size] = np.asarray(b_vals, dtype=np.float32)
            ds_errors[cur:new_size] = np.asarray(b_errs, dtype=np.float32)
            ds_masks[cur:new_size] = np.asarray(b_masks, dtype=np.float32)
            ds_times[cur:new_size] = np.asarray(b_times, dtype=np.float32)
            ds_slot_is_detection[cur:new_size] = np.asarray(b_slot_is_detection, dtype=np.float32)
            ds_zero_time_mjd_base[cur:new_size] = np.asarray(b_zero_time_mjd_base, dtype=np.float64)
            ds_coords[cur:new_size] = np.asarray(b_coords, dtype=np.float32)
            ds_parent[cur:new_size] = np.asarray(b_parent, dtype=np.int32)
            if ds_meta_n_det is not None:
                ds_meta_n_obs[cur:new_size] = np.asarray(b_meta_n_obs, dtype=np.int16)
                ds_meta_n_det[cur:new_size] = np.asarray(b_meta_n_det, dtype=np.int16)
                ds_meta_n_det_snr5[cur:new_size] = np.asarray(b_meta_n_det_snr5, dtype=np.int16)
                ds_meta_n_bands[cur:new_size] = np.asarray(b_meta_n_bands, dtype=np.int8)
                ds_meta_t_span[cur:new_size] = np.asarray(b_meta_t_span, dtype=np.float32)
                ds_meta_single_band_id[cur:new_size] = np.asarray(
                    b_meta_single_band_id, dtype=np.int8
                )

            optical_count = new_size
            b_vals.clear()
            b_errs.clear()
            b_masks.clear()
            b_times.clear()
            b_slot_is_detection.clear()
            b_zero_time_mjd_base.clear()
            b_coords.clear()
            b_parent.clear()
            b_meta_n_obs.clear()
            b_meta_n_det.clear()
            b_meta_n_det_snr5.clear()
            b_meta_n_bands.clear()
            b_meta_t_span.clear()
            b_meta_single_band_id.clear()

        worker_count = max(1, int(num_workers))

        def consume_positive_result(
            event_name: str,
            source_tag: str,
            lcs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
            event_stats: Dict[str, int],
            key_written: str,
        ) -> None:
            nonlocal gw_count
            stats["drop_nobs"] += int(event_stats["drop_nobs"])
            stats["drop_no_detection"] += int(event_stats["drop_no_detection"])
            stats["drop_empty_or_invalid"] += int(event_stats["drop_empty_or_invalid"])

            if len(lcs) == 0:
                return

            if max_lcs_per_event is not None and len(lcs) > int(max_lcs_per_event):
                n_drop = len(lcs) - int(max_lcs_per_event)
                stats["drop_lcs_over_event_cap"] += int(n_drop)
                lcs = lcs[: int(max_lcs_per_event)]

            gw_idx = gw_count
            ds_gw_ids.resize(gw_count + 1, axis=0)
            ds_gw_source.resize(gw_count + 1, axis=0)
            ds_gw_ids[gw_idx] = event_name
            ds_gw_source[gw_idx] = source_tag
            gw_count += 1
            stats[key_written] += 1

            for vals, errs, masks, times, coords, slot_is_detection, zero_time_mjd_base in lcs:
                b_vals.append(vals)
                b_errs.append(errs)
                b_masks.append(masks)
                b_times.append(times)
                b_slot_is_detection.append(slot_is_detection)
                b_zero_time_mjd_base.append(float(zero_time_mjd_base))
                b_coords.append(coords)
                b_parent.append(gw_idx)
                if ds_meta_n_det is not None:
                    (
                        n_obs_i,
                        n_det_snr5_i,
                        n_bands_i,
                        t_span_i,
                        single_band_i,
                    ) = compute_detection_meta_from_formatted(
                        masks,
                        times,
                        slot_is_detection,
                    )
                    b_meta_n_obs.append(int(n_obs_i))
                    b_meta_n_det.append(int(n_obs_i))
                    b_meta_n_det_snr5.append(int(n_det_snr5_i))
                    b_meta_n_bands.append(int(n_bands_i))
                    b_meta_t_span.append(float(t_span_i))
                    b_meta_single_band_id.append(int(single_band_i))

            if len(b_vals) >= int(buffer_limit):
                flush()

        for source_tag, sim_root, sim_name in [
            ("bns", bns_sim_root, bns_sim_name),
            ("nsbh", nsbh_sim_root, nsbh_sim_name),
        ]:
            event_dirs = iter_event_dirs(sim_root, sim_name)
            if max_events_per_source is not None:
                event_dirs = event_dirs[: int(max_events_per_source)]

            key_total = f"{source_tag}_events_total"
            key_written = f"{source_tag}_events_written"
            stats[key_total] = int(len(event_dirs))
            if worker_count == 1 or len(event_dirs) <= 1:
                for event_dir in tqdm(event_dirs, desc=f"Positive {source_tag.upper()}"):
                    lcs, event_stats = parse_kn_event(
                        event_dir=event_dir,
                        sim_name=sim_name,
                        snr_threshold=snr_threshold,
                        min_nobs=min_nobs,
                        fixed_offset_days=fixed_offset_days,
                        enforce_time_window=bool(enforce_time_window),
                        time_window_start=float(time_window_start),
                        time_window_end=float(time_window_end),
                        fluxcal_to_psfflux_factor=fluxcal_to_psfflux_factor,
                        psfflux_zp=psfflux_zp,
                        lupt_b_njy=lupt_b_njy,
                    )
                    consume_positive_result(
                        event_name=event_dir.name,
                        source_tag=source_tag,
                        lcs=lcs,
                        event_stats=event_stats,
                        key_written=key_written,
                    )
            else:
                event_dir_strs = [str(p) for p in event_dirs]
                n_rows = len(event_dir_strs)
                chunksize = max(1, n_rows // (worker_count * 8))
                arg_rows = [
                    (
                        event_dir_strs[i],
                        sim_name,
                        float(snr_threshold),
                        int(min_nobs),
                        float(fixed_offset_days),
                        bool(enforce_time_window),
                        float(time_window_start),
                        float(time_window_end),
                        float(fluxcal_to_psfflux_factor),
                        float(psfflux_zp),
                        tuple(float(x) for x in np.asarray(lupt_b_njy, dtype=np.float64).tolist()),
                    )
                    for i in range(n_rows)
                ]
                for event_name, lcs, event_stats in _parallel_map_with_fallback(
                    _parse_kn_event_worker,
                    arg_rows=arg_rows,
                    worker_count=worker_count,
                    chunksize=chunksize,
                    total=n_rows,
                    desc=f"Positive {source_tag.upper()}",
                ):
                    consume_positive_result(
                        event_name=event_name,
                        source_tag=source_tag,
                        lcs=lcs,
                        event_stats=event_stats,
                        key_written=key_written,
                    )

        flush()

        f.attrs["n_total_gw"] = int(gw_count)
        f.attrs["n_total_optical"] = int(optical_count)
        for k, v in stats.items():
            f.attrs[k] = int(v)

        f.attrs["time_zero_anchor"] = "uniform_window"
        f.attrs["first_detection_rule"] = "photflag_nonzero_else_snr_gt_5"
        f.attrs["time_scale_divisor_days"] = 100.0
        f.attrs["time_zero_version"] = "fd_v1"
        f.attrs["drop_no_detection_count"] = int(stats["drop_no_detection"])
        f.attrs["snr_threshold"] = float(snr_threshold)
        f.attrs["detection_photflags"] = "nonzero"
        f.attrs["fixed_offset_days"] = float(fixed_offset_days)
        f.attrs["num_workers"] = int(worker_count)
        f.attrs["time_zero_base_semantics"] = "uniform_sampled_mjd_in_fixed_window"
        f.attrs["zero_time_window_start"] = float(DEFAULT_ZERO_TIME_WINDOW_START)
        f.attrs["zero_time_window_end"] = float(DEFAULT_ZERO_TIME_WINDOW_END)
        f.attrs["time_unit"] = "mjd_days"
        f.attrs["runtime_offset_applied"] = 1
        f.attrs["enforce_time_window"] = int(bool(enforce_time_window))
        f.attrs["time_window_start"] = float(time_window_start)
        f.attrs["time_window_end"] = float(time_window_end)
        f.attrs["write_meta_features"] = int(bool(write_meta_features))
        f.attrs["slot_is_detection_semantics"] = "formatted_slot_is_detection_snr_gt_threshold"
        f.attrs["meta_n_obs_semantics"] = "formatted_observation_count"
        f.attrs["meta_n_det_snr5_semantics"] = "formatted_detection_count_snr_gt_threshold"
        if max_lcs_per_event is not None:
            f.attrs["max_lcs_per_event"] = int(max_lcs_per_event)
        write_luptitude_metadata_attrs(
            h5_obj=f,
            fluxcal_zp=fluxcal_zp,
            psfflux_zp=psfflux_zp,
            fluxcal_to_psfflux_factor=fluxcal_to_psfflux_factor,
            lupt_k=lupt_k,
            lupt_m5_mag=lupt_m5_mag,
            lupt_f5sigma_njy=lupt_f5sigma_njy,
            lupt_b_njy=lupt_b_njy,
        )

    print(f"[POS] Saved: {output_h5}")


def create_negative_h5(
    output_h5: Path,
    neg_sim_root: Path,
    neg_group: str,
    snr_threshold: float,
    min_nobs: int,
    fixed_offset_days: float,
    fluxcal_zp: float,
    psfflux_zp: float,
    lupt_k: float,
    lupt_m5_mag: np.ndarray,
    lupt_f5sigma_njy: np.ndarray,
    fluxcal_to_psfflux_factor: float,
    lupt_b_njy: np.ndarray,
    max_negative_heads: Optional[int],
    buffer_limit: int,
    num_workers: int,
    enforce_time_window: bool,
    time_window_start: float,
    time_window_end: float,
    write_meta_features: bool,
    neg_match_pos_density: bool,
    density_match_pos_h5: Optional[Path],
    density_bins_n_det: np.ndarray,
    density_bins_n_bands: np.ndarray,
    density_bins_t_span: np.ndarray,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 1024
    dt_str = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_h5, "w") as f:
        grp = f.create_group(neg_group)
        (
            ds_values,
            ds_errors,
            ds_masks,
            ds_times,
            ds_slot_is_detection,
            ds_zero_time_mjd_base,
            ds_coords,
            ds_meta_n_obs,
            ds_meta_n_det,
            ds_meta_n_det_snr5,
            ds_meta_n_bands,
            ds_meta_t_span,
            ds_meta_single_band_id,
        ) = _create_optical_group(grp, chunk_size, write_meta_features=bool(write_meta_features))
        ds_types = grp.create_dataset("types", (0,), maxshape=(None,), dtype=dt_str, chunks=(chunk_size,))
        ds_zero_time_mjd_cls_base = grp.create_dataset(
            "zero_time_mjd_cls_base",
            (0,),
            maxshape=(None,),
            dtype="f8",
            chunks=(chunk_size,),
        )

        total_optical = 0
        b_vals: List[np.ndarray] = []
        b_errs: List[np.ndarray] = []
        b_masks: List[np.ndarray] = []
        b_times: List[np.ndarray] = []
        b_slot_is_detection: List[np.ndarray] = []
        b_zero_time_mjd_base: List[float] = []
        b_zero_time_mjd_cls_base: List[float] = []
        b_coords: List[np.ndarray] = []
        b_types: List[str] = []
        b_meta_n_obs: List[int] = []
        b_meta_n_det: List[int] = []
        b_meta_n_det_snr5: List[int] = []
        b_meta_n_bands: List[int] = []
        b_meta_t_span: List[float] = []
        b_meta_single_band_id: List[int] = []

        stats = {
            "head_files_total": 0,
            "head_files_used": 0,
            "drop_nobs": 0,
            "drop_no_detection": 0,
            "drop_empty_or_invalid": 0,
            "drop_density_mismatch": 0,
            "density_candidates_total": 0,
            "density_kept_total": 0,
        }

        def flush() -> None:
            nonlocal total_optical
            if not b_vals:
                return
            n_new = len(b_vals)
            cur = total_optical
            new_size = cur + n_new
            ds_values.resize(new_size, axis=0)
            ds_errors.resize(new_size, axis=0)
            ds_masks.resize(new_size, axis=0)
            ds_times.resize(new_size, axis=0)
            ds_slot_is_detection.resize(new_size, axis=0)
            ds_zero_time_mjd_base.resize(new_size, axis=0)
            ds_coords.resize(new_size, axis=0)
            ds_types.resize(new_size, axis=0)
            ds_zero_time_mjd_cls_base.resize(new_size, axis=0)
            if ds_meta_n_det is not None:
                ds_meta_n_obs.resize(new_size, axis=0)
                ds_meta_n_det.resize(new_size, axis=0)
                ds_meta_n_det_snr5.resize(new_size, axis=0)
                ds_meta_n_bands.resize(new_size, axis=0)
                ds_meta_t_span.resize(new_size, axis=0)
                ds_meta_single_band_id.resize(new_size, axis=0)
            ds_values[cur:new_size] = np.asarray(b_vals, dtype=np.float32)
            ds_errors[cur:new_size] = np.asarray(b_errs, dtype=np.float32)
            ds_masks[cur:new_size] = np.asarray(b_masks, dtype=np.float32)
            ds_times[cur:new_size] = np.asarray(b_times, dtype=np.float32)
            ds_slot_is_detection[cur:new_size] = np.asarray(b_slot_is_detection, dtype=np.float32)
            ds_zero_time_mjd_base[cur:new_size] = np.asarray(b_zero_time_mjd_base, dtype=np.float64)
            ds_coords[cur:new_size] = np.asarray(b_coords, dtype=np.float32)
            ds_types[cur:new_size] = np.asarray(b_types, dtype=object)
            ds_zero_time_mjd_cls_base[cur:new_size] = np.asarray(
                b_zero_time_mjd_cls_base, dtype=np.float64
            )
            if ds_meta_n_det is not None:
                ds_meta_n_obs[cur:new_size] = np.asarray(b_meta_n_obs, dtype=np.int16)
                ds_meta_n_det[cur:new_size] = np.asarray(b_meta_n_det, dtype=np.int16)
                ds_meta_n_det_snr5[cur:new_size] = np.asarray(b_meta_n_det_snr5, dtype=np.int16)
                ds_meta_n_bands[cur:new_size] = np.asarray(b_meta_n_bands, dtype=np.int8)
                ds_meta_t_span[cur:new_size] = np.asarray(b_meta_t_span, dtype=np.float32)
                ds_meta_single_band_id[cur:new_size] = np.asarray(
                    b_meta_single_band_id, dtype=np.int8
                )

            total_optical = new_size
            b_vals.clear()
            b_errs.clear()
            b_masks.clear()
            b_times.clear()
            b_slot_is_detection.clear()
            b_zero_time_mjd_base.clear()
            b_zero_time_mjd_cls_base.clear()
            b_coords.clear()
            b_types.clear()
            b_meta_n_obs.clear()
            b_meta_n_det.clear()
            b_meta_n_det_snr5.clear()
            b_meta_n_bands.clear()
            b_meta_t_span.clear()
            b_meta_single_band_id.clear()

        worker_count = max(1, int(num_workers))
        density_enabled = bool(neg_match_pos_density)
        density_quota: Optional[np.ndarray] = None
        density_kept: Optional[np.ndarray] = None
        density_edges_n_det = np.asarray(density_bins_n_det, dtype=np.float64)
        density_edges_n_bands = np.asarray(density_bins_n_bands, dtype=np.float64)
        density_edges_t_span = np.asarray(density_bins_t_span, dtype=np.float64)
        n_det_bins = int(density_edges_n_det.size + 1)
        n_band_bins = int(density_edges_n_bands.size + 1)
        n_span_bins = int(density_edges_t_span.size + 1)
        n_joint_bins = int(n_det_bins * n_band_bins * n_span_bins)

        def sample_flat_bin(
            masks: np.ndarray,
            times: np.ndarray,
            slot_is_detection: np.ndarray,
        ) -> Tuple[int, int, int, int, float, int]:
            n_obs_i, n_det_i, n_bands_i, t_span_i, single_band_i = compute_detection_meta_from_formatted(
                masks, times, slot_is_detection
            )
            det_idx = int(np.searchsorted(density_edges_n_det, float(n_det_i), side="right"))
            band_idx = int(np.searchsorted(density_edges_n_bands, float(n_bands_i), side="right"))
            span_idx = int(np.searchsorted(density_edges_t_span, float(t_span_i), side="right"))
            flat_idx = int(((det_idx * n_band_bins) + band_idx) * n_span_bins + span_idx)
            return flat_idx, n_obs_i, n_det_i, n_bands_i, t_span_i, single_band_i

        def consume_negative_result(
            transient_type: str,
            lcs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]],
            file_stats: Dict[str, int],
        ) -> None:
            stats["head_files_used"] += 1
            stats["drop_nobs"] += int(file_stats["drop_nobs"])
            stats["drop_no_detection"] += int(file_stats["drop_no_detection"])
            stats["drop_empty_or_invalid"] += int(file_stats["drop_empty_or_invalid"])

            for vals, errs, masks, times, coords, slot_is_detection, zero_time_mjd_base in lcs:
                flat_idx, n_obs_i, n_det_i, n_bands_i, t_span_i, single_band_i = sample_flat_bin(
                    masks, times, slot_is_detection
                )
                if density_enabled:
                    if density_quota is None or density_kept is None:
                        raise RuntimeError("Density matching enabled but quota buffers are not initialized.")
                    if density_kept[flat_idx] >= density_quota[flat_idx]:
                        stats["drop_density_mismatch"] += 1
                        continue
                    density_kept[flat_idx] += 1
                    stats["density_kept_total"] += 1

                b_vals.append(vals)
                b_errs.append(errs)
                b_masks.append(masks)
                b_times.append(times)
                b_slot_is_detection.append(slot_is_detection)
                b_zero_time_mjd_base.append(float(zero_time_mjd_base))
                b_zero_time_mjd_cls_base.append(float(zero_time_mjd_base))
                b_coords.append(coords)
                b_types.append(transient_type)
                if ds_meta_n_det is not None:
                    b_meta_n_obs.append(int(n_obs_i))
                    b_meta_n_det.append(int(n_det_i))
                    b_meta_n_det_snr5.append(int(n_det_i))
                    b_meta_n_bands.append(int(n_bands_i))
                    b_meta_t_span.append(float(t_span_i))
                    b_meta_single_band_id.append(int(single_band_i))

            if len(b_vals) >= int(buffer_limit):
                flush()

        head_files = iter_negative_head_files(neg_sim_root)
        if max_negative_heads is not None:
            head_files = head_files[: int(max_negative_heads)]
        stats["head_files_total"] = int(len(head_files))

        tasks: List[Tuple[str, str, str]] = []
        for head_path in head_files:
            transient_type = infer_transient_type(head_path.parent.name)
            if transient_type is None:
                continue
            phot_path = find_pair_phot_file(head_path)
            if phot_path is None:
                continue
            tasks.append((str(head_path), str(phot_path), transient_type))

        if density_enabled:
            if density_match_pos_h5 is None:
                raise ValueError(
                    "neg_match_pos_density=true requires density_match_pos_h5 to be provided."
                )
            pos_hist = load_positive_density_histogram(
                pos_h5_path=Path(density_match_pos_h5),
                edges_n_det=density_edges_n_det,
                edges_n_bands=density_edges_n_bands,
                edges_t_span=density_edges_t_span,
            )
            if int(pos_hist.sum()) <= 0:
                raise ValueError(f"Positive density histogram is empty: {density_match_pos_h5}")

            candidate_hist = np.zeros((n_joint_bins,), dtype=np.int64)
            if worker_count == 1 or len(tasks) <= 1:
                for head_path_str, phot_path_str, transient_type in tqdm(
                    tasks, desc="Negative density pass-1"
                ):
                    lcs, _ = parse_negative_file(
                        head_path=Path(head_path_str),
                        phot_path=Path(phot_path_str),
                        transient_type=transient_type,
                        snr_threshold=snr_threshold,
                        min_nobs=min_nobs,
                        fixed_offset_days=fixed_offset_days,
                        enforce_time_window=bool(enforce_time_window),
                        time_window_start=float(time_window_start),
                        time_window_end=float(time_window_end),
                        fluxcal_to_psfflux_factor=fluxcal_to_psfflux_factor,
                        psfflux_zp=psfflux_zp,
                        lupt_b_njy=lupt_b_njy,
                    )
                    for _, _, masks, times, _, slot_is_detection, _ in lcs:
                        flat_idx, _, _, _, _, _ = sample_flat_bin(masks, times, slot_is_detection)
                        candidate_hist[flat_idx] += 1
            else:
                head_path_strs = [t[0] for t in tasks]
                phot_path_strs = [t[1] for t in tasks]
                transient_types = [t[2] for t in tasks]
                n_rows = len(tasks)
                chunksize = max(1, n_rows // (worker_count * 8))
                arg_rows_pass1 = [
                    (
                        head_path_strs[i],
                        phot_path_strs[i],
                            transient_types[i],
                            float(snr_threshold),
                            int(min_nobs),
                            float(fixed_offset_days),
                            bool(enforce_time_window),
                            float(time_window_start),
                            float(time_window_end),
                            float(fluxcal_to_psfflux_factor),
                            float(psfflux_zp),
                        tuple(float(x) for x in np.asarray(lupt_b_njy, dtype=np.float64).tolist()),
                    )
                    for i in range(n_rows)
                ]
                for _, lcs, _ in _parallel_map_with_fallback(
                    _parse_negative_file_worker,
                    arg_rows=arg_rows_pass1,
                    worker_count=worker_count,
                    chunksize=chunksize,
                    total=n_rows,
                    desc="Negative density pass-1",
                ):
                    for _, _, masks, times, _, slot_is_detection, _ in lcs:
                        flat_idx, _, _, _, _, _ = sample_flat_bin(masks, times, slot_is_detection)
                        candidate_hist[flat_idx] += 1

            density_quota = build_target_quota_from_pos_distribution(pos_hist=pos_hist, neg_candidate_hist=candidate_hist)
            density_kept = np.zeros_like(density_quota, dtype=np.int64)
            stats["density_candidates_total"] = int(candidate_hist.sum())
            print(
                "[NEG] Density matching enabled: "
                f"candidates={int(candidate_hist.sum())}, "
                f"quota_sum={int(density_quota.sum())}, "
                f"pos_ref={density_match_pos_h5}"
            )
        else:
            density_kept = np.zeros((n_joint_bins,), dtype=np.int64)
            density_quota = np.zeros((n_joint_bins,), dtype=np.int64)

        if worker_count == 1 or len(tasks) <= 1:
            for head_path_str, phot_path_str, transient_type in tqdm(tasks, desc="Negative HEAD files"):
                lcs, file_stats = parse_negative_file(
                    head_path=Path(head_path_str),
                    phot_path=Path(phot_path_str),
                    transient_type=transient_type,
                    snr_threshold=snr_threshold,
                    min_nobs=min_nobs,
                    fixed_offset_days=fixed_offset_days,
                    enforce_time_window=bool(enforce_time_window),
                    time_window_start=float(time_window_start),
                    time_window_end=float(time_window_end),
                    fluxcal_to_psfflux_factor=fluxcal_to_psfflux_factor,
                    psfflux_zp=psfflux_zp,
                    lupt_b_njy=lupt_b_njy,
                )
                consume_negative_result(
                    transient_type=transient_type,
                    lcs=lcs,
                    file_stats=file_stats,
                )
        else:
            head_path_strs = [t[0] for t in tasks]
            phot_path_strs = [t[1] for t in tasks]
            transient_types = [t[2] for t in tasks]
            n_rows = len(tasks)
            chunksize = max(1, n_rows // (worker_count * 8))
            arg_rows = [
                (
                    head_path_strs[i],
                    phot_path_strs[i],
                        transient_types[i],
                        float(snr_threshold),
                        int(min_nobs),
                        float(fixed_offset_days),
                        bool(enforce_time_window),
                        float(time_window_start),
                        float(time_window_end),
                        float(fluxcal_to_psfflux_factor),
                        float(psfflux_zp),
                    tuple(float(x) for x in np.asarray(lupt_b_njy, dtype=np.float64).tolist()),
                )
                for i in range(n_rows)
            ]
            for transient_type, lcs, file_stats in _parallel_map_with_fallback(
                _parse_negative_file_worker,
                arg_rows=arg_rows,
                worker_count=worker_count,
                chunksize=chunksize,
                total=n_rows,
                desc="Negative HEAD files",
            ):
                consume_negative_result(
                    transient_type=transient_type,
                    lcs=lcs,
                    file_stats=file_stats,
                )

        flush()
        if not density_enabled:
            stats["density_candidates_total"] = int(total_optical)
            stats["density_kept_total"] = int(total_optical)

        f.attrs["n_total_optical"] = int(total_optical)
        for k, v in stats.items():
            f.attrs[k] = int(v)
        f.attrs["time_zero_anchor"] = "uniform_window"
        f.attrs["first_detection_rule"] = "photflag_nonzero_else_snr_gt_5"
        f.attrs["time_scale_divisor_days"] = 100.0
        f.attrs["time_zero_version"] = "fd_v1"
        f.attrs["drop_no_detection_count"] = int(stats["drop_no_detection"])
        f.attrs["snr_threshold"] = float(snr_threshold)
        f.attrs["detection_photflags"] = "nonzero"
        f.attrs["fixed_offset_days"] = float(fixed_offset_days)
        f.attrs["num_workers"] = int(worker_count)
        f.attrs["time_zero_base_semantics"] = "uniform_sampled_mjd_in_fixed_window"
        f.attrs["zero_time_window_start"] = float(DEFAULT_ZERO_TIME_WINDOW_START)
        f.attrs["zero_time_window_end"] = float(DEFAULT_ZERO_TIME_WINDOW_END)
        f.attrs["time_unit"] = "mjd_days"
        f.attrs["runtime_offset_applied"] = 1
        f.attrs["enforce_time_window"] = int(bool(enforce_time_window))
        f.attrs["time_window_start"] = float(time_window_start)
        f.attrs["time_window_end"] = float(time_window_end)
        f.attrs["write_meta_features"] = int(bool(write_meta_features))
        f.attrs["slot_is_detection_semantics"] = "formatted_slot_is_detection_snr_gt_threshold"
        f.attrs["meta_n_obs_semantics"] = "formatted_observation_count"
        f.attrs["meta_n_det_snr5_semantics"] = "formatted_detection_count_snr_gt_threshold"
        f.attrs["neg_match_pos_density"] = int(bool(density_enabled))
        f.attrs["density_bins_n_det"] = np.asarray(density_bins_n_det, dtype=np.float64)
        f.attrs["density_bins_n_bands"] = np.asarray(density_bins_n_bands, dtype=np.float64)
        f.attrs["density_bins_t_span"] = np.asarray(density_bins_t_span, dtype=np.float64)
        if density_match_pos_h5 is not None:
            f.attrs["density_match_pos_h5"] = str(density_match_pos_h5)
        f.attrs["cls_anchor_policy"] = "equal_to_zero_time_mjd_base"
        write_luptitude_metadata_attrs(
            h5_obj=f,
            fluxcal_zp=fluxcal_zp,
            psfflux_zp=psfflux_zp,
            fluxcal_to_psfflux_factor=fluxcal_to_psfflux_factor,
            lupt_k=lupt_k,
            lupt_m5_mag=lupt_m5_mag,
            lupt_f5sigma_njy=lupt_f5sigma_njy,
            lupt_b_njy=lupt_b_njy,
        )

    print(f"[NEG] Saved: {output_h5}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create NEW optical-only H5 datasets with first-detection time zero."
    )
    p.add_argument("--build_positive", action="store_true")
    p.add_argument("--build_negative", action="store_true")

    p.add_argument("--output_pos_h5", type=str, default=None)
    p.add_argument("--output_neg_h5", type=str, default=None)
    p.add_argument("--neg_group", type=str, default="ELASTICC2/optical_data")

    p.add_argument(
        "--bns_sim_root",
        type=str,
        default="/fred/oz016/bgao_kn/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS_AUG",
    )
    p.add_argument("--bns_sim_name", type=str, default="LSST_KN_BNS_AUG")
    p.add_argument(
        "--nsbh_sim_root",
        type=str,
        default="/fred/oz016/bgao_kn/SNANA/SNDATA_ROOT/SIM/LSST_KN_NSBH_TRAIN",
    )
    p.add_argument("--nsbh_sim_name", type=str, default="LSST_KN_NSBH_TRAIN")

    p.add_argument(
        "--neg_sim_root",
        type=str,
        default=None,
        help="Root directory for non-KN simulations (HEAD/PHOT FITS). Required if --build_negative.",
    )

    p.add_argument("--max_events_per_source", type=int, default=None)
    p.add_argument("--max_lcs_per_event", type=int, default=None)
    p.add_argument("--max_negative_heads", type=int, default=None)
    p.add_argument("--buffer_limit", type=int, default=5000)
    p.add_argument("--min_nobs", type=int, default=5)
    p.add_argument("--snr_threshold", type=float, default=5.0)
    p.add_argument("--fixed_offset_days", type=float, default=0.0)
    p.add_argument(
        "--enforce_time_window",
        type=str,
        default="true",
        help="Whether to force rel_time window filtering before truncation.",
    )
    p.add_argument("--time_window_start", type=float, default=DEFAULT_TIME_WINDOW_START)
    p.add_argument("--time_window_end", type=float, default=DEFAULT_TIME_WINDOW_END)
    p.add_argument(
        "--write_meta_features",
        type=str,
        default="true",
        help="Whether to write meta_n_obs/meta_n_det/meta_n_det_snr5/meta_n_bands/meta_t_span/meta_single_band_id.",
    )
    p.add_argument(
        "--neg_match_pos_density",
        type=str,
        default="true",
        help="Whether to downsample negatives to match positive joint density in (n_det,n_bands,t_span).",
    )
    p.add_argument(
        "--density_match_pos_h5",
        type=str,
        default=None,
        help="Positive H5 reference for density matching (defaults to --output_pos_h5 when available).",
    )
    p.add_argument(
        "--density_bins_n_det",
        type=str,
        default=",".join(str(x) for x in DEFAULT_DENSITY_BINS_N_DET),
    )
    p.add_argument(
        "--density_bins_n_bands",
        type=str,
        default=",".join(str(x) for x in DEFAULT_DENSITY_BINS_N_BANDS),
    )
    p.add_argument(
        "--density_bins_t_span",
        type=str,
        default=",".join(str(x) for x in DEFAULT_DENSITY_BINS_T_SPAN),
    )
    p.add_argument("--fluxcal_zp", type=float, default=27.5)
    p.add_argument("--psfflux_zp", type=float, default=31.4)
    p.add_argument("--lupt_k", type=float, default=1.0)
    p.add_argument(
        "--lupt_m5_mag",
        type=str,
        default="23.9,25.0,24.7,24.0,23.3,22.1",
        help="Comma-separated 6 Rubin single-exposure m5 values (AB mag) in order u,g,r,i,z,Y.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes for FITS parsing. 1 means serial.",
    )
    return p


def main():
    args = build_parser().parse_args()

    lupt_m5_mag = parse_lupt_m5_mag(args.lupt_m5_mag)
    fluxcal_to_psfflux_factor, lupt_f5sigma_njy, lupt_b_njy = build_luptitude_params(
        fluxcal_zp=float(args.fluxcal_zp),
        psfflux_zp=float(args.psfflux_zp),
        lupt_k=float(args.lupt_k),
        lupt_m5_mag=lupt_m5_mag,
    )
    enforce_time_window = parse_bool_arg(args.enforce_time_window, name="--enforce_time_window")
    write_meta_features = parse_bool_arg(args.write_meta_features, name="--write_meta_features")
    neg_match_pos_density = parse_bool_arg(args.neg_match_pos_density, name="--neg_match_pos_density")
    time_window_start = float(args.time_window_start)
    time_window_end = float(args.time_window_end)
    if not (time_window_end > time_window_start):
        raise ValueError(
            f"time_window_end must be > time_window_start, got {time_window_start}..{time_window_end}"
        )
    density_bins_n_det = parse_bin_edges(
        args.density_bins_n_det, DEFAULT_DENSITY_BINS_N_DET, name="--density_bins_n_det"
    )
    density_bins_n_bands = parse_bin_edges(
        args.density_bins_n_bands, DEFAULT_DENSITY_BINS_N_BANDS, name="--density_bins_n_bands"
    )
    density_bins_t_span = parse_bin_edges(
        args.density_bins_t_span, DEFAULT_DENSITY_BINS_T_SPAN, name="--density_bins_t_span"
    )

    build_pos = bool(args.build_positive)
    build_neg = bool(args.build_negative)
    if not build_pos and not build_neg:
        if args.output_pos_h5 is not None:
            build_pos = True
        if args.output_neg_h5 is not None:
            build_neg = True
    if not build_pos and not build_neg:
        raise ValueError("Nothing to do. Set --build_positive/--build_negative or provide output paths.")

    if build_pos:
        if args.output_pos_h5 is None:
            raise ValueError("--output_pos_h5 is required when building positive dataset.")
        create_positive_h5(
            output_h5=Path(args.output_pos_h5),
            bns_sim_root=Path(args.bns_sim_root),
            bns_sim_name=args.bns_sim_name,
            nsbh_sim_root=Path(args.nsbh_sim_root),
            nsbh_sim_name=args.nsbh_sim_name,
            snr_threshold=float(args.snr_threshold),
            min_nobs=int(args.min_nobs),
            fixed_offset_days=float(args.fixed_offset_days),
            fluxcal_zp=float(args.fluxcal_zp),
            psfflux_zp=float(args.psfflux_zp),
            lupt_k=float(args.lupt_k),
            lupt_m5_mag=lupt_m5_mag,
            lupt_f5sigma_njy=lupt_f5sigma_njy,
            fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
            lupt_b_njy=lupt_b_njy,
            max_events_per_source=args.max_events_per_source,
            max_lcs_per_event=args.max_lcs_per_event,
            buffer_limit=int(args.buffer_limit),
            num_workers=int(args.num_workers),
            enforce_time_window=bool(enforce_time_window),
            time_window_start=float(time_window_start),
            time_window_end=float(time_window_end),
            write_meta_features=bool(write_meta_features),
        )

    if build_neg:
        if args.output_neg_h5 is None:
            raise ValueError("--output_neg_h5 is required when building negative dataset.")
        if args.neg_sim_root is None:
            raise ValueError("--neg_sim_root is required when building negative dataset.")
        density_match_pos_h5 = (
            Path(args.density_match_pos_h5)
            if args.density_match_pos_h5
            else (Path(args.output_pos_h5) if args.output_pos_h5 else None)
        )
        if neg_match_pos_density and density_match_pos_h5 is None:
            raise ValueError(
                "neg_match_pos_density=true requires --density_match_pos_h5 or --output_pos_h5."
            )
        create_negative_h5(
            output_h5=Path(args.output_neg_h5),
            neg_sim_root=Path(args.neg_sim_root),
            neg_group=args.neg_group,
            snr_threshold=float(args.snr_threshold),
            min_nobs=int(args.min_nobs),
            fixed_offset_days=float(args.fixed_offset_days),
            fluxcal_zp=float(args.fluxcal_zp),
            psfflux_zp=float(args.psfflux_zp),
            lupt_k=float(args.lupt_k),
            lupt_m5_mag=lupt_m5_mag,
            lupt_f5sigma_njy=lupt_f5sigma_njy,
            fluxcal_to_psfflux_factor=float(fluxcal_to_psfflux_factor),
            lupt_b_njy=lupt_b_njy,
            max_negative_heads=args.max_negative_heads,
            buffer_limit=int(args.buffer_limit),
            num_workers=int(args.num_workers),
            enforce_time_window=bool(enforce_time_window),
            time_window_start=float(time_window_start),
            time_window_end=float(time_window_end),
            write_meta_features=bool(write_meta_features),
            neg_match_pos_density=bool(neg_match_pos_density),
            density_match_pos_h5=density_match_pos_h5,
            density_bins_n_det=density_bins_n_det,
            density_bins_n_bands=density_bins_n_bands,
            density_bins_t_span=density_bins_t_span,
        )


if __name__ == "__main__":
    main()
