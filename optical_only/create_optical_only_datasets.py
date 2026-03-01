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
    if flux.size == 0:
        return None, None

    if photflag is not None:
        det_mask = photflag.astype(np.int64) != 0
        if np.any(det_mask):
            return int(np.argmax(det_mask)), "photflag"

    valid = np.isfinite(fluxerr) & (fluxerr > 0)
    if np.any(valid):
        snr = np.full(flux.shape, -np.inf, dtype=np.float64)
        snr[valid] = flux[valid] / fluxerr[valid]
        det_mask = snr > float(snr_threshold)
        if np.any(det_mask):
            return int(np.argmax(det_mask)), "snr"

    return None, None


def normalize_flux(
    flux: np.ndarray,
    fluxerr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    std = float(np.std(flux))
    mean = float(np.mean(flux))
    scale = std + 1e-8
    return (flux - mean) / scale, fluxerr / scale


def format_realization(
    mjd: np.ndarray,
    flux: np.ndarray,
    fluxerr: np.ndarray,
    flt: np.ndarray,
    ra: float,
    dec: float,
    t0_mjd: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rel_times = (mjd - float(t0_mjd)) / 100.0

    if len(mjd) > MAX_LC_LENGTH:
        keep = np.argsort(np.abs(rel_times))[:MAX_LC_LENGTH]
        keep = np.sort(keep)
        rel_times = rel_times[keep]
        flux = flux[keep]
        fluxerr = fluxerr[keep]
        flt = flt[keep]

    seq_len = min(len(rel_times), MAX_LC_LENGTH)
    val_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    err_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    mask_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
    time_vec = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)

    for t in range(seq_len):
        b_idx = band_index(flt[t])
        if b_idx is None:
            continue
        val_mat[t, b_idx] = flux[t]
        err_mat[t, b_idx] = fluxerr[t]
        mask_mat[t, b_idx] = 1.0
        time_vec[t] = rel_times[t]

    if not np.any(mask_mat > 0):
        raise ValueError("realization has no valid band after formatting")

    coords = np.array([ra, dec], dtype=np.float32)
    return val_mat, err_mat, mask_mat, time_vec, coords


def iter_event_dirs(base_dir: Path, sim_name: str) -> List[Path]:
    return sorted([p for p in base_dir.glob(f"{sim_name}_*") if p.is_dir()])


def parse_kn_event(
    event_dir: Path,
    sim_name: str,
    snr_threshold: float,
    min_nobs: int,
    fixed_offset_days: float,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], Dict[str, int]]:
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
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

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

                t0_mjd = float(lc_mjd[det_idx]) + float(fixed_offset_days)
                lc_flux, lc_fluxerr = normalize_flux(lc_flux, lc_fluxerr)

                try:
                    ra = float(data_head["RA"][i])
                    dec = float(data_head["DEC"][i])
                    out.append(
                        format_realization(
                            mjd=lc_mjd,
                            flux=lc_flux,
                            fluxerr=lc_fluxerr,
                            flt=lc_flt,
                            ra=ra,
                            dec=dec,
                            t0_mjd=t0_mjd,
                        )
                    )
                    stats["n_realizations_kept"] += 1
                except Exception:
                    stats["drop_empty_or_invalid"] += 1
                    continue
    except Exception:
        return out, stats

    return out, stats


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
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]], Dict[str, int]]:
    stats = {
        "n_realizations_total": 0,
        "n_realizations_kept": 0,
        "drop_nobs": 0,
        "drop_no_detection": 0,
        "drop_empty_or_invalid": 0,
    }
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

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

                t0_mjd = float(lc_mjd[det_idx]) + float(fixed_offset_days)
                lc_flux, lc_fluxerr = normalize_flux(lc_flux, lc_fluxerr)

                try:
                    ra = float(data_head["RA"][i])
                    dec = float(data_head["DEC"][i])
                    out.append(
                        format_realization(
                            mjd=lc_mjd,
                            flux=lc_flux,
                            fluxerr=lc_fluxerr,
                            flt=lc_flt,
                            ra=ra,
                            dec=dec,
                            t0_mjd=t0_mjd,
                        )
                    )
                    stats["n_realizations_kept"] += 1
                except Exception:
                    stats["drop_empty_or_invalid"] += 1
                    continue
    except Exception:
        return out, stats

    return out, stats


def _create_optical_group(
    grp,
    chunk_size: int,
):
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
    ds_coords = grp.create_dataset(
        "coordinates",
        (0, 2),
        maxshape=(None, 2),
        dtype="f4",
        chunks=(chunk_size, 2),
    )
    return ds_values, ds_errors, ds_masks, ds_times, ds_coords


def create_positive_h5(
    output_h5: Path,
    bns_sim_root: Path,
    bns_sim_name: str,
    nsbh_sim_root: Path,
    nsbh_sim_name: str,
    snr_threshold: float,
    min_nobs: int,
    fixed_offset_days: float,
    max_events_per_source: Optional[int],
    max_lcs_per_event: Optional[int],
    buffer_limit: int,
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
        ds_values, ds_errors, ds_masks, ds_times, ds_coords = _create_optical_group(opt_grp, chunk_size)
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
        b_coords: List[np.ndarray] = []
        b_parent: List[int] = []

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
            ds_coords.resize(new_size, axis=0)
            ds_parent.resize(new_size, axis=0)
            ds_values[cur:new_size] = np.asarray(b_vals, dtype=np.float32)
            ds_errors[cur:new_size] = np.asarray(b_errs, dtype=np.float32)
            ds_masks[cur:new_size] = np.asarray(b_masks, dtype=np.float32)
            ds_times[cur:new_size] = np.asarray(b_times, dtype=np.float32)
            ds_coords[cur:new_size] = np.asarray(b_coords, dtype=np.float32)
            ds_parent[cur:new_size] = np.asarray(b_parent, dtype=np.int32)

            optical_count = new_size
            b_vals.clear()
            b_errs.clear()
            b_masks.clear()
            b_times.clear()
            b_coords.clear()
            b_parent.clear()

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

            for event_dir in tqdm(event_dirs, desc=f"Positive {source_tag.upper()}"):
                lcs, event_stats = parse_kn_event(
                    event_dir=event_dir,
                    sim_name=sim_name,
                    snr_threshold=snr_threshold,
                    min_nobs=min_nobs,
                    fixed_offset_days=fixed_offset_days,
                )

                stats["drop_nobs"] += int(event_stats["drop_nobs"])
                stats["drop_no_detection"] += int(event_stats["drop_no_detection"])
                stats["drop_empty_or_invalid"] += int(event_stats["drop_empty_or_invalid"])

                if len(lcs) == 0:
                    continue

                if max_lcs_per_event is not None and len(lcs) > int(max_lcs_per_event):
                    n_drop = len(lcs) - int(max_lcs_per_event)
                    stats["drop_lcs_over_event_cap"] += int(n_drop)
                    lcs = lcs[: int(max_lcs_per_event)]

                gw_idx = gw_count
                ds_gw_ids.resize(gw_count + 1, axis=0)
                ds_gw_source.resize(gw_count + 1, axis=0)
                ds_gw_ids[gw_idx] = event_dir.name
                ds_gw_source[gw_idx] = source_tag
                gw_count += 1
                stats[key_written] += 1

                for vals, errs, masks, times, coords in lcs:
                    b_vals.append(vals)
                    b_errs.append(errs)
                    b_masks.append(masks)
                    b_times.append(times)
                    b_coords.append(coords)
                    b_parent.append(gw_idx)

                if len(b_vals) >= int(buffer_limit):
                    flush()

        flush()

        f.attrs["n_total_gw"] = int(gw_count)
        f.attrs["n_total_optical"] = int(optical_count)
        for k, v in stats.items():
            f.attrs[k] = int(v)

        f.attrs["time_zero_anchor"] = "first_detection"
        f.attrs["first_detection_rule"] = "photflag_nonzero_else_snr_gt_5"
        f.attrs["time_scale_divisor_days"] = 100.0
        f.attrs["time_zero_version"] = "fd_v1"
        f.attrs["drop_no_detection_count"] = int(stats["drop_no_detection"])
        f.attrs["snr_threshold"] = float(snr_threshold)
        f.attrs["detection_photflags"] = "nonzero"
        f.attrs["fixed_offset_days"] = float(fixed_offset_days)
        if max_lcs_per_event is not None:
            f.attrs["max_lcs_per_event"] = int(max_lcs_per_event)

    print(f"[POS] Saved: {output_h5}")


def create_negative_h5(
    output_h5: Path,
    neg_sim_root: Path,
    neg_group: str,
    snr_threshold: float,
    min_nobs: int,
    fixed_offset_days: float,
    max_negative_heads: Optional[int],
    buffer_limit: int,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = 1024
    dt_str = h5py.string_dtype(encoding="utf-8")

    with h5py.File(output_h5, "w") as f:
        grp = f.create_group(neg_group)
        ds_values, ds_errors, ds_masks, ds_times, ds_coords = _create_optical_group(grp, chunk_size)
        ds_types = grp.create_dataset("types", (0,), maxshape=(None,), dtype=dt_str, chunks=(chunk_size,))

        total_optical = 0
        b_vals: List[np.ndarray] = []
        b_errs: List[np.ndarray] = []
        b_masks: List[np.ndarray] = []
        b_times: List[np.ndarray] = []
        b_coords: List[np.ndarray] = []
        b_types: List[str] = []

        stats = {
            "head_files_total": 0,
            "head_files_used": 0,
            "drop_nobs": 0,
            "drop_no_detection": 0,
            "drop_empty_or_invalid": 0,
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
            ds_coords.resize(new_size, axis=0)
            ds_types.resize(new_size, axis=0)
            ds_values[cur:new_size] = np.asarray(b_vals, dtype=np.float32)
            ds_errors[cur:new_size] = np.asarray(b_errs, dtype=np.float32)
            ds_masks[cur:new_size] = np.asarray(b_masks, dtype=np.float32)
            ds_times[cur:new_size] = np.asarray(b_times, dtype=np.float32)
            ds_coords[cur:new_size] = np.asarray(b_coords, dtype=np.float32)
            ds_types[cur:new_size] = np.asarray(b_types, dtype=object)

            total_optical = new_size
            b_vals.clear()
            b_errs.clear()
            b_masks.clear()
            b_times.clear()
            b_coords.clear()
            b_types.clear()

        head_files = iter_negative_head_files(neg_sim_root)
        if max_negative_heads is not None:
            head_files = head_files[: int(max_negative_heads)]
        stats["head_files_total"] = int(len(head_files))

        for head_path in tqdm(head_files, desc="Negative HEAD files"):
            transient_type = infer_transient_type(head_path.parent.name)
            if transient_type is None:
                continue
            phot_path = find_pair_phot_file(head_path)
            if phot_path is None:
                continue

            lcs, file_stats = parse_negative_file(
                head_path=head_path,
                phot_path=phot_path,
                transient_type=transient_type,
                snr_threshold=snr_threshold,
                min_nobs=min_nobs,
                fixed_offset_days=fixed_offset_days,
            )
            stats["head_files_used"] += 1
            stats["drop_nobs"] += int(file_stats["drop_nobs"])
            stats["drop_no_detection"] += int(file_stats["drop_no_detection"])
            stats["drop_empty_or_invalid"] += int(file_stats["drop_empty_or_invalid"])

            for vals, errs, masks, times, coords in lcs:
                b_vals.append(vals)
                b_errs.append(errs)
                b_masks.append(masks)
                b_times.append(times)
                b_coords.append(coords)
                b_types.append(transient_type)

            if len(b_vals) >= int(buffer_limit):
                flush()

        flush()

        f.attrs["n_total_optical"] = int(total_optical)
        for k, v in stats.items():
            f.attrs[k] = int(v)
        f.attrs["time_zero_anchor"] = "first_detection"
        f.attrs["first_detection_rule"] = "photflag_nonzero_else_snr_gt_5"
        f.attrs["time_scale_divisor_days"] = 100.0
        f.attrs["time_zero_version"] = "fd_v1"
        f.attrs["drop_no_detection_count"] = int(stats["drop_no_detection"])
        f.attrs["snr_threshold"] = float(snr_threshold)
        f.attrs["detection_photflags"] = "nonzero"
        f.attrs["fixed_offset_days"] = float(fixed_offset_days)

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
    return p


def main():
    args = build_parser().parse_args()

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
            max_events_per_source=args.max_events_per_source,
            max_lcs_per_event=args.max_lcs_per_event,
            buffer_limit=int(args.buffer_limit),
        )

    if build_neg:
        if args.output_neg_h5 is None:
            raise ValueError("--output_neg_h5 is required when building negative dataset.")
        if args.neg_sim_root is None:
            raise ValueError("--neg_sim_root is required when building negative dataset.")
        create_negative_h5(
            output_h5=Path(args.output_neg_h5),
            neg_sim_root=Path(args.neg_sim_root),
            neg_group=args.neg_group,
            snr_threshold=float(args.snr_threshold),
            min_nobs=int(args.min_nobs),
            fixed_offset_days=float(args.fixed_offset_days),
            max_negative_heads=args.max_negative_heads,
            buffer_limit=int(args.buffer_limit),
        )


if __name__ == "__main__":
    main()
