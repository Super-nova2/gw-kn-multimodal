from __future__ import annotations

from typing import Tuple

import numpy as np

MERGE_WINDOW_HOURS = 2.0
MERGE_WINDOW_DAYS = MERGE_WINDOW_HOURS / 24.0
MERGE_MODE = "inverse_variance_weighted_same_band_psfflux"
MERGE_FLUX_DOMAIN = "psfflux"


def merge_photometry_psfflux(
    mjd: np.ndarray,
    fluxcal: np.ndarray,
    fluxcalerr: np.ndarray,
    flt: np.ndarray,
    fluxcal_to_psfflux_factor: float,
    max_gap_days: float = MERGE_WINDOW_DAYS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mjd = np.asarray(mjd, dtype=np.float64).reshape(-1)
    fluxcal = np.asarray(fluxcal, dtype=np.float64).reshape(-1)
    fluxcalerr = np.asarray(fluxcalerr, dtype=np.float64).reshape(-1)
    flt = np.asarray(flt).reshape(-1)

    if not (mjd.shape == fluxcal.shape == fluxcalerr.shape == flt.shape):
        raise ValueError("mjd, fluxcal, fluxcalerr, and flt must have identical 1D shape.")
    if not np.isfinite(fluxcal_to_psfflux_factor) or fluxcal_to_psfflux_factor <= 0:
        raise ValueError("fluxcal_to_psfflux_factor must be finite and > 0.")
    if not np.isfinite(max_gap_days) or max_gap_days <= 0:
        raise ValueError("max_gap_days must be finite and > 0.")
    if mjd.size == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=flt.dtype),
        )

    valid = (
        np.isfinite(mjd)
        & np.isfinite(fluxcal)
        & np.isfinite(fluxcalerr)
        & (fluxcalerr > 0)
    )
    if not np.any(valid):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=flt.dtype),
        )

    mjd_valid = mjd[valid]
    psfflux_valid = fluxcal[valid] * float(fluxcal_to_psfflux_factor)
    psffluxerr_valid = np.abs(fluxcalerr[valid]) * float(fluxcal_to_psfflux_factor)
    flt_valid = flt[valid]

    valid_psf = (
        np.isfinite(psfflux_valid)
        & np.isfinite(psffluxerr_valid)
        & (psffluxerr_valid > 0)
    )
    if not np.any(valid_psf):
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=flt.dtype),
        )

    mjd_valid = mjd_valid[valid_psf]
    psfflux_valid = psfflux_valid[valid_psf]
    psffluxerr_valid = psffluxerr_valid[valid_psf]
    flt_valid = flt_valid[valid_psf]

    merged_mjd = []
    merged_psfflux = []
    merged_psffluxerr = []
    merged_flt = []

    ordered_bands = []
    for band in flt_valid.tolist():
        if band not in ordered_bands:
            ordered_bands.append(band)

    for band in ordered_bands:
        band_mask = flt_valid == band
        if not np.any(band_mask):
            continue
        band_mjd = mjd_valid[band_mask]
        band_flux = psfflux_valid[band_mask]
        band_err = psffluxerr_valid[band_mask]

        order = np.argsort(band_mjd, kind="mergesort")
        band_mjd = band_mjd[order]
        band_flux = band_flux[order]
        band_err = band_err[order]

        start = 0
        for idx in range(1, band_mjd.size + 1):
            close_cluster = (
                idx == band_mjd.size
                or (band_mjd[idx] - band_mjd[idx - 1]) > float(max_gap_days)
            )
            if not close_cluster:
                continue

            sl = slice(start, idx)
            weights = 1.0 / np.square(band_err[sl])
            weight_sum = float(np.sum(weights))
            if np.isfinite(weight_sum) and weight_sum > 0:
                merged_mjd.append(float(np.sum(weights * band_mjd[sl]) / weight_sum))
                merged_psfflux.append(float(np.sum(weights * band_flux[sl]) / weight_sum))
                merged_psffluxerr.append(float(np.sqrt(1.0 / weight_sum)))
                merged_flt.append(band)
            start = idx

    if not merged_mjd:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=flt.dtype),
        )

    merged_mjd_arr = np.asarray(merged_mjd, dtype=np.float64)
    merged_psfflux_arr = np.asarray(merged_psfflux, dtype=np.float64)
    merged_psffluxerr_arr = np.asarray(merged_psffluxerr, dtype=np.float64)
    merged_flt_arr = np.asarray(merged_flt, dtype=flt_valid.dtype if flt_valid.dtype != object else object)

    order = np.argsort(merged_mjd_arr, kind="mergesort")
    return (
        merged_mjd_arr[order],
        merged_psfflux_arr[order],
        merged_psffluxerr_arr[order],
        merged_flt_arr[order],
    )
