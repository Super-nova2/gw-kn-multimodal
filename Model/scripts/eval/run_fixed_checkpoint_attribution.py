#!/usr/bin/env python3
"""Run one immutable fixed-checkpoint GW/optical attribution condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from retrieval_gallery import (  # noqa: E402
    aggregate_gallery_outcomes,
    build_prefixed_gallery_specs,
    build_synthetic_time_sky_candidate_sequences,
)
from data_loader import apply_runtime_input_window_torch  # noqa: E402
from scripts.eval.eval_retrieval_comparison import (  # noqa: E402
    _build_model_specs,
    _resolve_multimodal_scoring,
    enrich_gallery_outcomes,
    extract_negative_gallery_embeddings,
    extract_optical_candidate_embeddings,
    load_multimodal_bundle,
    load_selected_positive_bank,
    load_test_positive_index_map,
    remap_gallery_positive_indices,
    score_all_galleries_multimodal,
)
from scripts.eval.eval_run_io import (  # noqa: E402
    AtomicGzipCsvWriter,
    mark_run_success,
    prepare_output_directory,
    stable_digest,
    source_tree_digest,
    write_csv_atomic,
    write_json_atomic,
)
from scripts.eval.evaluate import (  # noqa: E402
    _resolve_eval_amp,
    load_gw_event_time_mjd_table,
    sample_negative_optical_source_indices,
)
GW_SCALAR_NAMES = (
    "mass1_detector",
    "mass2_detector",
    "spin1z",
    "spin2z",
    "costheta",
    "distmean_gpc",
    "diststd_gpc",
)
GW_SCALAR_INDEX = {name: idx for idx, name in enumerate(GW_SCALAR_NAMES)}
SKYMAP_DISTANCE_CHANNELS = (5, 6)
ASINH_MAG_FACTOR = 2.5 / np.log(10.0)
DEFAULT_PSFFLUX_ZP = 31.4
DEFAULT_LUPT_B_NJY = np.asarray(
    [200.0, 72.61557375, 95.72601982, 182.40220489, 347.56018001, 1049.61487237],
    dtype=np.float64,
)
MATCH_FEATURE_NAMES = (
    "log10_distance_gpc",
    "delay_days",
    "log1p_nobs",
    "band_fraction",
    "log1p_span_days",
)


@dataclass(frozen=True)
class AblationCondition:
    """Normalized definition of one fixed-checkpoint inference condition."""

    name: str
    gw_transform: str = "none"
    optical_transform: str = "none"
    time_delta_mode: str = "native"
    coordinate_mode: str = "native"


def normalize_condition(raw: Mapping[str, Any]) -> AblationCondition:
    """Validate a JSON condition and return its stable representation."""
    allowed = {
        "name",
        "gw_transform",
        "optical_transform",
        "time_delta_mode",
        "coordinate_mode",
    }
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"Unsupported condition fields: {sorted(unknown)}")
    name = str(raw.get("name", "")).strip()
    if not name:
        raise ValueError("Every attribution condition requires a non-empty name.")
    condition = AblationCondition(
        name=name,
        gw_transform=str(raw.get("gw_transform", "none")).strip().lower(),
        optical_transform=str(raw.get("optical_transform", "none")).strip().lower(),
        time_delta_mode=str(raw.get("time_delta_mode", "native")).strip().lower(),
        coordinate_mode=str(raw.get("coordinate_mode", "native")).strip().lower(),
    )
    valid_gw = {
        "none",
        "intrinsic_perm",
        "distance_perm",
        "mass1_perm",
        "mass2_perm",
        "spin1z_perm",
        "spin2z_perm",
        "inclination_abs_perm",
        "inclination_sign_flip",
        "skymap_distance_flat",
    }
    valid_optical = {
        "none",
        "brightness_norm",
        "per_band_norm",
        "time_shuffle",
        "peak_align",
    }
    if condition.gw_transform not in valid_gw:
        raise ValueError(f"Unsupported gw_transform={condition.gw_transform!r}")
    if condition.optical_transform not in valid_optical:
        raise ValueError(
            f"Unsupported optical_transform={condition.optical_transform!r}"
        )
    if condition.time_delta_mode not in {"native", "positive_shared", "zero"}:
        raise ValueError(f"Unsupported time_delta_mode={condition.time_delta_mode!r}")
    if condition.coordinate_mode not in {"native", "positive_shared"}:
        raise ValueError(
            f"Unsupported coordinate_mode={condition.coordinate_mode!r}"
        )
    return condition


def luptitude_to_flux_sigma(
    values: np.ndarray,
    errors: np.ndarray,
    *,
    psfflux_zp: float = DEFAULT_PSFFLUX_ZP,
    lupt_b_njy: Sequence[float] = DEFAULT_LUPT_B_NJY,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the repository's luptitude representation into nJy flux space."""
    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if values.shape != errors.shape or values.ndim < 1:
        raise ValueError("values and errors must have the same non-scalar shape.")
    b = np.asarray(lupt_b_njy, dtype=np.float64)
    if b.shape != (values.shape[-1],):
        raise ValueError(
            f"lupt_b_njy must match the final dimension {values.shape[-1]}."
        )
    shape = (1,) * (values.ndim - 1) + (values.shape[-1],)
    b = b.reshape(shape)
    two_b = 2.0 * b
    x = (float(psfflux_zp) - values) / ASINH_MAG_FACTOR - np.log(b)
    flux = two_b * np.sinh(x)
    sigma = (
        np.abs(errors)
        * np.sqrt(np.square(flux) + np.square(two_b))
        / ASINH_MAG_FACTOR
    )
    return flux, sigma


def flux_sigma_to_luptitude(
    flux: np.ndarray,
    sigma: np.ndarray,
    *,
    psfflux_zp: float = DEFAULT_PSFFLUX_ZP,
    lupt_b_njy: Sequence[float] = DEFAULT_LUPT_B_NJY,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert nJy flux and uncertainty to the model's luptitude inputs."""
    flux = np.asarray(flux, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if flux.shape != sigma.shape or flux.ndim < 1:
        raise ValueError("flux and sigma must have the same non-scalar shape.")
    b = np.asarray(lupt_b_njy, dtype=np.float64)
    if b.shape != (flux.shape[-1],):
        raise ValueError(
            f"lupt_b_njy must match the final dimension {flux.shape[-1]}."
        )
    shape = (1,) * (flux.ndim - 1) + (flux.shape[-1],)
    b = b.reshape(shape)
    values = float(psfflux_zp) - ASINH_MAG_FACTOR * (
        np.arcsinh(flux / (2.0 * b)) + np.log(b)
    )
    errors = (
        ASINH_MAG_FACTOR
        * np.abs(sigma)
        / np.sqrt(np.square(flux) + np.square(2.0 * b))
    )
    return values, errors


def _clone_optical_bank(bank: Mapping[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in bank.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().cpu().clone()
        elif isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        else:
            cloned[key] = value
    return cloned


def _as_numpy(value: Any, dtype=np.float64) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _restore_tensor(reference: Any, value: np.ndarray) -> Any:
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(value, dtype=reference.dtype)
    return value.astype(np.asarray(reference).dtype, copy=False)


def transform_optical_bank(
    bank: Mapping[str, Any],
    transform: str,
    *,
    seed: int,
    item_indices: Optional[Sequence[int]] = None,
    psfflux_zp: float = DEFAULT_PSFFLUX_ZP,
    lupt_b_njy: Sequence[float] = DEFAULT_LUPT_B_NJY,
    amplitude_quantile: float = 0.9,
    target_amplitude_njy: float = 100.0,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Apply a deterministic light-curve ablation and return audit rows.

    Brightness transforms operate in physical flux space and scale flux errors
    by the same factor, preserving per-observation S/N. ``brightness_norm``
    uses one scale per curve (colors retained); ``per_band_norm`` uses one scale
    per observed band (colors removed).
    """
    transform = str(transform or "none").strip().lower()
    if not np.isfinite(target_amplitude_njy) or target_amplitude_njy <= 0:
        raise ValueError("target_amplitude_njy must be finite and positive.")
    output = _clone_optical_bank(bank)
    if transform == "none":
        return output, []
    for key in ("times", "values", "masks", "errors"):
        if key not in output:
            raise KeyError(f"Optical bank is missing required key {key!r}")

    times = _as_numpy(output["times"])
    values = _as_numpy(output["values"])
    masks = _as_numpy(output["masks"])
    errors = _as_numpy(output["errors"])
    if values.shape != masks.shape or values.shape != errors.shape:
        raise ValueError("values, masks, and errors must have identical shapes.")
    if times.shape[:2] != values.shape[:2]:
        raise ValueError("times must share [N, L] dimensions with optical values.")
    valid = masks > 0
    if item_indices is None:
        item_indices_array = np.arange(values.shape[0], dtype=np.int64)
    else:
        item_indices_array = np.asarray(item_indices, dtype=np.int64).reshape(-1)
        if item_indices_array.size != values.shape[0]:
            raise ValueError("item_indices must match the optical bank length.")
    audit: list[Dict[str, Any]] = []

    if transform in {"brightness_norm", "per_band_norm"}:
        flux, sigma = luptitude_to_flux_sigma(
            values, errors, psfflux_zp=psfflux_zp, lupt_b_njy=lupt_b_njy
        )
        for idx in range(values.shape[0]):
            item_index = int(item_indices_array[idx])
            scales = np.ones((values.shape[-1],), dtype=np.float64)
            if transform == "brightness_norm":
                observed = np.abs(flux[idx][valid[idx]])
                amplitude = (
                    float(np.quantile(observed, amplitude_quantile))
                    if observed.size
                    else 1.0
                )
                scale = float(target_amplitude_njy) / max(amplitude, 1e-12)
                scales[:] = scale
            else:
                for band in range(values.shape[-1]):
                    observed = np.abs(flux[idx, :, band][valid[idx, :, band]])
                    amplitude = (
                        float(np.quantile(observed, amplitude_quantile))
                        if observed.size
                        else 1.0
                    )
                    scales[band] = float(target_amplitude_njy) / max(
                        amplitude, 1e-12
                    )
            flux[idx] *= scales.reshape(1, -1)
            sigma[idx] *= scales.reshape(1, -1)
            audit.append(
                {
                    "item_index": item_index,
                    "transform": transform,
                    "scale_min": float(np.min(scales)),
                    "scale_median": float(np.median(scales)),
                    "scale_max": float(np.max(scales)),
                }
            )
        new_values, new_errors = flux_sigma_to_luptitude(
            flux, sigma, psfflux_zp=psfflux_zp, lupt_b_njy=lupt_b_njy
        )
        values[valid] = new_values[valid]
        errors[valid] = new_errors[valid]
    elif transform == "time_shuffle":
        for idx in range(values.shape[0]):
            item_index = int(item_indices_array[idx])
            valid_rows = np.flatnonzero(np.any(valid[idx], axis=-1))
            if valid_rows.size > 1:
                rng = np.random.default_rng(int(seed) + 104729 * item_index)
                shuffled = times[idx, valid_rows].copy()
                rng.shuffle(shuffled)
                times[idx, valid_rows] = shuffled
            audit.append(
                {
                    "item_index": item_index,
                    "transform": transform,
                    "n_valid_rows": int(valid_rows.size),
                }
            )
    elif transform == "peak_align":
        flux, _ = luptitude_to_flux_sigma(
            values, errors, psfflux_zp=psfflux_zp, lupt_b_njy=lupt_b_njy
        )
        for idx in range(values.shape[0]):
            item_index = int(item_indices_array[idx])
            if np.any(valid[idx]):
                peak_flat = int(np.argmax(np.where(valid[idx], flux[idx], -np.inf)))
                peak_row = np.unravel_index(peak_flat, flux[idx].shape)[0]
                peak_time = float(times[idx, peak_row])
                valid_rows = np.any(valid[idx], axis=-1)
                times[idx, valid_rows] -= peak_time
            else:
                peak_time = 0.0
            audit.append(
                {
                    "item_index": item_index,
                    "transform": transform,
                    "peak_time_shift_days": peak_time,
                }
            )
    else:
        raise ValueError(f"Unsupported optical transform: {transform!r}")

    output["times"] = _restore_tensor(bank["times"], times)
    output["values"] = _restore_tensor(bank["values"], values)
    output["errors"] = _restore_tensor(bank["errors"], errors)
    for alias, source in (
        ("opt_t_raw", "times"),
        ("opt_v_raw", "values"),
        ("opt_err_raw", "errors"),
        ("opt_mask_raw", "masks"),
    ):
        if alias in output:
            output[alias] = output[source]
    return output, audit


def robust_standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median/IQR standardization with deterministic zero-spread handling."""
    features = np.asarray(features, dtype=np.float64)
    median = np.nanmedian(features, axis=0)
    q25, q75 = np.nanpercentile(features, [25.0, 75.0], axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    standardized = (features - median) / scale
    if not np.all(np.isfinite(standardized)):
        raise ValueError("Non-finite values in robust-standardized features.")
    return standardized, median, scale


def minimum_cost_derangement(features: np.ndarray) -> np.ndarray:
    """Return a globally minimum-cost donor permutation with no fixed points."""
    standardized, _, _ = robust_standardize(features)
    n_rows = standardized.shape[0]
    if n_rows < 2:
        raise ValueError("A derangement requires at least two rows.")
    cost = np.abs(standardized[:, None, :] - standardized[None, :, :]).sum(axis=2)
    finite_max = float(np.max(cost)) if cost.size else 1.0
    np.fill_diagonal(cost, finite_max + 1e9)
    # Stable donor-index epsilon makes exact ties reproducible across SciPy builds.
    cost += np.arange(n_rows, dtype=np.float64).reshape(1, -1) * 1e-12
    row_idx, donor_idx = linear_sum_assignment(cost)
    order = np.empty((n_rows,), dtype=np.int64)
    order[row_idx] = donor_idx
    if np.any(order == np.arange(n_rows)) or np.unique(order).size != n_rows:
        raise AssertionError("Failed to construct a valid derangement.")
    return order


def source_conditional_derangement(
    gw_ids: Sequence[int],
    source_types: Sequence[str],
    conditioning_features: np.ndarray,
) -> Dict[int, int]:
    """Build independent minimum-cost derangements within BNS/NSBH groups."""
    gw_ids = np.asarray(gw_ids, dtype=np.int64)
    source_types = np.asarray([str(value).lower() for value in source_types])
    features = np.asarray(conditioning_features, dtype=np.float64)
    if features.shape[0] != gw_ids.size or source_types.size != gw_ids.size:
        raise ValueError("gw_ids, source_types, and features must have equal length.")
    donors: Dict[int, int] = {}
    for source in sorted(np.unique(source_types).tolist()):
        group = np.flatnonzero(source_types == source)
        local_order = minimum_cost_derangement(features[group])
        for query_local, donor_local in enumerate(local_order.tolist()):
            donors[int(gw_ids[group[query_local]])] = int(gw_ids[group[donor_local]])
    return donors


class GWInputTransform:
    """Callable GW input transform with a query-to-donor audit trail."""

    def __init__(
        self,
        mode: str,
        gw_ids: Sequence[int],
        scalars: np.ndarray,
        skymaps: Optional[np.ndarray],
        source_types: Sequence[str],
    ) -> None:
        self.mode = str(mode or "none").strip().lower()
        self.gw_ids = np.asarray(gw_ids, dtype=np.int64)
        self.scalars = np.asarray(scalars, dtype=np.float32)
        self.distance_skymaps: Optional[np.ndarray] = None
        if skymaps is not None:
            skymaps = np.asarray(skymaps, dtype=np.float32)
            if skymaps.ndim != 3 or skymaps.shape[0] != self.gw_ids.size:
                raise ValueError("Unexpected GW skymap table shape.")
            if skymaps.shape[1] == len(SKYMAP_DISTANCE_CHANNELS):
                self.distance_skymaps = skymaps
            elif skymaps.shape[1] > max(SKYMAP_DISTANCE_CHANNELS):
                self.distance_skymaps = skymaps[:, SKYMAP_DISTANCE_CHANNELS, :]
            else:
                raise ValueError("GW skymaps do not contain distance channels 5/6.")
        self.source_types = np.asarray([str(v).lower() for v in source_types])
        self.position = {int(gw_id): idx for idx, gw_id in enumerate(self.gw_ids)}
        self.donors: Dict[int, int] = {}
        self.audit_rows: list[Dict[str, Any]] = []
        if self.scalars.shape != (self.gw_ids.size, len(GW_SCALAR_NAMES)):
            raise ValueError("Unexpected GW scalar table shape.")
        if self.mode == "distance_perm" and self.distance_skymaps is None:
            raise ValueError(
                "distance_perm requires the selected GW skymap distance channels."
            )
        self._prepare_donors()

    def _prepare_donors(self) -> None:
        if not self.mode.endswith("_perm"):
            return
        scalar = self.scalars.astype(np.float64)
        log_distance = np.log10(np.maximum(scalar[:, 5], 1e-8))
        log_distance_std = np.log10(np.maximum(scalar[:, 6], 1e-8))
        intrinsic = np.column_stack(
            [scalar[:, 0], scalar[:, 1], scalar[:, 2], scalar[:, 3], np.abs(scalar[:, 4])]
        )
        if self.mode in {
            "intrinsic_perm",
            "mass1_perm",
            "mass2_perm",
            "spin1z_perm",
            "spin2z_perm",
            "inclination_abs_perm",
        }:
            if self.mode == "intrinsic_perm":
                conditioning = np.column_stack([log_distance, log_distance_std])
            else:
                target_index = {
                    "mass1_perm": 0,
                    "mass2_perm": 1,
                    "spin1z_perm": 2,
                    "spin2z_perm": 3,
                    "inclination_abs_perm": 4,
                }[self.mode]
                conditioning = np.column_stack(
                    [np.delete(intrinsic, target_index, axis=1), log_distance, log_distance_std]
                )
        elif self.mode == "distance_perm":
            conditioning = intrinsic
        else:
            raise ValueError(f"Unsupported permutation mode {self.mode!r}")
        self.donors = source_conditional_derangement(
            self.gw_ids, self.source_types, conditioning
        )

    def __call__(
        self, gw_id: int, scalars: np.ndarray, skymap: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        gw_id = int(gw_id)
        if gw_id not in self.position:
            raise KeyError(f"gw_id={gw_id} is absent from the transform table.")
        scalars = np.asarray(scalars, dtype=np.float32).copy()
        skymap = np.asarray(skymap, dtype=np.float32).copy()
        scalars_before = scalars.copy()
        distance_skymap_before = skymap[list(SKYMAP_DISTANCE_CHANNELS)].copy()
        donor_id: Optional[int] = None
        if self.mode.endswith("_perm"):
            donor_id = self.donors[gw_id]
            donor_pos = self.position[donor_id]
            donor_scalars = self.scalars[donor_pos]
            if self.mode == "intrinsic_perm":
                scalars[:5] = donor_scalars[:5]
            elif self.mode == "distance_perm":
                scalars[5:7] = donor_scalars[5:7]
                if self.distance_skymaps is None:
                    raise AssertionError("Missing distance skymap donor table.")
                skymap[list(SKYMAP_DISTANCE_CHANNELS)] = self.distance_skymaps[
                    donor_pos
                ]
            elif self.mode == "inclination_abs_perm":
                sign = -1.0 if scalars[4] < 0 else 1.0
                scalars[4] = sign * abs(float(donor_scalars[4]))
            else:
                scalar_name = {
                    "mass1_perm": "mass1_detector",
                    "mass2_perm": "mass2_detector",
                    "spin1z_perm": "spin1z",
                    "spin2z_perm": "spin2z",
                }[self.mode]
                scalars[GW_SCALAR_INDEX[scalar_name]] = donor_scalars[
                    GW_SCALAR_INDEX[scalar_name]
                ]
        elif self.mode == "inclination_sign_flip":
            scalars[4] *= -1.0
        elif self.mode == "skymap_distance_flat":
            weights = np.maximum(skymap[4].astype(np.float64), 0.0)
            if float(weights.sum()) <= 0:
                weights = np.ones_like(weights)
            for channel in SKYMAP_DISTANCE_CHANNELS:
                mean = float(np.average(skymap[channel], weights=weights))
                skymap[channel] = mean
        elif self.mode != "none":
            raise ValueError(f"Unsupported GW transform {self.mode!r}")
        self.audit_rows.append(
            {
                "gw_id": gw_id,
                "source_type": str(self.source_types[self.position[gw_id]]),
                "transform": self.mode,
                "donor_gw_id": "" if donor_id is None else donor_id,
                "mass1_after": float(scalars[0]),
                "mass2_after": float(scalars[1]),
                "spin1z_after": float(scalars[2]),
                "spin2z_after": float(scalars[3]),
                "costheta_after": float(scalars[4]),
                "distmean_after_gpc": float(scalars[5]),
                "diststd_after_gpc": float(scalars[6]),
                "scalar_delta_l1": float(
                    np.abs(scalars - scalars_before).sum(dtype=np.float64)
                ),
                "skymap_distance_delta_mean_abs": float(
                    np.abs(
                        skymap[list(SKYMAP_DISTANCE_CHANNELS)]
                        - distance_skymap_before
                    ).mean(dtype=np.float64)
                ),
            }
        )
        return scalars, skymap


def optical_nuisance_features(
    optical_bank: Mapping[str, Any],
    parent_gw_idx: Sequence[int],
    gw_scalars: np.ndarray,
    gw_event_time_mjd: Sequence[float],
) -> np.ndarray:
    """Compute nuisance features used for same-source KN matching."""
    times = _as_numpy(optical_bank["times"])
    masks = _as_numpy(optical_bank["masks"])
    parent = np.asarray(parent_gw_idx, dtype=np.int64)
    scalars = np.asarray(gw_scalars, dtype=np.float64)
    event_time = np.asarray(gw_event_time_mjd, dtype=np.float64)
    first_detection = _as_numpy(optical_bank["first_detection_mjd"]).reshape(-1)
    if parent.size != times.shape[0]:
        raise ValueError("parent_gw_idx length does not match optical bank.")
    features = np.zeros((parent.size, len(MATCH_FEATURE_NAMES)), dtype=np.float64)
    for idx, gw_id in enumerate(parent.tolist()):
        valid = masks[idx] > 0
        valid_rows = np.any(valid, axis=-1)
        nobs = int(np.count_nonzero(valid))
        nband = int(np.count_nonzero(np.any(valid, axis=0)))
        observed_times = times[idx, valid_rows]
        span = (
            float(np.max(observed_times) - np.min(observed_times))
            if observed_times.size > 1
            else 0.0
        )
        delay = abs(float(first_detection[idx]) - float(event_time[gw_id]))
        features[idx] = (
            np.log10(max(float(scalars[gw_id, 5]), 1e-8)),
            delay,
            np.log1p(nobs),
            nband / float(masks.shape[-1]),
            np.log1p(max(span, 0.0)),
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite nuisance features in KN candidate bank.")
    return features


def build_kn_nuisance_matched_galleries(
    *,
    gw_positive_indices: Mapping[int, Sequence[int]],
    optical_parent_gw_idx: Sequence[int],
    gw_source_types: Sequence[str],
    nuisance_features: np.ndarray,
    gallery_sizes: Sequence[int],
    n_trials: int,
    seed: int,
) -> tuple[Dict[tuple[int, int, int], Dict[str, Any]], list[int]]:
    """Build deterministic, prefix-nested KN-vs-KN nuisance-matched galleries.

    Every negative is a KN light curve of the same source class but from a
    different parent event. A gallery never repeats a parent GW event.
    """
    parent = np.asarray(optical_parent_gw_idx, dtype=np.int64)
    source = np.asarray([str(v).lower() for v in gw_source_types])
    features = np.asarray(nuisance_features, dtype=np.float64)
    standardized = np.empty_like(features)
    for label in sorted(np.unique(source).tolist()):
        optical_rows = np.flatnonzero(source[parent] == label)
        standardized[optical_rows], _, _ = robust_standardize(features[optical_rows])
    gallery_sizes = sorted({int(value) for value in gallery_sizes})
    if not gallery_sizes or min(gallery_sizes) < 2:
        raise ValueError("KN matched galleries require sizes >= 2.")
    max_neg = max(gallery_sizes) - 1
    galleries: Dict[tuple[int, int, int], Dict[str, Any]] = {}
    used_gw = sorted(int(value) for value in gw_positive_indices)

    for trial in range(int(n_trials)):
        rng = np.random.default_rng(int(seed) + 15485863 * trial)
        for gw_id in used_gw:
            positive_pool = np.asarray(gw_positive_indices[gw_id], dtype=np.int64)
            positive_index = int(positive_pool[rng.integers(positive_pool.size)])
            candidates = np.flatnonzero(
                (source[parent] == source[gw_id]) & (parent != gw_id)
            )
            distance = np.abs(
                standardized[candidates] - standardized[positive_index]
            ).sum(axis=1)
            order = np.lexsort((candidates, distance))
            best_by_parent: list[int] = []
            match_costs: list[float] = []
            seen_parent: set[int] = set()
            for local_idx in order.tolist():
                candidate_idx = int(candidates[local_idx])
                candidate_parent = int(parent[candidate_idx])
                if candidate_parent in seen_parent:
                    continue
                seen_parent.add(candidate_parent)
                best_by_parent.append(candidate_idx)
                match_costs.append(float(distance[local_idx]))
                if len(best_by_parent) >= max_neg:
                    break
            if len(best_by_parent) < max_neg:
                raise ValueError(
                    f"gw_id={gw_id} has only {len(best_by_parent)} unique matched "
                    f"parents; need {max_neg}."
                )
            best = np.asarray(best_by_parent, dtype=np.int64)
            costs = np.asarray(match_costs, dtype=np.float32)
            for gallery_size in gallery_sizes:
                take = gallery_size - 1
                galleries[(gallery_size, trial, gw_id)] = {
                    "positive_index": positive_index,
                    "negative_indices": best[:take].copy(),
                    "negative_match_costs": costs[:take].copy(),
                    "negative_credible_levels": np.full(take, np.nan, np.float32),
                    "negative_abs_dt_days": np.full(take, np.nan, np.float32),
                    "requested_gallery_size": gallery_size,
                    "actual_gallery_size": gallery_size,
                    "coverage_met": True,
                    "is_undersized": False,
                }
    validate_kn_matched_galleries(galleries, parent)
    return galleries, used_gw


def build_kn_random_same_source_galleries(
    *,
    gw_positive_indices: Mapping[int, Sequence[int]],
    optical_parent_gw_idx: Sequence[int],
    gw_source_types: Sequence[str],
    nuisance_features: np.ndarray,
    gallery_sizes: Sequence[int],
    n_trials: int,
    seed: int,
) -> tuple[Dict[tuple[int, int, int], Dict[str, Any]], list[int]]:
    """Build a same-source random-parent control for nuisance-nearest galleries.

    Positives are copied from the deterministic nuisance-matched construction,
    so the two tasks differ only in their negative selection. Random negatives
    have distinct parent GW events and remain prefix-nested across gallery sizes.
    """
    parent = np.asarray(optical_parent_gw_idx, dtype=np.int64)
    source = np.asarray([str(value).lower() for value in gw_source_types])
    features = np.asarray(nuisance_features, dtype=np.float64)
    gallery_sizes = sorted({int(value) for value in gallery_sizes})
    if not gallery_sizes or min(gallery_sizes) < 2:
        raise ValueError("KN random galleries require sizes >= 2.")
    max_gallery_size = max(gallery_sizes)
    max_negatives = max_gallery_size - 1

    reference, used_gw = build_kn_nuisance_matched_galleries(
        gw_positive_indices=gw_positive_indices,
        optical_parent_gw_idx=parent,
        gw_source_types=source,
        nuisance_features=features,
        gallery_sizes=gallery_sizes,
        n_trials=n_trials,
        seed=seed,
    )
    standardized = np.empty_like(features)
    for label in sorted(np.unique(source).tolist()):
        optical_rows = np.flatnonzero(source[parent] == label)
        standardized[optical_rows], _, _ = robust_standardize(features[optical_rows])

    optical_by_parent = {
        int(parent_id): np.flatnonzero(parent == int(parent_id))
        for parent_id in np.unique(parent).tolist()
    }
    detected_parents = np.asarray(sorted(optical_by_parent), dtype=np.int64)
    galleries: Dict[tuple[int, int, int], Dict[str, Any]] = {}
    for trial in range(int(n_trials)):
        for gw_id in used_gw:
            positive_index = int(
                reference[(max_gallery_size, trial, gw_id)]["positive_index"]
            )
            eligible_parents = detected_parents[
                (source[detected_parents] == source[gw_id])
                & (detected_parents != int(gw_id))
            ]
            if eligible_parents.size < max_negatives:
                raise ValueError(
                    f"gw_id={gw_id} has only {eligible_parents.size} random "
                    f"same-source parents; need {max_negatives}."
                )
            rng = np.random.default_rng(
                int(seed) + 32452843 * int(trial) + 49979687 * int(gw_id)
            )
            selected_parents = rng.choice(
                eligible_parents, size=max_negatives, replace=False
            )
            selected_rows = np.asarray(
                [
                    int(rows[rng.integers(rows.size)])
                    for rows in (optical_by_parent[int(value)] for value in selected_parents)
                ],
                dtype=np.int64,
            )
            costs = np.abs(
                standardized[selected_rows] - standardized[positive_index]
            ).sum(axis=1).astype(np.float32)
            for gallery_size in gallery_sizes:
                take = gallery_size - 1
                galleries[(gallery_size, trial, gw_id)] = {
                    "positive_index": positive_index,
                    "negative_indices": selected_rows[:take].copy(),
                    "negative_match_costs": costs[:take].copy(),
                    "negative_credible_levels": np.full(take, np.nan, np.float32),
                    "negative_abs_dt_days": np.full(take, np.nan, np.float32),
                    "requested_gallery_size": gallery_size,
                    "actual_gallery_size": gallery_size,
                    "coverage_met": True,
                    "is_undersized": False,
                }
    validate_kn_matched_galleries(galleries, parent)
    return galleries, used_gw


def validate_kn_matched_galleries(
    galleries: Mapping[tuple[int, int, int], Mapping[str, Any]],
    optical_parent_gw_idx: Sequence[int],
) -> None:
    """Reject parent leakage, duplicate parents, and broken prefix nesting."""
    parent = np.asarray(optical_parent_gw_idx, dtype=np.int64)
    grouped: Dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    for (gallery_size, trial, gw_id), spec in galleries.items():
        negative = np.asarray(spec["negative_indices"], dtype=np.int64)
        negative_parent = parent[negative]
        if np.any(negative_parent == int(gw_id)):
            raise ValueError(f"Same-parent leakage in gw_id={gw_id}, trial={trial}.")
        if np.unique(negative_parent).size != negative_parent.size:
            raise ValueError(f"Duplicate negative parent in gw_id={gw_id}, trial={trial}.")
        grouped.setdefault((int(trial), int(gw_id)), []).append(
            (int(gallery_size), negative)
        )
    for key, specs in grouped.items():
        specs.sort(key=lambda item: item[0])
        for (_, smaller), (_, larger) in zip(specs, specs[1:]):
            if not np.array_equal(smaller, larger[: smaller.size]):
                raise ValueError(f"Gallery prefix nesting failed for trial/gw={key}.")


def gallery_identity_digest(
    galleries: Mapping[tuple[int, int, int], Mapping[str, Any]]
) -> str:
    """Hash query, positive, and ordered negative identities."""
    digest = hashlib.sha256()
    for key, spec in sorted(galleries.items()):
        digest.update(np.asarray(key, dtype=np.int64).tobytes())
        positive = spec.get("source_positive_index", spec["positive_index"])
        negative = spec.get("source_negative_indices", spec["negative_indices"])
        digest.update(np.asarray([positive], dtype=np.int64).tobytes())
        digest.update(np.asarray(negative, dtype=np.int64).tobytes())
    return digest.hexdigest()


def summarize_match_balance(
    galleries: Mapping[tuple[int, int, int], Mapping[str, Any]],
    nuisance_features: np.ndarray,
) -> list[Dict[str, Any]]:
    """Summarize raw nuisance imbalance for each gallery size and feature."""
    features = np.asarray(nuisance_features, dtype=np.float64)
    buckets: Dict[tuple[int, str], list[float]] = {}
    for (gallery_size, _trial, _gw_id), spec in galleries.items():
        positive = int(spec["positive_index"])
        negative = np.asarray(spec["negative_indices"], dtype=np.int64)
        delta = np.abs(features[negative] - features[positive])
        for column, name in enumerate(MATCH_FEATURE_NAMES):
            buckets.setdefault((int(gallery_size), name), []).extend(
                delta[:, column].tolist()
            )
    rows: list[Dict[str, Any]] = []
    for (gallery_size, name), values in sorted(buckets.items()):
        array = np.asarray(values, dtype=np.float64)
        rows.append(
            {
                "gallery_size": gallery_size,
                "feature": name,
                "n_pairs": int(array.size),
                "mean_abs_delta": float(np.mean(array)),
                "median_abs_delta": float(np.median(array)),
                "p95_abs_delta": float(np.percentile(array, 95.0)),
            }
        )
    return rows


def effective_physical_parameters(
    source_type: str, row: Mapping[str, Any]
) -> Dict[str, float]:
    """Return Bulla-template-effective ejecta and viewing parameters."""
    source = str(source_type).lower()
    dyn = float(row["mej_dyn"])
    wind = float(row["mej_wind"])
    if source == "bns":
        dyn = float(np.clip(dyn, 0.001, 0.02))
        wind = float(np.clip(wind, 0.01, 0.13))
    elif source == "nsbh":
        dyn = float(np.clip(dyn, 0.01, 0.09))
        wind = float(np.clip(wind, 0.01, 0.09))
    else:
        raise ValueError(f"Unsupported source_type={source_type!r}")
    return {
        "effective_mej_dyn": dyn,
        "effective_mej_wind": wind,
        "effective_mej_total": dyn + wind,
        "abs_costheta": abs(float(row["costheta"])),
    }


def physical_mismatch(
    query: Mapping[str, float], candidate: Mapping[str, float]
) -> Dict[str, float]:
    """Compute interpretable template-effective candidate mismatches."""
    return {
        "delta_effective_mej_dyn": abs(
            float(query["effective_mej_dyn"])
            - float(candidate["effective_mej_dyn"])
        ),
        "delta_effective_mej_wind": abs(
            float(query["effective_mej_wind"])
            - float(candidate["effective_mej_wind"])
        ),
        "delta_effective_mej_total": abs(
            float(query["effective_mej_total"])
            - float(candidate["effective_mej_total"])
        ),
        "delta_abs_costheta": abs(
            float(query["abs_costheta"]) - float(candidate["abs_costheta"])
        ),
    }


def iter_candidate_indices(spec: Mapping[str, Any]) -> Iterable[tuple[int, bool]]:
    """Yield positive then ordered negatives for one gallery specification."""
    yield int(spec["positive_index"]), True
    for value in np.asarray(spec["negative_indices"], dtype=np.int64).tolist():
        yield int(value), False

OUTCOME_FIELDS = (
    "task",
    "condition",
    "seed",
    "trial",
    "gw_id",
    "source_type",
    "gallery_size",
    "actual_gallery_size",
    "rank_zero_based",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "coverage",
    "positive_score",
    "best_negative_score",
    "score_margin",
)

EVAL_SEEDS = (42, 123, 456)
OPERATIONAL_CONDITIONS: tuple[Dict[str, str], ...] = (
    {"name": "native"},
    {"name": "dt_zero", "time_delta_mode": "zero"},
    {"name": "dt_shared", "time_delta_mode": "positive_shared"},
    {"name": "coord_shared", "coordinate_mode": "positive_shared"},
    {
        "name": "controlled",
        "time_delta_mode": "positive_shared",
        "coordinate_mode": "positive_shared",
    },
    *tuple(
        {
            "name": name,
            "gw_transform": transform,
            "time_delta_mode": "positive_shared",
            "coordinate_mode": "positive_shared",
        }
        for name, transform in (
            ("intrinsic_perm", "intrinsic_perm"),
            ("distance_perm", "distance_perm"),
            ("skymap_distance_flat", "skymap_distance_flat"),
        )
    ),
    *tuple(
        {
            "name": name,
            "optical_transform": transform,
            "time_delta_mode": "positive_shared",
            "coordinate_mode": "positive_shared",
        }
        for name, transform in (
            ("brightness_norm", "brightness_norm"),
            ("per_band_norm", "per_band_norm"),
            ("time_shuffle", "time_shuffle"),
            ("peak_align", "peak_align"),
        )
    ),
)
MATCHED_CONDITIONS: tuple[Dict[str, str], ...] = (
    {
        "name": "baseline",
        "time_delta_mode": "positive_shared",
        "coordinate_mode": "positive_shared",
    },
    *tuple(
        {
            "name": name,
            "gw_transform": transform,
            "time_delta_mode": "positive_shared",
            "coordinate_mode": "positive_shared",
        }
        for name, transform in (
            ("intrinsic_perm", "intrinsic_perm"),
            ("mass1_perm", "mass1_perm"),
            ("mass2_perm", "mass2_perm"),
            ("spin1z_perm", "spin1z_perm"),
            ("spin2z_perm", "spin2z_perm"),
            ("inclination_abs_perm", "inclination_abs_perm"),
            ("inclination_sign_flip", "inclination_sign_flip"),
            ("distance_perm", "distance_perm"),
            ("skymap_distance_flat", "skymap_distance_flat"),
        )
    ),
    *tuple(
        {
            "name": name,
            "optical_transform": transform,
            "time_delta_mode": "positive_shared",
            "coordinate_mode": "positive_shared",
        }
        for name, transform in (
            ("brightness_norm", "brightness_norm"),
            ("per_band_norm", "per_band_norm"),
            ("time_shuffle", "time_shuffle"),
            ("peak_align", "peak_align"),
        )
    ),
    *tuple(
        {
            "name": name,
            "gw_transform": gw_transform,
            "optical_transform": optical_transform,
            "time_delta_mode": "positive_shared",
            "coordinate_mode": "positive_shared",
        }
        for name, gw_transform, optical_transform in (
            ("distance_perm__brightness_norm", "distance_perm", "brightness_norm"),
            ("spin1z_perm__time_shuffle", "spin1z_perm", "time_shuffle"),
            (
                "inclination_abs_perm__per_band_norm",
                "inclination_abs_perm",
                "per_band_norm",
            ),
            (
                "inclination_abs_perm__time_shuffle",
                "inclination_abs_perm",
                "time_shuffle",
            ),
        )
    ),
)
RANDOM_MATCHED_CONDITIONS: tuple[Dict[str, str], ...] = (
    {
        "name": "baseline",
        "time_delta_mode": "positive_shared",
        "coordinate_mode": "positive_shared",
    },
)
SCORE_SUMMARY_FIELDS = (
    "task",
    "condition",
    "seed",
    "source_type",
    "gallery_size",
    "n_queries",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "coverage",
    "positive_score_mean",
    "best_negative_score_mean",
    "score_margin_mean",
)
CANDIDATE_FIELDS = (
    "task",
    "condition",
    "seed",
    "trial",
    "gw_id",
    "source_type",
    "gallery_size",
    "candidate_position",
    "candidate_optical_index",
    "candidate_parent_gw_id",
    "is_positive",
    "score",
    "match_cost",
    "delta_effective_mej_dyn",
    "delta_effective_mej_wind",
    "delta_effective_mej_total",
    "delta_abs_costheta",
)
TRANSFORM_AUDIT_FIELDS = (
    "audit_type",
    "bank",
    "item_index",
    "gw_id",
    "source_type",
    "transform",
    "donor_gw_id",
    "mass1_after",
    "mass2_after",
    "spin1z_after",
    "spin2z_after",
    "costheta_after",
    "distmean_after_gpc",
    "diststd_after_gpc",
    "scalar_delta_l1",
    "skymap_distance_delta_mean_abs",
    "scale_min",
    "scale_median",
    "scale_max",
    "n_valid_rows",
    "peak_time_shift_days",
)
MATCH_BALANCE_FIELDS = (
    "gallery_size",
    "feature",
    "n_pairs",
    "mean_abs_delta",
    "median_abs_delta",
    "p95_abs_delta",
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _resolve(base: Path, value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    return str(path.resolve() if path.is_absolute() else (base / path).resolve())


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _expand_placeholders(value: Any, *, repo_root: Path) -> Any:
    if isinstance(value, str):
        return value.replace("<REPO_ROOT>", str(repo_root)).replace(
            "<BASE_DIR>", str(repo_root.parent)
        )
    if isinstance(value, list):
        return [_expand_placeholders(item, repo_root=repo_root) for item in value]
    if isinstance(value, dict):
        return {
            key: _expand_placeholders(item, repo_root=repo_root)
            for key, item in value.items()
        }
    return value


def _write_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite generated config: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _phase_seeds(phase: str) -> tuple[int, ...]:
    return {
        "smoke": (42,),
        "seed42": (42,),
        "remaining_seeds": (123, 456),
        "three_seed": EVAL_SEEDS,
    }[phase]


def _task_settings(
    template: Mapping[str, Any], phase: str, task: str
) -> Dict[str, Any]:
    smoke = phase == "smoke"
    if task == "operational_nonkn":
        return {
            "gallery_sizes": [
                int(template.get("smoke_operational_gallery_size", 100))
                if smoke
                else int(template.get("operational_gallery_size", 1000))
            ],
            "gallery_trials": (
                int(template.get("smoke_gallery_trials", 2)) if smoke else 10
            ),
            "max_gw_events": (
                int(template.get("smoke_max_gw_events", 64)) if smoke else 0
            ),
            "n_neg_samples": (
                int(template.get("smoke_n_neg_samples", 20000))
                if smoke
                else int(template.get("n_neg_samples", 500000))
            ),
        }
    return {
        "gallery_sizes": (
            [int(template.get("matched_gallery_size", 16))]
            if smoke
            else [
                int(template.get("matched_gallery_size", 16)),
                int(template.get("matched_sensitivity_gallery_size", 32)),
            ]
        ),
        "gallery_trials": (
            int(template.get("smoke_gallery_trials", 2)) if smoke else 10
        ),
        "max_gw_events": (
            int(template.get("smoke_max_gw_events", 64)) if smoke else 0
        ),
    }


def prepare_suite(
    template_path: Path, *, phase: str, dry_run: bool
) -> tuple[Path, list[Dict[str, Any]]]:
    """Generate one immutable config per task, condition, and seed."""
    template = _expand_placeholders(
        _load_json(template_path), repo_root=SCRIPT_DIR.parents[2]
    )
    base = template_path.parent
    generated_root = _resolve_path(base, template["generated_config_root"]) / phase
    output_root = _resolve_path(base, template["output_root"]) / phase
    excluded = {
        "generated_config_root",
        "output_root",
        "smoke_operational_gallery_size",
        "operational_gallery_size",
        "matched_gallery_size",
        "matched_sensitivity_gallery_size",
        "smoke_gallery_trials",
        "smoke_max_gw_events",
        "smoke_n_neg_samples",
    }
    common = {key: value for key, value in template.items() if key not in excluded}
    jobs: list[Dict[str, Any]] = []
    for seed in _phase_seeds(phase):
        for task, conditions in (
            ("operational_nonkn", OPERATIONAL_CONDITIONS),
            ("kn_nuisance_matched", MATCHED_CONDITIONS),
            ("kn_same_source_random", RANDOM_MATCHED_CONDITIONS),
        ):
            for condition in conditions:
                config_path = (
                    generated_root
                    / f"seed_{seed}"
                    / task
                    / f"{condition['name']}.json"
                )
                output_dir = output_root / f"seed_{seed}" / task / condition["name"]
                payload = {
                    **common,
                    **_task_settings(template, phase, task),
                    "experiment_id": (
                        f"{template.get('experiment_id', 'fixed_checkpoint_attribution_v1')}"
                        f"_{phase}"
                    ),
                    "task": task,
                    "condition": condition,
                    "seed": seed,
                    "output_dir": str(output_dir),
                    "resume": False,
                }
                jobs.append(
                    {
                        "index": len(jobs),
                        "phase": phase,
                        "seed": seed,
                        "task": task,
                        "condition": condition["name"],
                        "config": str(config_path),
                        "output_dir": str(output_dir),
                        "payload": payload,
                    }
                )
    manifest_path = generated_root / "suite_manifest.json"
    conflicts = [Path(job["config"]) for job in jobs if Path(job["config"]).exists()]
    if manifest_path.exists():
        conflicts.append(manifest_path)
    if conflicts and not dry_run:
        raise FileExistsError(
            "Refusing to overwrite generated suite files: "
            + ", ".join(str(path) for path in conflicts[:8])
        )
    if not dry_run:
        for job in jobs:
            _write_json_new(Path(job["config"]), job["payload"])
        _write_json_new(
            manifest_path,
            {
                "phase": phase,
                "seeds": list(_phase_seeds(phase)),
                "n_jobs": len(jobs),
                "jobs": [
                    {key: value for key, value in job.items() if key != "payload"}
                    for job in jobs
                ],
            },
        )
    print(f"{'Would generate' if dry_run else 'Generated'} {len(jobs)} attribution configs")
    print(manifest_path)
    return manifest_path, jobs


def _decode_strings(values: Sequence[Any]) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_gw_tables(test_data_path: str) -> Dict[str, Any]:
    with h5py.File(test_data_path, "r") as handle:
        gw = handle["events/gw_data"]
        opt = handle["events/optical_data"]
        return {
            "scalars": np.asarray(gw["scalars"][:], dtype=np.float32),
            "source_types": _decode_strings(gw["source_type"][:]),
            "event_ids": _decode_strings(gw["ids"][:]),
            "event_time_mjd": np.asarray(gw["event_time_mjd"][:], dtype=np.float64),
            "optical_parent": np.asarray(opt["parent_gw_idx"][:], dtype=np.int64),
        }


def _read_selected_distance_skymaps(
    test_data_path: str, gw_ids: Sequence[int]
) -> np.ndarray:
    """Read only channels 5/6 for selected queries used by distance_perm."""
    selected = np.asarray(gw_ids, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("gw_ids must be a non-empty one-dimensional sequence.")
    if np.any(np.diff(selected) <= 0):
        raise ValueError("gw_ids must be sorted and unique for HDF5 fancy indexing.")
    with h5py.File(test_data_path, "r") as handle:
        skymaps = handle["events/gw_data/skymaps"]
        return np.asarray(skymaps[selected, 5:7, :], dtype=np.float32)


def _limit_gw_ids(gw_ids: Sequence[int], max_gw_events: int, seed: int) -> list[int]:
    gw_ids = sorted(int(value) for value in gw_ids)
    if max_gw_events <= 0 or len(gw_ids) <= max_gw_events:
        return gw_ids
    rng = np.random.default_rng(seed)
    return sorted(
        int(value)
        for value in rng.choice(gw_ids, size=max_gw_events, replace=False).tolist()
    )


def _load_full_model_spec(raw: Mapping[str, Any], config_dir: Path) -> Dict[str, Any]:
    specs = _build_model_specs(dict(raw), config_dir)
    selected = [
        spec
        for spec in specs
        if spec["type"] == "multimodal"
        and str(spec["name"]).strip().lower() == "full"
    ]
    if len(selected) != 1:
        raise ValueError(
            "Attribution config must contain exactly one multimodal model named 'Full'."
        )
    return selected[0]


def _read_luptitude_params(
    runtime_model_args: Mapping[str, Any],
) -> tuple[float, np.ndarray]:
    zp = float(runtime_model_args.get("mtan_lupt_psfflux_zp", DEFAULT_PSFFLUX_ZP))
    k = float(runtime_model_args.get("mtan_lupt_k", 1.0))
    m5 = runtime_model_args.get("mtan_lupt_m5_mag")
    if m5 is None:
        return zp, DEFAULT_LUPT_B_NJY.copy()
    m5 = np.asarray(m5, dtype=np.float64)
    b = k * (10.0 ** ((zp - m5) / 2.5) / 5.0)
    return zp, b


def _load_all_kn_bank(
    test_data_path: str, comparison_window: tuple[float, float]
) -> Dict[str, torch.Tensor]:
    with h5py.File(test_data_path, "r") as handle:
        n_optical = int(handle["events/optical_data/values"].shape[0])
    bank, remap = load_selected_positive_bank(
        test_data_path,
        np.arange(n_optical, dtype=np.int64),
        runtime_input_window_start=comparison_window[0],
        runtime_input_window_end=comparison_window[1],
    )
    if any(int(key) != int(value) for key, value in remap.items()):
        raise AssertionError("Full KN bank unexpectedly changed optical indices.")
    return bank


def _load_physical_metadata(
    event_ids: Sequence[str],
    catalog_paths: Mapping[str, str],
) -> Dict[int, Dict[str, float]]:
    catalogs: Dict[str, pd.DataFrame] = {}
    for source, path in catalog_paths.items():
        frame = pd.read_csv(path)
        if frame["simulation_id"].duplicated().any():
            raise ValueError(f"Duplicate simulation_id values in {path}")
        catalogs[str(source).lower()] = frame.set_index("simulation_id", drop=False)
    metadata: Dict[int, Dict[str, float]] = {}
    for gw_id, event_id in enumerate(event_ids):
        source, simulation_id = str(event_id).rsplit("_", 1)
        source = source.lower()
        if source not in catalogs or int(simulation_id) not in catalogs[source].index:
            raise KeyError(f"No physical catalog row for event_id={event_id}")
        row = catalogs[source].loc[int(simulation_id)]
        metadata[gw_id] = effective_physical_parameters(source, row)
    return metadata


def _build_operational_task(
    cfg: Mapping[str, Any],
    gw_positive_indices: Mapping[int, np.ndarray],
    selected_gw: Sequence[int],
    comparison_window: tuple[float, float],
) -> tuple[Dict[str, Any], np.ndarray, Dict[Any, Any], list[int]]:
    target_samples = int(cfg["n_neg_samples"])
    sampled_source_indices = sample_negative_optical_source_indices(
        cfg["neg_data_path"],
        cfg["neg_group"],
        n_samples=None if target_samples <= 0 else target_samples,
        seed=int(cfg["seed"]),
        negative_sample_strategy=str(cfg["negative_sample_strategy"]),
        negative_sample_block_rows=cfg.get("negative_sample_block_rows"),
        negative_sample_shuffle=bool(cfg["negative_sample_shuffle"]),
    )
    negative_stub = {
        "times": np.empty((sampled_source_indices.size, 0), dtype=np.float32)
    }
    candidate_sequences, _, _ = build_synthetic_time_sky_candidate_sequences(
        test_data_path=cfg["test_data_path"],
        unique_gw_ids=selected_gw,
        neg_optical_data=negative_stub,
        gallery_sizes=cfg["gallery_sizes"],
        n_trials=int(cfg["gallery_trials"]),
        seed=int(cfg["seed"]),
        time_window_days=float(cfg["gallery_candidate_time_window_days"]),
        credible_level_max=float(cfg["gallery_candidate_credible_level_max"]),
    )
    selected_map = {int(gw_id): gw_positive_indices[int(gw_id)] for gw_id in selected_gw}
    galleries, unique_gw = build_prefixed_gallery_specs(
        gw_positive_indices=selected_map,
        candidate_sequences=candidate_sequences,
        gallery_sizes=cfg["gallery_sizes"],
        n_trials=int(cfg["gallery_trials"]),
        seed=int(cfg["seed"]),
        include_undersized=False,
    )
    if not galleries:
        raise ValueError("Operational gallery construction produced no instances.")
    selected_positive_indices = np.unique(
        np.asarray([spec["positive_index"] for spec in galleries.values()], np.int64)
    )
    positive, remap = load_selected_positive_bank(
        cfg["test_data_path"],
        selected_positive_indices,
        runtime_input_window_start=comparison_window[0],
        runtime_input_window_end=comparison_window[1],
    )
    galleries = remap_gallery_positive_indices(galleries, remap)
    galleries, selected_negative_sources = _compact_operational_negative_galleries(
        galleries, sampled_source_indices
    )
    return positive, selected_negative_sources, galleries, unique_gw


def _compact_operational_negative_galleries(
    galleries: Mapping[Any, Mapping[str, Any]],
    sampled_source_indices: Sequence[int],
) -> tuple[Dict[Any, Dict[str, Any]], np.ndarray]:
    """Remap sampled-pool positions to a sorted compact HDF5 source bank."""
    sampled = np.asarray(sampled_source_indices, dtype=np.int64).reshape(-1)
    negative_parts = [
        np.asarray(spec["negative_indices"], dtype=np.int64).reshape(-1)
        for spec in galleries.values()
    ]
    used_pool_positions = np.unique(np.concatenate(negative_parts))
    if used_pool_positions.size == 0:
        raise ValueError("Operational galleries contain no negative candidates.")
    if used_pool_positions[0] < 0 or used_pool_positions[-1] >= sampled.size:
        raise IndexError("Gallery negative index is outside the sampled pool.")
    selected_sources = np.unique(sampled[used_pool_positions])

    compacted: Dict[Any, Dict[str, Any]] = {}
    for key, spec in galleries.items():
        pool_positions = np.asarray(spec["negative_indices"], dtype=np.int64)
        source_indices = sampled[pool_positions]
        compact_indices = np.searchsorted(selected_sources, source_indices)
        if not np.array_equal(selected_sources[compact_indices], source_indices):
            raise AssertionError("Failed to compact operational negative indices.")
        updated = dict(spec)
        updated["source_negative_indices"] = source_indices.copy()
        updated["negative_indices"] = compact_indices.astype(np.int64, copy=False)
        compacted[key] = updated
    return compacted, selected_sources


def _load_negative_source_rows(
    cfg: Mapping[str, Any],
    source_indices: np.ndarray,
    comparison_window: tuple[float, float],
) -> Dict[str, torch.Tensor]:
    """Load one sorted negative-bank chunk and apply the runtime time window."""
    selected = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    if selected.size == 0 or np.any(np.diff(selected) <= 0):
        raise ValueError("Negative source indices must be non-empty, sorted, unique.")
    with h5py.File(cfg["neg_data_path"], "r") as handle:
        group = handle[cfg["neg_group"]]
        bank = {
            field: torch.from_numpy(np.asarray(group[field][selected], np.float32))
            for field in ("times", "values", "masks", "errors", "coordinates")
        }
    times, values, masks, errors, _ = apply_runtime_input_window_torch(
        bank["times"],
        bank["values"],
        bank["masks"],
        bank["errors"],
        window_start=comparison_window[0],
        window_end=comparison_window[1],
    )
    bank.update(
        {
            "times": times,
            "values": values,
            "masks": masks,
            "errors": errors,
            "source_indices": torch.from_numpy(selected.copy()),
        }
    )
    return bank


def _encode_streamed_negative_bank(
    *,
    model: Any,
    cfg: Mapping[str, Any],
    source_indices: np.ndarray,
    comparison_window: tuple[float, float],
    transform: str,
    transform_seed: int,
    psfflux_zp: float,
    lupt_b_njy: np.ndarray,
    device: torch.device,
    runtime_model_args: Mapping[str, Any],
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[Dict[str, torch.Tensor], list[Dict[str, Any]]]:
    """Load, transform, and encode a large negative bank in bounded chunks."""
    selected = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    chunk_size = int(cfg["negative_encode_chunk_size"])
    embedding_parts: Dict[str, list[torch.Tensor]] = defaultdict(list)
    audit_rows: list[Dict[str, Any]] = []
    starts = tqdm(
        range(0, selected.size, chunk_size),
        desc="  Streaming negative candidate bank",
    )
    for start in starts:
        end = min(start + chunk_size, selected.size)
        source_chunk = selected[start:end]
        raw_bank = _load_negative_source_rows(cfg, source_chunk, comparison_window)
        transformed, audit = transform_optical_bank(
            raw_bank,
            transform,
            seed=transform_seed,
            item_indices=source_chunk,
            psfflux_zp=psfflux_zp,
            lupt_b_njy=lupt_b_njy,
        )
        encoded = extract_negative_gallery_embeddings(
            model,
            transformed,
            device,
            n_ref=int(runtime_model_args.get("n_ref", 64)),
            ref_start=float(runtime_model_args["ref_start"]),
            ref_end=float(runtime_model_args["ref_end"]),
            retain_raw=False,
            show_progress=False,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        for key, value in encoded.items():
            embedding_parts[key].append(value)
        audit_rows.extend(audit)
    return (
        {key: torch.cat(parts, dim=0) for key, parts in embedding_parts.items()},
        audit_rows,
    )


def _build_matched_task(
    cfg: Mapping[str, Any],
    gw_positive_indices: Mapping[int, np.ndarray],
    selected_gw: Sequence[int],
    comparison_window: tuple[float, float],
    gw_tables: Mapping[str, Any],
) -> tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[Any, Any],
    list[int],
    np.ndarray,
]:
    bank = _load_all_kn_bank(cfg["test_data_path"], comparison_window)
    nuisance = optical_nuisance_features(
        bank,
        gw_tables["optical_parent"],
        gw_tables["scalars"],
        gw_tables["event_time_mjd"],
    )
    selected_map = {int(gw_id): gw_positive_indices[int(gw_id)] for gw_id in selected_gw}
    builder = (
        build_kn_random_same_source_galleries
        if cfg["task"] == "kn_same_source_random"
        else build_kn_nuisance_matched_galleries
    )
    galleries, unique_gw = builder(
        gw_positive_indices=selected_map,
        optical_parent_gw_idx=gw_tables["optical_parent"],
        gw_source_types=gw_tables["source_types"],
        nuisance_features=nuisance,
        gallery_sizes=cfg["gallery_sizes"],
        n_trials=int(cfg["gallery_trials"]),
        seed=int(cfg["seed"]),
    )
    return bank, bank, galleries, unique_gw, nuisance


def _outcome_rows(
    *,
    task: str,
    condition: str,
    seed: int,
    outcomes: Mapping[Any, Mapping[str, Any]],
    details: Mapping[Any, Mapping[str, Any]],
    source_types: Sequence[str],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for (gallery_size, trial, gw_id), outcome in sorted(outcomes.items()):
        rank = int(outcome["rank"])
        actual = int(outcome["actual_gallery_size"])
        detail = details[(gallery_size, trial, gw_id)]
        rows.append(
            {
                "task": task,
                "condition": condition,
                "seed": seed,
                "trial": trial,
                "gw_id": gw_id,
                "source_type": str(source_types[gw_id]).lower(),
                "gallery_size": gallery_size,
                "actual_gallery_size": actual,
                "rank_zero_based": rank,
                "recall_at_1": float(rank < 1),
                "recall_at_5": float(rank < 5),
                "recall_at_10": float(rank < 10),
                "mrr": 1.0 / float(rank + 1),
                "coverage": float(actual >= int(gallery_size)),
                "positive_score": float(detail["positive_score"]),
                "best_negative_score": float(detail["best_negative_score"]),
                "score_margin": float(detail["score_margin"]),
            }
        )
    return rows


def _score_summary(rows: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    buckets: Dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        gallery_size = int(row["gallery_size"])
        buckets[("all", gallery_size)].append(row)
        buckets[(str(row["source_type"]), gallery_size)].append(row)
    output: list[Dict[str, Any]] = []
    for (source, gallery_size), values in sorted(buckets.items()):
        output.append(
            {
                "task": values[0]["task"],
                "condition": values[0]["condition"],
                "seed": values[0]["seed"],
                "source_type": source,
                "gallery_size": gallery_size,
                "n_queries": len(values),
                **{
                    field: float(np.mean([float(row[field]) for row in values]))
                    for field in (
                        "recall_at_1",
                        "recall_at_5",
                        "recall_at_10",
                        "mrr",
                        "coverage",
                    )
                },
                "positive_score_mean": float(
                    np.mean([float(row["positive_score"]) for row in values])
                ),
                "best_negative_score_mean": float(
                    np.mean([float(row["best_negative_score"]) for row in values])
                ),
                "score_margin_mean": float(
                    np.mean([float(row["score_margin"]) for row in values])
                ),
            }
        )
    return output


def _candidate_rows(
    *,
    cfg: Mapping[str, Any],
    condition_name: str,
    galleries: Mapping[Any, Mapping[str, Any]],
    score_details: Mapping[Any, Mapping[str, Any]],
    gw_tables: Mapping[str, Any],
    physical: Mapping[int, Mapping[str, float]],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    parent = gw_tables["optical_parent"]
    source = gw_tables["source_types"]
    for key, spec in sorted(galleries.items()):
        gallery_size, trial, gw_id = key
        scores = np.asarray(score_details[key]["candidate_scores"], dtype=np.float64)
        costs = np.concatenate(
            [
                np.asarray([0.0]),
                np.asarray(spec.get("negative_match_costs", []), dtype=np.float64),
            ]
        )
        candidates = list(iter_candidate_indices(spec))
        if len(candidates) != scores.size or costs.size != scores.size:
            raise AssertionError("Candidate audit arrays are misaligned.")
        query_physical = physical[int(gw_id)]
        for position, ((optical_index, is_positive), score, cost) in enumerate(
            zip(candidates, scores.tolist(), costs.tolist())
        ):
            candidate_parent = int(parent[optical_index])
            mismatch = physical_mismatch(
                query_physical, physical[candidate_parent]
            )
            rows.append(
                {
                    "task": cfg["task"],
                    "condition": condition_name,
                    "seed": cfg["seed"],
                    "trial": trial,
                    "gw_id": gw_id,
                    "source_type": str(source[gw_id]).lower(),
                    "gallery_size": gallery_size,
                    "candidate_position": position,
                    "candidate_optical_index": optical_index,
                    "candidate_parent_gw_id": candidate_parent,
                    "is_positive": int(is_positive),
                    "score": float(score),
                    "match_cost": float(cost),
                    **mismatch,
                }
            )
    return rows


def _write_gzip_rows(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    writer = AtomicGzipCsvWriter(path, fieldnames)
    try:
        writer.writerows(rows)
        writer.commit()
    finally:
        writer.close()


def _normalize_config(raw: Mapping[str, Any], path: Path) -> Dict[str, Any]:
    base = path.parent
    task = str(raw.get("task", "")).strip().lower()
    if task not in {
        "operational_nonkn",
        "kn_nuisance_matched",
        "kn_same_source_random",
    }:
        raise ValueError(
            "task must be one of {'operational_nonkn', 'kn_nuisance_matched', "
            "'kn_same_source_random'}"
        )
    condition = normalize_condition(raw.get("condition", {}))
    gallery_sizes_raw = raw.get("gallery_sizes", [1000])
    if isinstance(gallery_sizes_raw, str):
        gallery_sizes = [int(value) for value in gallery_sizes_raw.split(",")]
    else:
        gallery_sizes = [int(value) for value in gallery_sizes_raw]
    cfg = {
        **dict(raw),
        "task": task,
        "condition_normalized": condition,
        "test_data_path": _resolve(base, raw.get("test_data_path")),
        "neg_data_path": _resolve(base, raw.get("neg_data_path")),
        "output_dir": _resolve(base, raw.get("output_dir")),
        "gallery_sizes": gallery_sizes,
        "gallery_trials": int(raw.get("gallery_trials", 10)),
        "seed": int(raw.get("seed", 42)),
        "max_gw_events": int(raw.get("max_gw_events", 0) or 0),
        "neg_group": str(raw.get("neg_group", "ELASTICC/optical_data")),
        "n_neg_samples": int(raw.get("n_neg_samples", 500000)),
        "negative_encode_chunk_size": int(
            raw.get("negative_encode_chunk_size", 2048)
        ),
        "nonkn_cls_base_field": str(
            raw.get("nonkn_cls_base_field", "zero_time_mjd_cls_base")
        ),
        "negative_sample_strategy": str(
            raw.get("negative_sample_strategy", "block_random")
        ),
        "negative_sample_block_rows": raw.get("negative_sample_block_rows"),
        "negative_sample_shuffle": bool(raw.get("negative_sample_shuffle", True)),
        "gallery_candidate_time_window_days": float(
            raw.get("gallery_candidate_time_window_days", 30.0)
        ),
        "gallery_candidate_credible_level_max": float(
            raw.get("gallery_candidate_credible_level_max", 0.9)
        ),
        "comparison_window": tuple(
            float(value) for value in raw.get("comparison_window", [-0.1, 0.2])
        ),
        "physical_catalogs": {
            str(key).lower(): _resolve(base, value)
            for key, value in raw.get("physical_catalogs", {}).items()
        },
    }
    if not cfg["test_data_path"] or not cfg["output_dir"]:
        raise ValueError("test_data_path and output_dir are required.")
    if task == "operational_nonkn" and not cfg["neg_data_path"]:
        raise ValueError("operational_nonkn requires neg_data_path.")
    if cfg["negative_encode_chunk_size"] <= 0:
        raise ValueError("negative_encode_chunk_size must be positive.")
    if task in {"kn_nuisance_matched", "kn_same_source_random"}:
        if condition.time_delta_mode != "positive_shared":
            raise ValueError("KN-matched conditions must use positive_shared time delta.")
        if condition.coordinate_mode != "positive_shared":
            raise ValueError("KN-matched conditions must use positive_shared coordinates.")
        if set(cfg["physical_catalogs"]) != {"bns", "nsbh"}:
            raise ValueError("KN-matched task requires BNS and NSBH physical catalogs.")
    return cfg


def run_condition(config: str | Path) -> None:
    """Run one fixed-checkpoint attribution condition from an immutable config."""
    config_path = Path(config).expanduser().resolve()
    raw = _load_json(config_path)
    cfg = _normalize_config(raw, config_path)
    condition = cfg["condition_normalized"]
    seed = int(cfg["seed"])
    _seed_all(seed)

    output_dir = Path(cfg["output_dir"])
    experiment_config = dict(raw)
    for volatile in ("output_dir", "resume"):
        experiment_config.pop(volatile, None)
    manifest = prepare_output_directory(
        output_dir,
        manifest={
            "experiment_id": str(raw.get("experiment_id", "fixed_checkpoint_attribution_v1")),
            "experiment_digest": stable_digest(experiment_config),
            "code_digest": source_tree_digest(MODEL_DIR),
            "seed": seed,
            "task": cfg["task"],
            "condition": condition.name,
            "input_config": str(config_path),
            "test_data_path": cfg["test_data_path"],
            "neg_data_path": cfg["neg_data_path"],
        },
        resume=bool(raw.get("resume", False)),
    )

    gw_tables = _read_gw_tables(cfg["test_data_path"])
    gw_positive_indices, all_positive_gw = load_test_positive_index_map(
        cfg["test_data_path"]
    )
    selected_gw = _limit_gw_ids(all_positive_gw, cfg["max_gw_events"], seed)
    comparison_window = tuple(cfg["comparison_window"])

    if cfg["task"] == "operational_nonkn":
        positive_bank, negative_source_indices, galleries, unique_gw = _build_operational_task(
            cfg,
            gw_positive_indices,
            selected_gw,
            comparison_window,
        )
        negative_bank = None
        nuisance = None
    else:
        positive_bank, negative_bank, galleries, unique_gw, nuisance = _build_matched_task(
            cfg,
            gw_positive_indices,
            selected_gw,
            comparison_window,
            gw_tables,
        )
        negative_source_indices = None

    gallery_digest = gallery_identity_digest(galleries)
    model_spec = _load_full_model_spec(raw, config_path.parent)
    device_name = str(raw.get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    amp_dtype, amp_enabled = _resolve_eval_amp(str(raw.get("amp_dtype", "bf16")), device)
    model, runtime_model_args, saved_args = load_multimodal_bundle(
        model_spec["resolved_checkpoint"],
        model_spec["resolved_config"],
        device,
        test_data_path=cfg["test_data_path"],
        neg_data_path=cfg["neg_data_path"],
        comparison_window=comparison_window,
        nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
    )
    if _resolve_multimodal_scoring(model_spec, saved_args) != "logits":
        raise ValueError("The Full attribution checkpoint must support fusion logits.")

    zp, lupt_b = _read_luptitude_params(runtime_model_args)
    positive_transformed, positive_audit = transform_optical_bank(
        positive_bank,
        condition.optical_transform,
        seed=seed,
        psfflux_zp=zp,
        lupt_b_njy=lupt_b,
    )
    positive_embeddings = extract_optical_candidate_embeddings(
        model,
        positive_transformed,
        device,
        n_ref=int(runtime_model_args.get("n_ref", 64)),
        ref_start=float(runtime_model_args["ref_start"]),
        ref_end=float(runtime_model_args["ref_end"]),
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
        desc="  Encoding positive/KN candidate bank",
    )
    if cfg["task"] in {"kn_nuisance_matched", "kn_same_source_random"}:
        if negative_bank is not positive_bank:
            raise AssertionError("KN-matched candidates must share one compact bank.")
        negative_embeddings = positive_embeddings
        negative_audit: list[Dict[str, Any]] = []
    else:
        if negative_source_indices is None:
            raise AssertionError("Operational task is missing negative source rows.")
        negative_embeddings, negative_audit = _encode_streamed_negative_bank(
            model=model,
            cfg=cfg,
            source_indices=negative_source_indices,
            comparison_window=comparison_window,
            transform=condition.optical_transform,
            transform_seed=seed + 1000003,
            psfflux_zp=zp,
            lupt_b_njy=lupt_b,
            device=device,
            runtime_model_args=runtime_model_args,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )

    query_positions = np.asarray(unique_gw, dtype=np.int64)
    distance_skymaps = None
    if condition.gw_transform == "distance_perm":
        distance_skymaps = _read_selected_distance_skymaps(
            cfg["test_data_path"], query_positions
        )
    query_transform = GWInputTransform(
        condition.gw_transform,
        query_positions,
        gw_tables["scalars"][query_positions],
        distance_skymaps,
        [gw_tables["source_types"][idx] for idx in query_positions],
    )
    event_time_table = load_gw_event_time_mjd_table(
        cfg["test_data_path"], device, required=False
    )
    ranks, score_details = score_all_galleries_multimodal(
        model,
        positive_embeddings,
        negative_embeddings,
        galleries,
        unique_gw,
        test_data_path=cfg["test_data_path"],
        device=device,
        model_args=runtime_model_args,
        gw_event_time_mjd_table=event_time_table,
        candidate_time_delta_mode=condition.time_delta_mode,
        candidate_coordinate_mode=condition.coordinate_mode,
        query_input_transform=query_transform,
        return_score_details=True,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    outcomes = enrich_gallery_outcomes(ranks, galleries)
    gw_source_map = {
        int(gw_id): str(gw_tables["source_types"][gw_id]).lower()
        for gw_id in unique_gw
    }
    retrieval, retrieval_by_source, coverage = aggregate_gallery_outcomes(
        outcomes=outcomes,
        gallery_sizes=cfg["gallery_sizes"],
        n_trials=cfg["gallery_trials"],
        unique_gw=unique_gw,
        gw_source_map=gw_source_map,
    )
    outcome_rows = _outcome_rows(
        task=cfg["task"],
        condition=condition.name,
        seed=seed,
        outcomes=outcomes,
        details=score_details,
        source_types=gw_tables["source_types"],
    )
    summary_rows = _score_summary(outcome_rows)
    _write_gzip_rows(
        output_dir / "retrieval_outcomes.csv.gz", OUTCOME_FIELDS, outcome_rows
    )
    write_csv_atomic(
        output_dir / "score_summaries.csv", summary_rows, SCORE_SUMMARY_FIELDS
    )

    audit_rows: list[Dict[str, Any]] = []
    for row in query_transform.audit_rows:
        audit_rows.append({"audit_type": "gw", "bank": "query", **row})
    for bank_name, rows in (("positive", positive_audit), ("negative", negative_audit)):
        for row in rows:
            audit_rows.append(
                {"audit_type": "optical", "bank": bank_name, **row}
            )
    normalized_audit = (
        {field: row.get(field, "") for field in TRANSFORM_AUDIT_FIELDS}
        for row in audit_rows
    )
    _write_gzip_rows(
        output_dir / "transform_audit.csv.gz",
        TRANSFORM_AUDIT_FIELDS,
        normalized_audit,
    )

    artifacts = [
        "retrieval_outcomes.csv.gz",
        "score_summaries.csv",
        "transform_audit.csv.gz",
        "attribution_result.json",
        "gallery_identity.json",
    ]
    if cfg["task"] in {"kn_nuisance_matched", "kn_same_source_random"}:
        physical = _load_physical_metadata(
            gw_tables["event_ids"], cfg["physical_catalogs"]
        )
        candidate_rows = _candidate_rows(
            cfg=cfg,
            condition_name=condition.name,
            galleries=galleries,
            score_details=score_details,
            gw_tables=gw_tables,
            physical=physical,
        )
        _write_gzip_rows(
            output_dir / "candidate_scores.csv.gz",
            CANDIDATE_FIELDS,
            candidate_rows,
        )
        match_rows = summarize_match_balance(galleries, nuisance)
        write_csv_atomic(
            output_dir / "match_balance.csv",
            match_rows,
            MATCH_BALANCE_FIELDS,
        )
        artifacts.extend(["candidate_scores.csv.gz", "match_balance.csv"])

    write_json_atomic(
        output_dir / "gallery_identity.json",
        {
            "sha256": gallery_digest,
            "n_galleries": len(galleries),
            "n_unique_gw": len(unique_gw),
            "gallery_sizes": cfg["gallery_sizes"],
            "gallery_trials": cfg["gallery_trials"],
            "n_negative_source_rows": (
                0
                if negative_source_indices is None
                else int(negative_source_indices.size)
            ),
            "negative_source_rows_sha256": (
                None
                if negative_source_indices is None
                else stable_digest(negative_source_indices.tolist())
            ),
        },
    )
    write_json_atomic(
        output_dir / "attribution_result.json",
        {
            "task": cfg["task"],
            "condition": condition.__dict__,
            "seed": seed,
            "model": "Full",
            "resolved_checkpoint": model_spec["resolved_checkpoint"],
            "resolved_config": model_spec["resolved_config"],
            "comparison_window": list(comparison_window),
            "gallery_identity_sha256": gallery_digest,
            "n_unique_gw": len(unique_gw),
            "n_outcomes": len(outcome_rows),
            "retrieval": retrieval,
            "retrieval_by_source": retrieval_by_source,
            "coverage": coverage,
        },
    )
    mark_run_success(output_dir, manifest, artifacts)
    print(f"Completed {cfg['task']} / {condition.name}: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a fixed-checkpoint attribution suite or run one condition."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", help="Run one generated condition config.")
    mode.add_argument("--template", help="Generate configs from a suite template.")
    parser.add_argument(
        "--phase",
        choices=("smoke", "seed42", "remaining_seeds", "three_seed"),
        default="smoke",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --template, validate without writing generated configs.",
    )
    args = parser.parse_args()
    if args.template:
        prepare_suite(
            Path(args.template).expanduser().resolve(),
            phase=args.phase,
            dry_run=args.dry_run,
        )
        return
    if args.dry_run:
        parser.error("--dry-run is only valid with --template")
    run_condition(args.config)


if __name__ == "__main__":
    main()
