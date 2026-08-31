#!/usr/bin/env python3
"""Measure fixed-checkpoint sensitivity to the physical GW--KN pairing.

The experiment uses crossed event pairs.  For two same-source events A and B,
the score interaction is

    0.5 * (s(G_A, O_A) + s(G_B, O_B) - s(G_A, O_B) - s(G_B, O_A)).

Candidate sky position and GW-to-detection delay are shared across all four
cells of each crossed block.  The statistic therefore cancels additive GW-only
and optical-only preferences while retaining joint GW-conditioned information,
including physically meaningful light-curve brightness.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval import eval_retrieval_comparison as base
from scripts.eval.eval_run_io import (
    AtomicGzipCsvWriter,
    mark_run_success,
    prepare_output_directory,
    stable_digest,
    write_csv_atomic,
    write_json_atomic,
)

PHYSICAL_FEATURE_NAMES = (
    "chirp_mass_detector",
    "mass_ratio",
    "chi_eff",
    "abs_costheta",
    "log10_distance_gpc",
)
SINGLE_PARAMETER_PROFILE_NAME = "source_physical_v1"
SINGLE_PARAMETER_FEATURES_BY_SOURCE = {
    "bns": (
        "chirp_mass_detector",
        "mass_ratio",
        "chi_eff",
        "abs_costheta",
        "log10_distance_gpc",
    ),
    "nsbh": (
        "chirp_mass_detector",
        "mass_ratio",
        "primary_spin_z",
        "abs_costheta",
        "log10_distance_gpc",
    ),
}
SINGLE_PARAMETER_FEATURE_NAMES = (
    "chirp_mass_detector",
    "mass_ratio",
    "chi_eff",
    "primary_spin_z",
    "abs_costheta",
    "log10_distance_gpc",
)
SINGLE_PARAMETER_COMMON_TARGETS = (
    "chirp_mass_detector",
    "mass_ratio",
    "abs_costheta",
    "log10_distance_gpc",
)
CONDITION_FIELDS = (
    "condition_id",
    "target_parameter",
    "caliper_iqr",
    "is_primary_caliper",
)
EVENT_NUISANCE_FEATURE_NAMES = (
    "log10_fractional_distance_uncertainty",
    "median_log1p_nobs",
    "median_band_fraction",
    "median_log1p_time_span_days",
)
CURVE_SAMPLING_FEATURE_NAMES = (
    "log1p_nobs",
    "band_fraction",
    "log1p_time_span_days",
)
PAIR_FIELDS = (
    "pair_id",
    "source_type",
    "gw_a",
    "gw_b",
    "physical_distance",
    "physical_distance_threshold",
    "nuisance_l1_distance",
    "nuisance_max_abs_difference",
    *tuple(f"delta_{name}" for name in PHYSICAL_FEATURE_NAMES),
    *tuple(f"delta_{name}" for name in EVENT_NUISANCE_FEATURE_NAMES),
)
CURVE_PAIR_FIELDS = (
    "pair_id",
    "curve_pair_id",
    "source_type",
    "gw_a",
    "gw_b",
    "optical_index_a",
    "optical_index_b",
    "curve_sampling_l1_distance",
    *tuple(f"delta_{name}" for name in CURVE_SAMPLING_FEATURE_NAMES),
)
SCORE_FIELDS = (
    "model",
    "model_type",
    "pair_id",
    "curve_pair_id",
    "source_type",
    "anchor",
    "query_side",
    "candidate_side",
    "query_gw_id",
    "candidate_parent_gw_id",
    "optical_index",
    "is_aligned",
    "score",
    "anchor_ra",
    "anchor_dec",
    "anchor_dt_days",
)
ANCHOR_INTERACTION_FIELDS = (
    "model",
    "model_type",
    "pair_id",
    "curve_pair_id",
    "source_type",
    "anchor",
    "physical_distance",
    "score_aa",
    "score_ab",
    "score_ba",
    "score_bb",
    "margin_a",
    "margin_b",
    "interaction",
    "directional_win_rate",
)
PAIR_METRIC_FIELDS = (
    "model",
    "model_type",
    "pair_id",
    "source_type",
    "gw_a",
    "gw_b",
    "physical_distance",
    "n_curve_pairs",
    "interaction",
    "directional_win_rate",
    "margin_a",
    "margin_b",
)
SUMMARY_FIELDS = (
    "endpoint",
    "model",
    "baseline_model",
    "source",
    "n_pairs",
    "mean_interaction",
    "ci95_low",
    "ci95_high",
    "directional_win_rate",
    "permutation_p_one_sided",
    "bootstrap_samples",
    "permutation_samples",
)
QUARTILE_FIELDS = (
    "model",
    "source",
    "physical_distance_quartile",
    "n_pairs",
    "physical_distance_mean",
    "interaction_mean",
    "directional_win_rate",
)
SINGLE_PAIR_FIELDS = (
    *CONDITION_FIELDS,
    "pair_id",
    "source_type",
    "gw_a",
    "gw_b",
    "target_delta_iqr",
    "other_parameter_l1_distance",
    "other_parameter_max_abs_difference",
    "physical_distance",
    "nuisance_l1_distance",
    "nuisance_max_abs_difference",
    *tuple(f"delta_{name}" for name in SINGLE_PARAMETER_FEATURE_NAMES),
    *tuple(f"delta_{name}" for name in EVENT_NUISANCE_FEATURE_NAMES),
)
SINGLE_CURVE_PAIR_FIELDS = (*CONDITION_FIELDS, *CURVE_PAIR_FIELDS)
SINGLE_SCORE_FIELDS = (
    "model",
    "model_type",
    *CONDITION_FIELDS,
    *SCORE_FIELDS[2:],
)
SINGLE_ANCHOR_INTERACTION_FIELDS = (
    "model",
    "model_type",
    *CONDITION_FIELDS,
    *ANCHOR_INTERACTION_FIELDS[2:],
)
SINGLE_PAIR_METRIC_FIELDS = (
    "model",
    "model_type",
    *CONDITION_FIELDS,
    *PAIR_METRIC_FIELDS[2:],
    "target_delta_iqr",
    "other_parameter_l1_distance",
    "other_parameter_max_abs_difference",
    "nuisance_l1_distance",
    "nuisance_max_abs_difference",
)
SINGLE_SUMMARY_FIELDS = (
    "endpoint",
    "metric",
    "family",
    "target_parameter",
    "caliper_iqr",
    "is_primary_caliper",
    "model",
    "baseline_model",
    "source",
    "n_pairs",
    "estimate",
    "null_value",
    "ci95_low",
    "ci95_high",
    "permutation_p_one_sided",
    "holm_adjusted_p",
    "holm_reject_0_05",
    "median",
    "trimmed_mean_5pct",
    "positive_fraction",
    "bootstrap_samples",
    "permutation_samples",
)
MATCHING_BALANCE_FIELDS = (
    "condition_id",
    "target_parameter",
    "source_type",
    "caliper_iqr",
    "is_primary_caliper",
    "n_pairs",
    "feature",
    "role",
    "minimum",
    "median",
    "p90",
    "maximum",
)
DOSE_RESPONSE_FIELDS = (
    "target_parameter",
    "source",
    "model",
    "dose_bin",
    "n_pairs",
    "target_delta_min",
    "target_delta_median",
    "target_delta_max",
    "interaction_mean",
    "directional_win_rate",
)
TREND_FIELDS = (
    "target_parameter",
    "source",
    "model",
    "n_pairs",
    "spearman_target_delta_interaction",
)


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _decode_strings(values: Sequence[Any]) -> list[str]:
    return [
        (
            value.decode("utf-8", errors="ignore")
            if isinstance(value, (bytes, np.bytes_))
            else str(value)
        )
        for value in values
    ]


def _resolve_path(config_dir: Path, value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    path = (config_dir / path).resolve() if not path.is_absolute() else path.resolve()
    jobfs_dir = os.environ.get("JOBFS_DIR")
    if jobfs_dir:
        staged = Path(jobfs_dir) / path.name
        if staged.is_file():
            return str(staged)
    return str(path)


def normalise_config(raw: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    """Normalize and validate the public experiment configuration."""
    cfg = dict(raw)
    config_dir = config_path.parent
    cfg["requested_test_data_path"] = str(cfg.get("test_data_path", ""))
    cfg["test_data_path"] = _resolve_path(config_dir, cfg.get("test_data_path"))
    cfg["output_dir"] = _resolve_path(config_dir, cfg.get("output_dir"))
    cfg["seed"] = int(cfg.get("seed", 42))
    cfg["comparison_window"] = [
        float(value) for value in cfg.get("comparison_window", [-0.1, 0.2])
    ]
    cfg["physical_distance_quantile"] = float(
        cfg.get("physical_distance_quantile", 0.5)
    )
    cfg["event_nuisance_l1_max"] = float(cfg.get("event_nuisance_l1_max", 3.0))
    cfg["event_nuisance_feature_max_abs"] = float(
        cfg.get("event_nuisance_feature_max_abs", 1.0)
    )
    cfg["max_curve_pairs_per_event_pair"] = int(
        cfg.get("max_curve_pairs_per_event_pair", 5)
    )
    cfg["min_pairs_per_source"] = int(cfg.get("min_pairs_per_source", 250))
    cfg["bootstrap_samples"] = int(cfg.get("bootstrap_samples", 10000))
    cfg["permutation_samples"] = int(cfg.get("permutation_samples", 10000))
    cfg["optical_null_tolerance"] = float(cfg.get("optical_null_tolerance", 1e-6))
    cfg["device"] = str(cfg.get("device", "cuda"))
    cfg["amp_dtype"] = str(cfg.get("amp_dtype", "bf16"))
    cfg["nonkn_cls_base_field"] = str(
        cfg.get("nonkn_cls_base_field", "zero_time_mjd_cls_base")
    )
    cfg["new_model_name"] = str(cfg.get("new_model_name", "Mixed Gallery v1"))
    cfg["baseline_model_name"] = str(cfg.get("baseline_model_name", "Default MAGIKS"))
    cfg["optical_null_model_name"] = str(
        cfg.get("optical_null_model_name", "Optical-only")
    )
    cfg["resume"] = bool(cfg.get("resume", False))
    cfg["pairing_mode"] = str(cfg.get("pairing_mode", "joint")).strip().lower()
    cfg["single_parameter_profile"] = str(
        cfg.get("single_parameter_profile", SINGLE_PARAMETER_PROFILE_NAME)
    ).strip()
    cfg["target_min_abs_iqr"] = float(cfg.get("target_min_abs_iqr", 1.0))
    cfg["other_parameter_calipers_iqr"] = sorted(
        {
            float(value)
            for value in cfg.get("other_parameter_calipers_iqr", [0.25, 0.5, 0.75])
        }
    )
    cfg["primary_other_parameter_caliper_iqr"] = float(
        cfg.get("primary_other_parameter_caliper_iqr", 0.5)
    )
    raw_min_pairs = cfg.get("min_pairs_by_caliper", {"0.25": 30, "0.5": 75, "0.75": 75})
    cfg["min_pairs_by_caliper"] = {
        float(key): int(value) for key, value in dict(raw_min_pairs).items()
    }
    cfg["dose_response_bins"] = int(cfg.get("dose_response_bins", 3))
    cfg["multiple_testing_method"] = (
        str(cfg.get("multiple_testing_method", "holm")).strip().lower()
    )

    if not cfg["test_data_path"] or not cfg["output_dir"]:
        raise ValueError("test_data_path and output_dir are required")
    if len(cfg["comparison_window"]) != 2:
        raise ValueError("comparison_window must contain exactly two values")
    if not cfg["comparison_window"][0] < cfg["comparison_window"][1]:
        raise ValueError("comparison_window must be strictly increasing")
    if not 0.0 < cfg["physical_distance_quantile"] < 1.0:
        raise ValueError("physical_distance_quantile must be in (0, 1)")
    for key in (
        "event_nuisance_l1_max",
        "event_nuisance_feature_max_abs",
        "optical_null_tolerance",
    ):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in (
        "max_curve_pairs_per_event_pair",
        "min_pairs_per_source",
        "bootstrap_samples",
        "permutation_samples",
    ):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be >= 1")
    if cfg["resume"]:
        raise ValueError(
            "gw_kn_pairing_sensitivity produces immutable runs; choose a new "
            "output_dir instead of resume=true"
        )
    if cfg["pairing_mode"] not in {"joint", "single_parameter"}:
        raise ValueError("pairing_mode must be 'joint' or 'single_parameter'")
    if cfg["pairing_mode"] == "single_parameter":
        if cfg["single_parameter_profile"] != SINGLE_PARAMETER_PROFILE_NAME:
            raise ValueError("single_parameter_profile must be source_physical_v1")
        if not np.isfinite(cfg["target_min_abs_iqr"]) or cfg["target_min_abs_iqr"] <= 0:
            raise ValueError("target_min_abs_iqr must be finite and positive")
        calipers = cfg["other_parameter_calipers_iqr"]
        primary = cfg["primary_other_parameter_caliper_iqr"]
        if not calipers or any(
            not np.isfinite(value) or value <= 0 for value in calipers
        ):
            raise ValueError(
                "other_parameter_calipers_iqr must contain positive finite values"
            )
        if not any(
            np.isclose(primary, value, rtol=0.0, atol=1e-12) for value in calipers
        ):
            raise ValueError(
                "primary_other_parameter_caliper_iqr must appear in "
                "other_parameter_calipers_iqr"
            )
        for caliper in calipers:
            if caliper not in cfg["min_pairs_by_caliper"]:
                raise ValueError(f"Missing min_pairs_by_caliper for {caliper:g}")
            if cfg["min_pairs_by_caliper"][caliper] < 1:
                raise ValueError("Every min_pairs_by_caliper value must be >= 1")
        if cfg["dose_response_bins"] < 2:
            raise ValueError("dose_response_bins must be >= 2")
        if cfg["multiple_testing_method"] != "holm":
            raise ValueError("Only Holm multiple-testing correction is supported")

    models = list(cfg.get("models", []))
    names = [str(model.get("name", "")) for model in models]
    required_names = {
        cfg["new_model_name"],
        cfg["baseline_model_name"],
        cfg["optical_null_model_name"],
    }
    if set(names) != required_names or len(names) != len(required_names):
        raise ValueError(
            "models must contain exactly the configured new, baseline, and "
            "optical-null model names"
        )
    type_by_name = {str(model["name"]): str(model.get("type", "")) for model in models}
    if type_by_name[cfg["new_model_name"]] != "multimodal":
        raise ValueError("new_model_name must identify a multimodal model")
    if type_by_name[cfg["baseline_model_name"]] != "multimodal":
        raise ValueError("baseline_model_name must identify a multimodal model")
    if type_by_name[cfg["optical_null_model_name"]] != "optical":
        raise ValueError("optical_null_model_name must identify an optical model")
    return cfg


def robust_standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median/IQR-standardize a finite two-dimensional feature matrix."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("values must be a non-empty two-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("values contain non-finite entries")
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (values - center) / scale, center, scale


def derive_physical_features(gw_scalars: np.ndarray) -> np.ndarray:
    """Return model-visible GW features used to define pairing distance."""
    scalars = np.asarray(gw_scalars, dtype=np.float64)
    if scalars.ndim != 2 or scalars.shape[1] != 7:
        raise ValueError("gw_scalars must have shape [N, 7]")
    mass_high = np.maximum(scalars[:, 0], scalars[:, 1])
    mass_low = np.minimum(scalars[:, 0], scalars[:, 1])
    if np.any(mass_low <= 0.0) or np.any(scalars[:, 5] <= 0.0):
        raise ValueError("GW masses and distance means must be positive")
    chirp_mass = (mass_high * mass_low) ** (3.0 / 5.0) / (mass_high + mass_low) ** (
        1.0 / 5.0
    )
    mass_ratio = mass_low / mass_high
    chi_eff = (scalars[:, 0] * scalars[:, 2] + scalars[:, 1] * scalars[:, 3]) / (
        scalars[:, 0] + scalars[:, 1]
    )
    output = np.column_stack(
        [
            chirp_mass,
            mass_ratio,
            chi_eff,
            np.abs(scalars[:, 4]),
            np.log10(scalars[:, 5]),
        ]
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("Derived physical features contain non-finite values")
    return output


def derive_single_parameter_features(
    gw_scalars: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the recovered, model-visible coordinates used by the v1 profile."""
    scalars = np.asarray(gw_scalars, dtype=np.float64)
    if scalars.ndim != 2 or scalars.shape[1] != 7:
        raise ValueError("gw_scalars must have shape [N, 7]")
    mass_1, mass_2 = scalars[:, 0], scalars[:, 1]
    spin_1, spin_2 = scalars[:, 2], scalars[:, 3]
    if np.any(mass_1 <= 0.0) or np.any(mass_2 <= 0.0):
        raise ValueError("GW masses must be positive")
    if np.any(scalars[:, 5] <= 0.0):
        raise ValueError("GW distance means must be positive")
    mass_high = np.maximum(mass_1, mass_2)
    mass_low = np.minimum(mass_1, mass_2)
    features = {
        "chirp_mass_detector": (mass_1 * mass_2) ** (3.0 / 5.0)
        / (mass_1 + mass_2) ** (1.0 / 5.0),
        "mass_ratio": mass_low / mass_high,
        "chi_eff": (mass_1 * spin_1 + mass_2 * spin_2) / (mass_1 + mass_2),
        "primary_spin_z": np.where(mass_1 >= mass_2, spin_1, spin_2),
        "abs_costheta": np.abs(scalars[:, 4]),
        "log10_distance_gpc": np.log10(scalars[:, 5]),
    }
    if any(not np.all(np.isfinite(values)) for values in features.values()):
        raise ValueError("Derived single-parameter features contain non-finite values")
    return features


def optical_sampling_features(
    times: np.ndarray,
    masks: np.ndarray,
    *,
    window_start: float,
    window_end: float,
) -> np.ndarray:
    """Compute observing-coverage features without reading flux or errors."""
    times = np.asarray(times, dtype=np.float64)
    masks = np.asarray(masks) > 0
    if masks.ndim != 3 or times.shape != masks.shape[:2]:
        raise ValueError("times/masks shapes must be [N,L] and [N,L,B]")
    in_window = (times >= float(window_start)) & (times <= float(window_end))
    valid = masks & in_window[:, :, None]
    nobs = valid.sum(axis=(1, 2)).astype(np.float64)
    nband = np.any(valid, axis=1).sum(axis=1).astype(np.float64) / masks.shape[2]
    spans = np.zeros(times.shape[0], dtype=np.float64)
    for idx in range(times.shape[0]):
        valid_rows = np.any(valid[idx], axis=1)
        selected = times[idx, valid_rows]
        spans[idx] = float(np.ptp(selected)) if selected.size > 1 else 0.0
    return np.column_stack([np.log1p(nobs), nband, np.log1p(spans)])


def _standardize_by_source(
    features: np.ndarray, source_types: Sequence[str]
) -> tuple[np.ndarray, dict[str, dict[str, list[float]]]]:
    source = np.asarray([str(value).strip().lower() for value in source_types])
    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] != source.size:
        raise ValueError("features and source_types must have equal length")
    standardized = np.empty_like(features)
    audit: dict[str, dict[str, list[float]]] = {}
    for label in sorted(np.unique(source).tolist()):
        rows = np.flatnonzero(source == label)
        standardized[rows], center, scale = robust_standardize(features[rows])
        audit[label] = {"center": center.tolist(), "scale": scale.tolist()}
    return standardized, audit


def _parent_feature_medians(
    parent_to_curves: Mapping[int, np.ndarray],
    gw_ids: Sequence[int],
    curve_features: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            np.median(curve_features[np.asarray(parent_to_curves[int(gw_id)])], axis=0)
            for gw_id in gw_ids
        ],
        dtype=np.float64,
    )


def build_event_pairs(
    *,
    gw_ids: Sequence[int],
    source_types: Sequence[str],
    physical_features: np.ndarray,
    nuisance_features: np.ndarray,
    physical_distance_quantile: float,
    nuisance_l1_max: float,
    nuisance_feature_max_abs: float,
    min_pairs_per_source: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic, event-disjoint, same-source crossed pairs."""
    gw_ids = np.asarray(gw_ids, dtype=np.int64)
    source = np.asarray([str(value).strip().lower() for value in source_types])
    physical = np.asarray(physical_features, dtype=np.float64)
    nuisance = np.asarray(nuisance_features, dtype=np.float64)
    if not (
        gw_ids.ndim == 1
        and source.shape == gw_ids.shape
        and physical.shape[0] == gw_ids.size
        and nuisance.shape[0] == gw_ids.size
    ):
        raise ValueError("Event arrays must have equal first dimensions")
    physical_z, physical_audit = _standardize_by_source(physical, source)
    nuisance_z, nuisance_audit = _standardize_by_source(nuisance, source)
    pairs: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    thresholds: dict[str, float] = {}

    for label in sorted(np.unique(source).tolist()):
        local = np.flatnonzero(source == label)
        p = physical_z[local]
        n = nuisance_z[local]
        physical_matrix = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        upper = physical_matrix[np.triu_indices(local.size, k=1)]
        threshold = float(np.quantile(upper, float(physical_distance_quantile)))
        thresholds[label] = threshold
        edges: list[tuple[float, float, int, int, float]] = []
        for left in range(local.size):
            differences = np.abs(n[left + 1 :] - n[left])
            if differences.size == 0:
                continue
            l1 = differences.sum(axis=1)
            max_abs = differences.max(axis=1)
            physical_distance = physical_matrix[left, left + 1 :]
            valid = (
                (physical_distance >= threshold)
                & (l1 <= float(nuisance_l1_max))
                & (max_abs <= float(nuisance_feature_max_abs))
            )
            for offset in np.flatnonzero(valid).tolist():
                right = left + 1 + int(offset)
                edges.append(
                    (
                        float(l1[offset]),
                        -float(physical_distance[offset]),
                        int(left),
                        int(right),
                        float(max_abs[offset]),
                    )
                )
        used: set[int] = set()
        source_pairs: list[dict[str, Any]] = []
        for nuisance_l1, negative_physical, left, right, nuisance_max in sorted(
            edges,
            key=lambda item: (
                item[0],
                item[1],
                int(gw_ids[local[item[2]]]),
                int(gw_ids[local[item[3]]]),
            ),
        ):
            if left in used or right in used:
                continue
            used.update((left, right))
            row_a = int(local[left])
            row_b = int(local[right])
            if int(gw_ids[row_b]) < int(gw_ids[row_a]):
                row_a, row_b = row_b, row_a
                left, right = right, left
            record: dict[str, Any] = {
                "pair_id": "",
                "source_type": label,
                "gw_a": int(gw_ids[row_a]),
                "gw_b": int(gw_ids[row_b]),
                "physical_distance": float(-negative_physical),
                "physical_distance_threshold": threshold,
                "nuisance_l1_distance": float(nuisance_l1),
                "nuisance_max_abs_difference": float(nuisance_max),
            }
            for feature_idx, name in enumerate(PHYSICAL_FEATURE_NAMES):
                record[f"delta_{name}"] = float(
                    abs(physical_z[row_a, feature_idx] - physical_z[row_b, feature_idx])
                )
            for feature_idx, name in enumerate(EVENT_NUISANCE_FEATURE_NAMES):
                record[f"delta_{name}"] = float(
                    abs(nuisance_z[row_a, feature_idx] - nuisance_z[row_b, feature_idx])
                )
            source_pairs.append(record)
        source_pairs.sort(key=lambda row: (row["gw_a"], row["gw_b"]))
        for pair_index, record in enumerate(source_pairs):
            record["pair_id"] = f"{label}_{pair_index:04d}"
        if len(source_pairs) < int(min_pairs_per_source):
            raise ValueError(
                f"{label} produced only {len(source_pairs)} event pairs; "
                f"minimum is {min_pairs_per_source}. Refusing to relax matching calipers."
            )
        counts[label] = len(source_pairs)
        pairs.extend(source_pairs)

    audit = {
        "pair_counts": counts,
        "physical_distance_thresholds": thresholds,
        "physical_standardization": physical_audit,
        "nuisance_standardization": nuisance_audit,
    }
    return pairs, audit


def _condition_id(target: str, source: str, caliper: float) -> str:
    caliper_text = f"{float(caliper):g}".replace(".", "p")
    return f"{target}__{source}__caliper_{caliper_text}"


def build_single_parameter_event_pairs(
    *,
    gw_ids: Sequence[int],
    source_types: Sequence[str],
    feature_values: Mapping[str, np.ndarray],
    nuisance_features: np.ndarray,
    target_min_abs_iqr: float,
    other_calipers_iqr: Sequence[float],
    primary_caliper_iqr: float,
    min_pairs_by_caliper: Mapping[float, int],
    nuisance_l1_max: float,
    nuisance_feature_max_abs: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build deterministic, disjoint pairs that primarily differ in one GW coordinate."""
    gw_ids = np.asarray(gw_ids, dtype=np.int64)
    source = np.asarray([str(value).strip().lower() for value in source_types])
    nuisance = np.asarray(nuisance_features, dtype=np.float64)
    if gw_ids.ndim != 1 or source.shape != gw_ids.shape:
        raise ValueError("gw_ids and source_types must be equal one-dimensional arrays")
    if nuisance.ndim != 2 or nuisance.shape[0] != gw_ids.size:
        raise ValueError("nuisance_features must have one row per GW event")
    values = {
        str(name): np.asarray(column, dtype=np.float64).reshape(-1)
        for name, column in feature_values.items()
    }
    required = set().union(*SINGLE_PARAMETER_FEATURES_BY_SOURCE.values())
    if not required.issubset(values):
        raise ValueError(
            f"Missing single-parameter features: {sorted(required.difference(values))}"
        )
    if any(column.shape != gw_ids.shape for column in values.values()):
        raise ValueError("Every single-parameter feature must have one value per event")
    if any(not np.all(np.isfinite(column)) for column in values.values()):
        raise ValueError("Single-parameter features contain non-finite values")

    nuisance_z, nuisance_audit = _standardize_by_source(nuisance, source)
    all_pairs: list[dict[str, Any]] = []
    condition_audit: dict[str, Any] = {}
    feature_audit: dict[str, Any] = {}
    seen_sources = set(np.unique(source).tolist())
    missing_sources = set(SINGLE_PARAMETER_FEATURES_BY_SOURCE).difference(seen_sources)
    if missing_sources:
        raise ValueError(f"Missing source types: {sorted(missing_sources)}")

    for source_label, basis_names in SINGLE_PARAMETER_FEATURES_BY_SOURCE.items():
        local = np.flatnonzero(source == source_label)
        raw_basis = np.column_stack([values[name][local] for name in basis_names])
        physical_z, center, scale = robust_standardize(raw_basis)
        feature_audit[source_label] = {
            "feature_names": list(basis_names),
            "center": center.tolist(),
            "scale": scale.tolist(),
        }
        nuisance_local = nuisance_z[local]
        position = {name: index for index, name in enumerate(basis_names)}
        for target in basis_names:
            target_index = position[target]
            other_indices = [
                index for index in range(len(basis_names)) if index != target_index
            ]
            for caliper in sorted(float(value) for value in other_calipers_iqr):
                condition = _condition_id(target, source_label, caliper)
                edges: list[tuple[Any, ...]] = []
                for left in range(local.size - 1):
                    physical_delta = np.abs(physical_z[left + 1 :] - physical_z[left])
                    nuisance_delta = np.abs(
                        nuisance_local[left + 1 :] - nuisance_local[left]
                    )
                    target_delta = physical_delta[:, target_index]
                    other_delta = physical_delta[:, other_indices]
                    other_l1 = other_delta.sum(axis=1)
                    other_max = other_delta.max(axis=1)
                    nuisance_l1 = nuisance_delta.sum(axis=1)
                    nuisance_max = nuisance_delta.max(axis=1)
                    valid = (
                        (target_delta >= float(target_min_abs_iqr))
                        & (other_max <= caliper)
                        & (nuisance_l1 <= float(nuisance_l1_max))
                        & (nuisance_max <= float(nuisance_feature_max_abs))
                    )
                    for offset in np.flatnonzero(valid).tolist():
                        right = left + 1 + int(offset)
                        edges.append(
                            (
                                float(other_max[offset]),
                                float(other_l1[offset]),
                                float(nuisance_l1[offset]),
                                -float(target_delta[offset]),
                                int(gw_ids[local[left]]),
                                int(gw_ids[local[right]]),
                                int(left),
                                int(right),
                                float(nuisance_max[offset]),
                            )
                        )
                used: set[int] = set()
                selected: list[dict[str, Any]] = []
                for edge in sorted(edges):
                    (
                        other_max,
                        other_l1,
                        nuisance_l1,
                        negative_target,
                        _,
                        _,
                        left,
                        right,
                        nuisance_max,
                    ) = edge
                    if left in used or right in used:
                        continue
                    used.update((left, right))
                    row_a = int(local[left])
                    row_b = int(local[right])
                    if int(gw_ids[row_b]) < int(gw_ids[row_a]):
                        row_a, row_b = row_b, row_a
                    physical_delta = np.abs(
                        physical_z[np.flatnonzero(local == row_a)[0]]
                        - physical_z[np.flatnonzero(local == row_b)[0]]
                    )
                    nuisance_delta = np.abs(nuisance_z[row_a] - nuisance_z[row_b])
                    record: dict[str, Any] = {
                        "condition_id": condition,
                        "target_parameter": target,
                        "caliper_iqr": float(caliper),
                        "is_primary_caliper": bool(
                            np.isclose(
                                caliper, primary_caliper_iqr, atol=1e-12, rtol=0.0
                            )
                        ),
                        "pair_id": "",
                        "source_type": source_label,
                        "gw_a": int(gw_ids[row_a]),
                        "gw_b": int(gw_ids[row_b]),
                        "target_delta_iqr": float(-negative_target),
                        "other_parameter_l1_distance": float(other_l1),
                        "other_parameter_max_abs_difference": float(other_max),
                        "physical_distance": float(np.linalg.norm(physical_delta)),
                        "nuisance_l1_distance": float(nuisance_l1),
                        "nuisance_max_abs_difference": float(nuisance_max),
                    }
                    for name in SINGLE_PARAMETER_FEATURE_NAMES:
                        record[f"delta_{name}"] = None
                    for feature_index, name in enumerate(basis_names):
                        record[f"delta_{name}"] = float(physical_delta[feature_index])
                    for feature_index, name in enumerate(EVENT_NUISANCE_FEATURE_NAMES):
                        record[f"delta_{name}"] = float(nuisance_delta[feature_index])
                    selected.append(record)
                selected.sort(key=lambda row: (row["gw_a"], row["gw_b"]))
                for pair_index, record in enumerate(selected):
                    record["pair_id"] = f"{condition}__pair_{pair_index:04d}"
                minimum = int(min_pairs_by_caliper[float(caliper)])
                if len(selected) < minimum:
                    raise ValueError(
                        f"{condition} produced only {len(selected)} event pairs; "
                        f"minimum is {minimum}. Refusing to relax matching calipers."
                    )
                condition_audit[condition] = {
                    "target_parameter": target,
                    "source_type": source_label,
                    "caliper_iqr": float(caliper),
                    "is_primary_caliper": bool(
                        np.isclose(caliper, primary_caliper_iqr, atol=1e-12, rtol=0.0)
                    ),
                    "n_pairs": len(selected),
                    "candidate_edges": len(edges),
                }
                all_pairs.extend(selected)

    return all_pairs, {
        "profile": SINGLE_PARAMETER_PROFILE_NAME,
        "target_min_abs_iqr": float(target_min_abs_iqr),
        "conditions": condition_audit,
        "feature_standardization": feature_audit,
        "nuisance_standardization": nuisance_audit,
    }


def pair_curve_realizations(
    *,
    event_pairs: Sequence[Mapping[str, Any]],
    parent_to_curves: Mapping[int, np.ndarray],
    curve_features: np.ndarray,
    curve_source_types: Sequence[str],
    max_curve_pairs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose sampling-matched curves without inspecting brightness or flux."""
    curve_z, standardization = _standardize_by_source(
        np.asarray(curve_features, dtype=np.float64), curve_source_types
    )
    records: list[dict[str, Any]] = []
    counts: list[int] = []
    for pair in event_pairs:
        curves_a = np.asarray(parent_to_curves[int(pair["gw_a"])], dtype=np.int64)
        curves_b = np.asarray(parent_to_curves[int(pair["gw_b"])], dtype=np.int64)
        edges: list[tuple[float, int, int]] = []
        for optical_a in curves_a.tolist():
            distances = np.abs(curve_z[curves_b] - curve_z[optical_a]).sum(axis=1)
            edges.extend(
                (float(distance), int(optical_a), int(optical_b))
                for distance, optical_b in zip(distances.tolist(), curves_b.tolist())
            )
        used_a: set[int] = set()
        used_b: set[int] = set()
        selected: list[tuple[float, int, int]] = []
        for distance, optical_a, optical_b in sorted(
            edges, key=lambda item: (item[0], item[1], item[2])
        ):
            if optical_a in used_a or optical_b in used_b:
                continue
            used_a.add(optical_a)
            used_b.add(optical_b)
            selected.append((distance, optical_a, optical_b))
            if len(selected) >= int(max_curve_pairs):
                break
        if not selected:
            raise ValueError(f"No curve realization pair for {pair['pair_id']}")
        counts.append(len(selected))
        for curve_pair_id, (distance, optical_a, optical_b) in enumerate(selected):
            record: dict[str, Any] = {
                "pair_id": str(pair["pair_id"]),
                "curve_pair_id": int(curve_pair_id),
                "source_type": str(pair["source_type"]),
                "gw_a": int(pair["gw_a"]),
                "gw_b": int(pair["gw_b"]),
                "optical_index_a": int(optical_a),
                "optical_index_b": int(optical_b),
                "curve_sampling_l1_distance": float(distance),
            }
            for feature_idx, name in enumerate(CURVE_SAMPLING_FEATURE_NAMES):
                record[f"delta_{name}"] = float(
                    abs(
                        curve_z[optical_a, feature_idx]
                        - curve_z[optical_b, feature_idx]
                    )
                )
            records.append(record)
    return records, {
        "standardization": standardization,
        "curve_pairs_per_event_pair": {
            "min": int(min(counts)),
            "median": float(np.median(counts)),
            "max": int(max(counts)),
        },
    }


def crossed_interaction(
    score_aa: float, score_ab: float, score_ba: float, score_bb: float
) -> dict[str, float]:
    """Compute the crossed score interaction and tie-aware directional wins."""
    values = np.asarray([score_aa, score_ab, score_ba, score_bb], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Crossed scores must be finite")
    margin_a = float(score_aa - score_ab)
    margin_b = float(score_bb - score_ba)

    def win(margin: float) -> float:
        return 1.0 if margin > 0.0 else 0.0 if margin < 0.0 else 0.5

    return {
        "margin_a": margin_a,
        "margin_b": margin_b,
        "interaction": 0.5 * (margin_a + margin_b),
        "directional_win_rate": 0.5 * (win(margin_a) + win(margin_b)),
    }


def build_candidate_layout(
    curve_pairs: Sequence[Mapping[str, Any]],
    *,
    compact_index: Mapping[int, int],
    coordinates: np.ndarray,
    first_detection_mjd: np.ndarray,
    event_time_mjd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, str]]]:
    """Build repeated A/B candidates for the two counterbalanced anchors."""
    candidate_indices: list[int] = []
    candidate_coordinates: list[np.ndarray] = []
    candidate_dt: list[float] = []
    slots: list[tuple[int, str]] = []
    for curve_pair in curve_pairs:
        optical_a = int(curve_pair["optical_index_a"])
        optical_b = int(curve_pair["optical_index_b"])
        gw_a = int(curve_pair["gw_a"])
        gw_b = int(curve_pair["gw_b"])
        anchors = (
            (
                "a",
                np.asarray(coordinates[optical_a], dtype=np.float32),
                abs(
                    float(first_detection_mjd[optical_a]) - float(event_time_mjd[gw_a])
                ),
            ),
            (
                "b",
                np.asarray(coordinates[optical_b], dtype=np.float32),
                abs(
                    float(first_detection_mjd[optical_b]) - float(event_time_mjd[gw_b])
                ),
            ),
        )
        for anchor, coordinate, dt_days in anchors:
            candidate_indices.extend(
                [compact_index[optical_a], compact_index[optical_b]]
            )
            candidate_coordinates.extend([coordinate, coordinate])
            candidate_dt.extend([dt_days, dt_days])
            slots.append((int(curve_pair["curve_pair_id"]), anchor))
    return (
        np.asarray(candidate_indices, dtype=np.int64),
        np.asarray(candidate_coordinates, dtype=np.float32),
        np.asarray(candidate_dt, dtype=np.float32),
        slots,
    )


def _read_metadata(
    test_data_path: str, comparison_window: Sequence[float]
) -> dict[str, Any]:
    with h5py.File(test_data_path, "r") as handle:
        gw = handle["events/gw_data"]
        opt = handle["events/optical_data"]
        parent = np.asarray(opt["parent_gw_idx"][:], dtype=np.int64)
        source_types = np.asarray(
            [value.strip().lower() for value in _decode_strings(gw["source_type"][:])],
            dtype=object,
        )
        curve_sources = source_types[parent]
        sampling = optical_sampling_features(
            np.asarray(opt["times"][:], dtype=np.float32),
            np.asarray(opt["masks"][:], dtype=np.float32),
            window_start=float(comparison_window[0]),
            window_end=float(comparison_window[1]),
        )
        first_detection = (
            np.asarray(opt["first_detection_mjd"][:], dtype=np.float64)
            if "first_detection_mjd" in opt
            else np.asarray(opt["zero_time_mjd_base"][:], dtype=np.float64)
        )
        return {
            "gw_scalars": np.asarray(gw["scalars"][:], dtype=np.float64),
            "gw_source_types": source_types,
            "gw_ids": _decode_strings(gw["ids"][:]),
            "event_time_mjd": np.asarray(gw["event_time_mjd"][:], dtype=np.float64),
            "optical_parent": parent,
            "curve_source_types": curve_sources,
            "curve_sampling_features": sampling,
            "coordinates": np.asarray(opt["coordinates"][:], dtype=np.float32),
            "first_detection_mjd": first_detection,
        }


def _prepare_pairs(metadata: Mapping[str, Any], cfg: Mapping[str, Any]):
    parent = np.asarray(metadata["optical_parent"], dtype=np.int64)
    parent_to_curves: dict[int, list[int]] = defaultdict(list)
    for optical_index, gw_id in enumerate(parent.tolist()):
        parent_to_curves[int(gw_id)].append(int(optical_index))
    parent_map = {
        gw_id: np.asarray(indices, dtype=np.int64)
        for gw_id, indices in parent_to_curves.items()
    }
    detected_gw = np.asarray(sorted(parent_map), dtype=np.int64)
    scalars = np.asarray(metadata["gw_scalars"], dtype=np.float64)
    curve_sampling = np.asarray(metadata["curve_sampling_features"], dtype=np.float64)
    source = np.asarray(metadata["gw_source_types"], dtype=object)[detected_gw]
    parent_sampling = _parent_feature_medians(parent_map, detected_gw, curve_sampling)
    fractional_distance_uncertainty = scalars[detected_gw, 6] / np.maximum(
        scalars[detected_gw, 5], 1e-12
    )
    if np.any(fractional_distance_uncertainty <= 0.0):
        raise ValueError("Fractional GW distance uncertainties must be positive")
    nuisance = np.column_stack(
        [np.log10(fractional_distance_uncertainty), parent_sampling]
    )
    physical = derive_physical_features(scalars[detected_gw])
    event_pairs, event_audit = build_event_pairs(
        gw_ids=detected_gw,
        source_types=source,
        physical_features=physical,
        nuisance_features=nuisance,
        physical_distance_quantile=cfg["physical_distance_quantile"],
        nuisance_l1_max=cfg["event_nuisance_l1_max"],
        nuisance_feature_max_abs=cfg["event_nuisance_feature_max_abs"],
        min_pairs_per_source=cfg["min_pairs_per_source"],
    )
    curve_pairs, curve_audit = pair_curve_realizations(
        event_pairs=event_pairs,
        parent_to_curves=parent_map,
        curve_features=curve_sampling,
        curve_source_types=metadata["curve_source_types"],
        max_curve_pairs=cfg["max_curve_pairs_per_event_pair"],
    )
    return event_pairs, curve_pairs, {"event": event_audit, "curve": curve_audit}


def _prepare_single_parameter_pairs(
    metadata: Mapping[str, Any], cfg: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    parent = np.asarray(metadata["optical_parent"], dtype=np.int64)
    parent_to_curves: dict[int, list[int]] = defaultdict(list)
    for optical_index, gw_id in enumerate(parent.tolist()):
        parent_to_curves[int(gw_id)].append(int(optical_index))
    parent_map = {
        gw_id: np.asarray(indices, dtype=np.int64)
        for gw_id, indices in parent_to_curves.items()
    }
    detected_gw = np.asarray(sorted(parent_map), dtype=np.int64)
    scalars = np.asarray(metadata["gw_scalars"], dtype=np.float64)
    curve_sampling = np.asarray(metadata["curve_sampling_features"], dtype=np.float64)
    source = np.asarray(metadata["gw_source_types"], dtype=object)[detected_gw]
    parent_sampling = _parent_feature_medians(parent_map, detected_gw, curve_sampling)
    fractional_distance_uncertainty = scalars[detected_gw, 6] / np.maximum(
        scalars[detected_gw, 5], 1e-12
    )
    if np.any(fractional_distance_uncertainty <= 0.0):
        raise ValueError("Fractional GW distance uncertainties must be positive")
    nuisance = np.column_stack(
        [np.log10(fractional_distance_uncertainty), parent_sampling]
    )
    features = derive_single_parameter_features(scalars[detected_gw])
    event_pairs, event_audit = build_single_parameter_event_pairs(
        gw_ids=detected_gw,
        source_types=source,
        feature_values=features,
        nuisance_features=nuisance,
        target_min_abs_iqr=cfg["target_min_abs_iqr"],
        other_calipers_iqr=cfg["other_parameter_calipers_iqr"],
        primary_caliper_iqr=cfg["primary_other_parameter_caliper_iqr"],
        min_pairs_by_caliper=cfg["min_pairs_by_caliper"],
        nuisance_l1_max=cfg["event_nuisance_l1_max"],
        nuisance_feature_max_abs=cfg["event_nuisance_feature_max_abs"],
    )
    curve_pairs, curve_audit = pair_curve_realizations(
        event_pairs=event_pairs,
        parent_to_curves=parent_map,
        curve_features=curve_sampling,
        curve_source_types=metadata["curve_source_types"],
        max_curve_pairs=cfg["max_curve_pairs_per_event_pair"],
    )
    pair_lookup = _pair_lookup(event_pairs)
    for curve_pair in curve_pairs:
        pair = pair_lookup[str(curve_pair["pair_id"])]
        curve_pair.update({field: pair[field] for field in CONDITION_FIELDS})
    return event_pairs, curve_pairs, {"event": event_audit, "curve": curve_audit}


def _pair_lookup(
    event_pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {str(row["pair_id"]): row for row in event_pairs}


def _optional_condition_values(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {field: pair[field] for field in CONDITION_FIELDS if field in pair}


def _score_rows_for_block(
    *,
    model_name: str,
    model_type: str,
    pair: Mapping[str, Any],
    curve_pair: Mapping[str, Any],
    anchor: str,
    anchor_coordinate: np.ndarray,
    anchor_dt: float,
    scores: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    score_aa, score_ab, score_ba, score_bb = scores
    optical_a = int(curve_pair["optical_index_a"])
    optical_b = int(curve_pair["optical_index_b"])
    cells = (
        ("a", "a", int(pair["gw_a"]), int(pair["gw_a"]), optical_a, True, score_aa),
        ("a", "b", int(pair["gw_a"]), int(pair["gw_b"]), optical_b, False, score_ab),
        ("b", "a", int(pair["gw_b"]), int(pair["gw_a"]), optical_a, False, score_ba),
        ("b", "b", int(pair["gw_b"]), int(pair["gw_b"]), optical_b, True, score_bb),
    )
    return [
        {
            "model": model_name,
            "model_type": model_type,
            "pair_id": str(pair["pair_id"]),
            **_optional_condition_values(pair),
            "curve_pair_id": int(curve_pair["curve_pair_id"]),
            "source_type": str(pair["source_type"]),
            "anchor": anchor,
            "query_side": query_side,
            "candidate_side": candidate_side,
            "query_gw_id": query_gw,
            "candidate_parent_gw_id": candidate_parent,
            "optical_index": optical_index,
            "is_aligned": bool(aligned),
            "score": float(score),
            "anchor_ra": float(anchor_coordinate[0]),
            "anchor_dec": float(anchor_coordinate[1]),
            "anchor_dt_days": float(anchor_dt),
        }
        for (
            query_side,
            candidate_side,
            query_gw,
            candidate_parent,
            optical_index,
            aligned,
            score,
        ) in cells
    ]


def _interaction_row(
    *,
    model_name: str,
    model_type: str,
    pair: Mapping[str, Any],
    curve_pair_id: int,
    anchor: str,
    scores: tuple[float, float, float, float],
) -> dict[str, Any]:
    values = crossed_interaction(*scores)
    return {
        "model": model_name,
        "model_type": model_type,
        "pair_id": str(pair["pair_id"]),
        **_optional_condition_values(pair),
        "curve_pair_id": int(curve_pair_id),
        "source_type": str(pair["source_type"]),
        "anchor": anchor,
        "physical_distance": float(pair["physical_distance"]),
        "score_aa": float(scores[0]),
        "score_ab": float(scores[1]),
        "score_ba": float(scores[2]),
        "score_bb": float(scores[3]),
        **values,
    }


def _score_multimodal(
    *,
    model_spec: Mapping[str, Any],
    cfg: Mapping[str, Any],
    bank: Mapping[str, torch.Tensor],
    compact_index: Mapping[int, int],
    event_pairs: Sequence[Mapping[str, Any]],
    curve_pairs: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    name = str(model_spec["name"])
    comparison_window = tuple(cfg["comparison_window"])
    model, model_args, saved_args = base.load_multimodal_bundle(
        model_spec["resolved_checkpoint"],
        model_spec["resolved_config"],
        device,
        test_data_path=cfg["test_data_path"],
        neg_data_path=None,
        comparison_window=comparison_window,
        nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
    )
    if base._resolve_multimodal_scoring(dict(model_spec), saved_args) != "logits":
        raise ValueError(f"{name} must support fusion-logit scoring")
    embeddings = base.extract_optical_candidate_embeddings(
        model,
        bank,
        device,
        n_ref=int(model_args.get("n_ref", 64)),
        ref_start=float(model_args["ref_start"]),
        ref_end=float(model_args["ref_end"]),
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
        desc=f"  Encoding selected KN curves for {name}",
    )
    unique_gw = sorted(
        {int(value) for pair in event_pairs for value in (pair["gw_a"], pair["gw_b"])}
    )
    query_cache = base._build_gallery_query_cache(
        model,
        unique_gw,
        cfg["test_data_path"],
        device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    dual = base._is_dual_fusion_model(model)
    curves_by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for curve_pair in curve_pairs:
        curves_by_pair[str(curve_pair["pair_id"])].append(curve_pair)
    score_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []

    for pair in tqdm(event_pairs, desc=f"  Cross-scoring {name}"):
        selected_curves = sorted(
            curves_by_pair[str(pair["pair_id"])],
            key=lambda row: int(row["curve_pair_id"]),
        )
        indices, coordinates, dt_days, slots = build_candidate_layout(
            selected_curves,
            compact_index=compact_index,
            coordinates=metadata["coordinates"],
            first_detection_mjd=metadata["first_detection_mjd"],
            event_time_mjd=metadata["event_time_mjd"],
        )
        scores_a = base._score_candidate_bank_with_logits(
            model,
            query_cache[int(pair["gw_a"])],
            indices,
            embeddings,
            device,
            dual,
            candidate_coords=coordinates,
            candidate_abs_dt_days=dt_days,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        scores_b = base._score_candidate_bank_with_logits(
            model,
            query_cache[int(pair["gw_b"])],
            indices,
            embeddings,
            device,
            dual,
            candidate_coords=coordinates,
            candidate_abs_dt_days=dt_days,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        curve_lookup = {int(row["curve_pair_id"]): row for row in selected_curves}
        for slot_index, (curve_pair_id, anchor) in enumerate(slots):
            offset = slot_index * 2
            scores = (
                float(scores_a[offset]),
                float(scores_a[offset + 1]),
                float(scores_b[offset]),
                float(scores_b[offset + 1]),
            )
            curve_pair = curve_lookup[curve_pair_id]
            anchor_optical_index = int(curve_pair[f"optical_index_{anchor}"])
            anchor_coordinate = np.asarray(
                metadata["coordinates"][anchor_optical_index], dtype=np.float32
            )
            anchor_gw = int(pair[f"gw_{anchor}"])
            anchor_dt = abs(
                float(metadata["first_detection_mjd"][anchor_optical_index])
                - float(metadata["event_time_mjd"][anchor_gw])
            )
            score_rows.extend(
                _score_rows_for_block(
                    model_name=name,
                    model_type="multimodal",
                    pair=pair,
                    curve_pair=curve_pair,
                    anchor=anchor,
                    anchor_coordinate=anchor_coordinate,
                    anchor_dt=anchor_dt,
                    scores=scores,
                )
            )
            interaction_rows.append(
                _interaction_row(
                    model_name=name,
                    model_type="multimodal",
                    pair=pair,
                    curve_pair_id=curve_pair_id,
                    anchor=anchor,
                    scores=scores,
                )
            )
    model_info = {
        "name": name,
        "type": "multimodal",
        "resolved_checkpoint": str(model_spec["resolved_checkpoint"]),
        "resolved_config": str(model_spec["resolved_config"]),
    }
    del query_cache, embeddings, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score_rows, interaction_rows, model_info


def _score_optical_null(
    *,
    model_spec: Mapping[str, Any],
    cfg: Mapping[str, Any],
    bank: Mapping[str, torch.Tensor],
    compact_index: Mapping[int, int],
    event_pairs: Sequence[Mapping[str, Any]],
    curve_pairs: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    name = str(model_spec["name"])
    model, checkpoint_args = base.load_optical_model(
        model_spec["resolved_checkpoint"], device
    )
    all_scores = base.score_gallery_optical_only(
        model,
        np.arange(bank["times"].shape[0], dtype=np.int64),
        bank["times"],
        bank["values"],
        bank["masks"],
        bank["errors"],
        device,
        n_ref=int(checkpoint_args.get("n_ref", 64)),
        ref_start=float(cfg["comparison_window"][0]),
        ref_end=float(cfg["comparison_window"][1]),
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    pair_lookup = _pair_lookup(event_pairs)
    score_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    for curve_pair in curve_pairs:
        pair = pair_lookup[str(curve_pair["pair_id"])]
        score_a = float(all_scores[compact_index[int(curve_pair["optical_index_a"])]])
        score_b = float(all_scores[compact_index[int(curve_pair["optical_index_b"])]])
        scores = (score_a, score_b, score_a, score_b)
        for anchor in ("a", "b"):
            anchor_optical_index = int(curve_pair[f"optical_index_{anchor}"])
            anchor_coordinate = np.asarray(
                metadata["coordinates"][anchor_optical_index], dtype=np.float32
            )
            anchor_gw = int(pair[f"gw_{anchor}"])
            anchor_dt = abs(
                float(metadata["first_detection_mjd"][anchor_optical_index])
                - float(metadata["event_time_mjd"][anchor_gw])
            )
            score_rows.extend(
                _score_rows_for_block(
                    model_name=name,
                    model_type="optical",
                    pair=pair,
                    curve_pair=curve_pair,
                    anchor=anchor,
                    anchor_coordinate=anchor_coordinate,
                    anchor_dt=anchor_dt,
                    scores=scores,
                )
            )
            interaction_rows.append(
                _interaction_row(
                    model_name=name,
                    model_type="optical",
                    pair=pair,
                    curve_pair_id=int(curve_pair["curve_pair_id"]),
                    anchor=anchor,
                    scores=scores,
                )
            )
    model_info = {
        "name": name,
        "type": "optical",
        "resolved_checkpoint": str(model_spec["resolved_checkpoint"]),
        "resolved_config": model_spec.get("resolved_config"),
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return score_rows, interaction_rows, model_info


def aggregate_pair_metrics(
    interaction_rows: Sequence[Mapping[str, Any]],
    event_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Average anchors, then curve realizations, leaving event pairs independent."""
    frame = pd.DataFrame(interaction_rows)
    if frame.empty:
        raise ValueError("No interaction rows to aggregate")
    curve = frame.groupby(
        ["model", "model_type", "pair_id", "curve_pair_id", "source_type"],
        as_index=False,
        sort=True,
    )[
        [
            "physical_distance",
            "interaction",
            "directional_win_rate",
            "margin_a",
            "margin_b",
        ]
    ].mean()
    pair_frame = curve.groupby(
        ["model", "model_type", "pair_id", "source_type"],
        as_index=False,
        sort=True,
    )[
        [
            "physical_distance",
            "interaction",
            "directional_win_rate",
            "margin_a",
            "margin_b",
        ]
    ].mean()
    counts = curve.groupby(["model", "pair_id"]).size().rename("n_curve_pairs")
    pair_frame = pair_frame.merge(
        counts, on=["model", "pair_id"], validate="one_to_one"
    )
    pair_lookup = _pair_lookup(event_pairs)
    pair_frame["gw_a"] = pair_frame["pair_id"].map(
        lambda pair_id: int(pair_lookup[str(pair_id)]["gw_a"])
    )
    pair_frame["gw_b"] = pair_frame["pair_id"].map(
        lambda pair_id: int(pair_lookup[str(pair_id)]["gw_b"])
    )
    return pair_frame[list(PAIR_METRIC_FIELDS)].to_dict("records")


def aggregate_single_parameter_pair_metrics(
    interaction_rows: Sequence[Mapping[str, Any]],
    event_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = aggregate_pair_metrics(interaction_rows, event_pairs)
    pair_lookup = _pair_lookup(event_pairs)
    for row in rows:
        pair = pair_lookup[str(row["pair_id"])]
        row.update({field: pair[field] for field in CONDITION_FIELDS})
        for field in (
            "target_delta_iqr",
            "other_parameter_l1_distance",
            "other_parameter_max_abs_difference",
            "nuisance_l1_distance",
            "nuisance_max_abs_difference",
        ):
            row[field] = pair[field]
    return [{field: row[field] for field in SINGLE_PAIR_METRIC_FIELDS} for row in rows]


def matching_balance_rows(
    event_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(event_pairs)
    rows: list[dict[str, Any]] = []
    for condition, group in frame.groupby("condition_id", sort=True):
        first = group.iloc[0]
        source = str(first["source_type"])
        target = str(first["target_parameter"])
        basis = SINGLE_PARAMETER_FEATURES_BY_SOURCE[source]
        columns: list[tuple[str, str, str]] = [
            (
                name,
                "target" if name == target else "other_gw",
                f"delta_{name}",
            )
            for name in basis
        ]
        columns.extend(
            (name, "nuisance", f"delta_{name}") for name in EVENT_NUISANCE_FEATURE_NAMES
        )
        columns.extend(
            [
                (
                    "other_parameter_l1_distance",
                    "aggregate",
                    "other_parameter_l1_distance",
                ),
                (
                    "other_parameter_max_abs_difference",
                    "aggregate",
                    "other_parameter_max_abs_difference",
                ),
                ("nuisance_l1_distance", "aggregate", "nuisance_l1_distance"),
                (
                    "nuisance_max_abs_difference",
                    "aggregate",
                    "nuisance_max_abs_difference",
                ),
            ]
        )
        for feature, role, column in columns:
            values = group[column].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "condition_id": str(condition),
                    "target_parameter": target,
                    "source_type": source,
                    "caliper_iqr": float(first["caliper_iqr"]),
                    "is_primary_caliper": bool(first["is_primary_caliper"]),
                    "n_pairs": len(group),
                    "feature": feature,
                    "role": role,
                    "minimum": float(np.min(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.quantile(values, 0.9)),
                    "maximum": float(np.max(values)),
                }
            )
    return rows


def _source_macro_value(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("source_type", sort=True)[column].mean().mean())


def resampling_summary(
    frame: pd.DataFrame,
    *,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, float]:
    """Pair bootstrap and one-sided sign-flip test for a source-macro endpoint."""
    if frame.empty:
        raise ValueError("Cannot summarize an empty pair frame")
    groups = {
        str(source): group.reset_index(drop=True)
        for source, group in frame.groupby("source_type", sort=True)
    }
    observed = float(
        np.mean([group["interaction"].mean() for group in groups.values()])
    )
    win_rate = float(
        np.mean([group["directional_win_rate"].mean() for group in groups.values()])
    )
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(bootstrap_samples), dtype=np.float64)
    permutation = np.empty(int(permutation_samples), dtype=np.float64)
    for start in range(0, int(bootstrap_samples), 250):
        stop = min(start + 250, int(bootstrap_samples))
        source_draws = []
        for group in groups.values():
            values = group["interaction"].to_numpy(dtype=np.float64)
            indices = rng.integers(0, values.size, size=(stop - start, values.size))
            source_draws.append(values[indices].mean(axis=1))
        bootstrap[start:stop] = np.mean(source_draws, axis=0)
    for start in range(0, int(permutation_samples), 250):
        stop = min(start + 250, int(permutation_samples))
        source_draws = []
        for group in groups.values():
            values = group["interaction"].to_numpy(dtype=np.float64)
            signs = rng.choice(
                np.asarray([-1.0, 1.0]), size=(stop - start, values.size)
            )
            source_draws.append((signs * values).mean(axis=1))
        permutation[start:stop] = np.mean(source_draws, axis=0)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    p_value = float(
        (1 + np.count_nonzero(permutation >= observed)) / (int(permutation_samples) + 1)
    )
    return {
        "mean_interaction": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "directional_win_rate": win_rate,
        "permutation_p_one_sided": p_value,
    }


def summarize_pair_metrics(
    pair_metric_rows: Sequence[Mapping[str, Any]],
    *,
    new_model_name: str,
    baseline_model_name: str,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(pair_metric_rows)
    rows: list[dict[str, Any]] = []
    for model_index, (model_name, model_frame) in enumerate(
        frame.groupby("model", sort=True)
    ):
        scopes = [("source_macro", model_frame)] + [
            (str(source), group)
            for source, group in model_frame.groupby("source_type", sort=True)
        ]
        for scope_index, (scope, scope_frame) in enumerate(scopes):
            stats_frame = scope_frame.copy()
            if scope != "source_macro":
                stats_frame["source_type"] = scope
            stats = resampling_summary(
                stats_frame,
                bootstrap_samples=bootstrap_samples,
                permutation_samples=permutation_samples,
                seed=int(seed) + 1009 * model_index + 101 * scope_index,
            )
            rows.append(
                {
                    "endpoint": "absolute_sensitivity",
                    "model": str(model_name),
                    "baseline_model": "",
                    "source": scope,
                    "n_pairs": int(scope_frame["pair_id"].nunique()),
                    **stats,
                    "bootstrap_samples": int(bootstrap_samples),
                    "permutation_samples": int(permutation_samples),
                }
            )

    new = frame[frame["model"] == str(new_model_name)].copy()
    baseline = frame[frame["model"] == str(baseline_model_name)].copy()
    merged = new.merge(
        baseline,
        on=["pair_id", "source_type", "gw_a", "gw_b"],
        suffixes=("_new", "_baseline"),
        validate="one_to_one",
    )
    if merged.empty or len(merged) != len(new) or len(merged) != len(baseline):
        raise ValueError("New/baseline models do not share an identical pair manifest")
    delta = pd.DataFrame(
        {
            "pair_id": merged["pair_id"],
            "source_type": merged["source_type"],
            "interaction": merged["interaction_new"] - merged["interaction_baseline"],
            "directional_win_rate": (
                merged["directional_win_rate_new"]
                - merged["directional_win_rate_baseline"]
            ),
        }
    )
    stats = resampling_summary(
        delta,
        bootstrap_samples=bootstrap_samples,
        permutation_samples=permutation_samples,
        seed=int(seed) + 7919,
    )
    rows.append(
        {
            "endpoint": "new_minus_baseline",
            "model": str(new_model_name),
            "baseline_model": str(baseline_model_name),
            "source": "source_macro",
            "n_pairs": int(delta["pair_id"].nunique()),
            **stats,
            "bootstrap_samples": int(bootstrap_samples),
            "permutation_samples": int(permutation_samples),
        }
    )
    return rows


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def _trimmed_mean(values: np.ndarray, fraction: float = 0.05) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    trim = int(np.floor(float(fraction) * values.size))
    selected = values[trim : values.size - trim] if trim else values
    return float(np.mean(selected))


def resample_column_summary(
    frame: pd.DataFrame,
    *,
    column: str,
    null_value: float,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap a source-macro mean and sign-flip its centered pair values."""
    if frame.empty:
        raise ValueError("Cannot summarize an empty pair frame")
    groups = {
        str(source): group.reset_index(drop=True)
        for source, group in frame.groupby("source_type", sort=True)
    }
    raw_values = {
        source: group[column].to_numpy(dtype=np.float64)
        for source, group in groups.items()
    }
    if any(
        values.size == 0 or not np.all(np.isfinite(values))
        for values in raw_values.values()
    ):
        raise ValueError(f"Non-finite or empty values for {column}")
    observed = float(np.mean([values.mean() for values in raw_values.values()]))
    centered_observed = observed - float(null_value)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(bootstrap_samples), dtype=np.float64)
    permutation = np.empty(int(permutation_samples), dtype=np.float64)
    for start in range(0, int(bootstrap_samples), 250):
        stop = min(start + 250, int(bootstrap_samples))
        source_draws = []
        for values in raw_values.values():
            indices = rng.integers(0, values.size, size=(stop - start, values.size))
            source_draws.append(values[indices].mean(axis=1))
        bootstrap[start:stop] = np.mean(source_draws, axis=0)
    for start in range(0, int(permutation_samples), 250):
        stop = min(start + 250, int(permutation_samples))
        source_draws = []
        for values in raw_values.values():
            centered = values - float(null_value)
            signs = rng.choice(
                np.asarray([-1.0, 1.0]), size=(stop - start, values.size)
            )
            source_draws.append((signs * centered).mean(axis=1))
        permutation[start:stop] = np.mean(source_draws, axis=0)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    medians = [float(np.median(values)) for values in raw_values.values()]
    trimmed = [_trimmed_mean(values) for values in raw_values.values()]
    positive = [
        float(np.mean(values > float(null_value))) for values in raw_values.values()
    ]
    return {
        "estimate": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "permutation_p_one_sided": float(
            (1 + np.count_nonzero(permutation >= centered_observed))
            / (int(permutation_samples) + 1)
        ),
        "median": float(np.mean(medians)),
        "trimmed_mean_5pct": float(np.mean(trimmed)),
        "positive_fraction": float(np.mean(positive)),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm step-down adjusted p-values in the original order."""
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    running = 0.0
    count = values.size
    for rank, index in enumerate(order):
        running = max(running, float((count - rank) * values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def _single_parameter_scopes(
    target: str, frame: pd.DataFrame
) -> list[tuple[str, pd.DataFrame]]:
    scopes = [
        (str(source), group.copy())
        for source, group in frame.groupby("source_type", sort=True)
    ]
    sources = {source for source, _ in scopes}
    if target in SINGLE_PARAMETER_COMMON_TARGETS and sources == {"bns", "nsbh"}:
        return [("source_macro", frame.copy()), *scopes]
    return scopes


def _is_primary_scope(target: str, source: str) -> bool:
    if target in SINGLE_PARAMETER_COMMON_TARGETS:
        return source == "source_macro"
    if target == "chi_eff":
        return source == "bns"
    if target == "primary_spin_z":
        return source == "nsbh"
    return False


def summarize_single_parameter_metrics(
    pair_metric_rows: Sequence[Mapping[str, Any]],
    *,
    new_model_name: str,
    baseline_model_name: str,
    primary_caliper_iqr: float,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(pair_metric_rows)
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby(["target_parameter", "caliper_iqr"], sort=True)
    for (target_value, caliper_value), target_frame in grouped:
        target = str(target_value)
        caliper = float(caliper_value)
        is_primary = bool(
            np.isclose(caliper, primary_caliper_iqr, atol=1e-12, rtol=0.0)
        )
        for model_name, model_frame in target_frame.groupby("model", sort=True):
            for scope, scope_frame in _single_parameter_scopes(target, model_frame):
                for metric, null_value in (
                    ("interaction", 0.0),
                    ("directional_win_rate", 0.5),
                ):
                    stats = resample_column_summary(
                        scope_frame,
                        column=metric,
                        null_value=null_value,
                        bootstrap_samples=bootstrap_samples,
                        permutation_samples=permutation_samples,
                        seed=_stable_seed(
                            seed, "absolute", target, caliper, model_name, scope, metric
                        ),
                    )
                    family = ""
                    if (
                        is_primary
                        and str(model_name) == str(new_model_name)
                        and _is_primary_scope(target, scope)
                    ):
                        family = f"mixed_absolute_{metric}"
                    rows.append(
                        {
                            "endpoint": "absolute_sensitivity",
                            "metric": metric,
                            "family": family,
                            "target_parameter": target,
                            "caliper_iqr": caliper,
                            "is_primary_caliper": is_primary,
                            "model": str(model_name),
                            "baseline_model": "",
                            "source": scope,
                            "n_pairs": int(scope_frame["pair_id"].nunique()),
                            **stats,
                            "null_value": null_value,
                            "holm_adjusted_p": "",
                            "holm_reject_0_05": "",
                            "bootstrap_samples": int(bootstrap_samples),
                            "permutation_samples": int(permutation_samples),
                        }
                    )

        new = target_frame[target_frame["model"] == str(new_model_name)].copy()
        baseline = target_frame[
            target_frame["model"] == str(baseline_model_name)
        ].copy()
        merged = new.merge(
            baseline,
            on=[
                "pair_id",
                "source_type",
                "condition_id",
                "target_parameter",
                "caliper_iqr",
            ],
            suffixes=("_new", "_baseline"),
            validate="one_to_one",
        )
        if merged.empty or len(merged) != len(new) or len(merged) != len(baseline):
            raise ValueError(
                f"Models do not share an identical pair manifest for {target}/{caliper:g}"
            )
        delta = pd.DataFrame(
            {
                "pair_id": merged["pair_id"],
                "source_type": merged["source_type"],
                "interaction_delta": (
                    merged["interaction_new"] - merged["interaction_baseline"]
                ),
                "directional_win_rate_delta": (
                    merged["directional_win_rate_new"]
                    - merged["directional_win_rate_baseline"]
                ),
            }
        )
        for scope, scope_frame in _single_parameter_scopes(target, delta):
            for metric in ("interaction_delta", "directional_win_rate_delta"):
                stats = resample_column_summary(
                    scope_frame,
                    column=metric,
                    null_value=0.0,
                    bootstrap_samples=bootstrap_samples,
                    permutation_samples=permutation_samples,
                    seed=_stable_seed(
                        seed, "comparison", target, caliper, scope, metric
                    ),
                )
                family = ""
                if is_primary and _is_primary_scope(target, scope):
                    family = f"mixed_minus_default_{metric}"
                rows.append(
                    {
                        "endpoint": "new_minus_baseline",
                        "metric": metric,
                        "family": family,
                        "target_parameter": target,
                        "caliper_iqr": caliper,
                        "is_primary_caliper": is_primary,
                        "model": str(new_model_name),
                        "baseline_model": str(baseline_model_name),
                        "source": scope,
                        "n_pairs": int(scope_frame["pair_id"].nunique()),
                        **stats,
                        "null_value": 0.0,
                        "holm_adjusted_p": "",
                        "holm_reject_0_05": "",
                        "bootstrap_samples": int(bootstrap_samples),
                        "permutation_samples": int(permutation_samples),
                    }
                )

    summary = pd.DataFrame(rows)
    families = sorted(value for value in summary["family"].unique() if value)
    for family in families:
        indices = summary.index[summary["family"] == family]
        if len(indices) != 6:
            raise ValueError(
                f"Primary Holm family {family} has {len(indices)} endpoints; expected 6"
            )
        adjusted = holm_adjust(
            summary.loc[indices, "permutation_p_one_sided"].to_numpy(float)
        )
        summary.loc[indices, "holm_adjusted_p"] = adjusted
        summary.loc[indices, "holm_reject_0_05"] = adjusted <= 0.05
    return summary[list(SINGLE_SUMMARY_FIELDS)].to_dict("records")


def _spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average")
    y_rank = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average")
    if float(x_rank.std()) <= 1e-12 or float(y_rank.std()) <= 1e-12:
        return 0.0
    return float(x_rank.corr(y_rank))


def single_parameter_dose_response(
    pair_metric_rows: Sequence[Mapping[str, Any]],
    *,
    primary_caliper_iqr: float,
    n_bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.DataFrame(pair_metric_rows)
    frame = frame[
        np.isclose(
            frame["caliper_iqr"].to_numpy(float),
            float(primary_caliper_iqr),
            atol=1e-12,
            rtol=0.0,
        )
    ].copy()
    labelled_parts: list[pd.DataFrame] = []
    for (target, source), group in frame.groupby(
        ["target_parameter", "source_type"], sort=True
    ):
        unique = group.drop_duplicates("pair_id")[
            ["pair_id", "target_delta_iqr"]
        ].sort_values(["target_delta_iqr", "pair_id"], kind="stable")
        assignments: dict[str, str] = {}
        for bin_index, indices in enumerate(
            np.array_split(np.arange(len(unique)), int(n_bins)), start=1
        ):
            label = f"T{bin_index}_{'low' if bin_index == 1 else 'high' if bin_index == n_bins else 'mid'}"
            for pair_id in unique.iloc[indices]["pair_id"].tolist():
                assignments[str(pair_id)] = label
        part = group.copy()
        part["dose_bin"] = part["pair_id"].map(assignments)
        if part["dose_bin"].isna().any():
            raise AssertionError(f"Missing dose assignment for {target}/{source}")
        labelled_parts.append(part)
    labelled = pd.concat(labelled_parts, ignore_index=True)
    rows: list[dict[str, Any]] = []
    trends: list[dict[str, Any]] = []
    for (target, source, model), group in labelled.groupby(
        ["target_parameter", "source_type", "model"], sort=True
    ):
        trends.append(
            {
                "target_parameter": str(target),
                "source": str(source),
                "model": str(model),
                "n_pairs": int(group["pair_id"].nunique()),
                "spearman_target_delta_interaction": _spearman_rank_correlation(
                    group["target_delta_iqr"].to_numpy(float),
                    group["interaction"].to_numpy(float),
                ),
            }
        )
        for dose_bin, dose in group.groupby("dose_bin", sort=True):
            rows.append(
                {
                    "target_parameter": str(target),
                    "source": str(source),
                    "model": str(model),
                    "dose_bin": str(dose_bin),
                    "n_pairs": int(dose["pair_id"].nunique()),
                    "target_delta_min": float(dose["target_delta_iqr"].min()),
                    "target_delta_median": float(dose["target_delta_iqr"].median()),
                    "target_delta_max": float(dose["target_delta_iqr"].max()),
                    "interaction_mean": float(dose["interaction"].mean()),
                    "directional_win_rate": float(dose["directional_win_rate"].mean()),
                }
            )
    source_rows = pd.DataFrame(rows)
    source_trends = pd.DataFrame(trends)
    for target in SINGLE_PARAMETER_COMMON_TARGETS:
        target_rows = source_rows[source_rows["target_parameter"] == target]
        for (model, dose_bin), group in target_rows.groupby(
            ["model", "dose_bin"], sort=True
        ):
            if set(group["source"]) != {"bns", "nsbh"}:
                raise ValueError(f"Incomplete source-macro dose rows for {target}")
            rows.append(
                {
                    "target_parameter": target,
                    "source": "source_macro",
                    "model": str(model),
                    "dose_bin": str(dose_bin),
                    "n_pairs": int(group["n_pairs"].sum()),
                    "target_delta_min": float(group["target_delta_min"].min()),
                    "target_delta_median": float(group["target_delta_median"].mean()),
                    "target_delta_max": float(group["target_delta_max"].max()),
                    "interaction_mean": float(group["interaction_mean"].mean()),
                    "directional_win_rate": float(group["directional_win_rate"].mean()),
                }
            )
        target_trends = source_trends[source_trends["target_parameter"] == target]
        for model, group in target_trends.groupby("model", sort=True):
            if set(group["source"]) != {"bns", "nsbh"}:
                raise ValueError(f"Incomplete source-macro trend rows for {target}")
            trends.append(
                {
                    "target_parameter": target,
                    "source": "source_macro",
                    "model": str(model),
                    "n_pairs": int(group["n_pairs"].sum()),
                    "spearman_target_delta_interaction": float(
                        group["spearman_target_delta_interaction"].mean()
                    ),
                }
            )
    return rows, trends


def physical_distance_quartiles(
    pair_metric_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(pair_metric_rows)
    labelled_parts = []
    labels = ["Q1", "Q2", "Q3", "Q4"]
    for source, source_frame in frame.groupby("source_type", sort=True):
        distances = source_frame.drop_duplicates("pair_id").set_index("pair_id")[
            "physical_distance"
        ]
        quartile = pd.qcut(distances, q=4, labels=labels, duplicates="raise")
        part = source_frame.copy()
        part["physical_distance_quartile"] = part["pair_id"].map(quartile)
        labelled_parts.append(part)
    labelled = pd.concat(labelled_parts, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for (model, source, quartile), group in labelled.groupby(
        ["model", "source_type", "physical_distance_quartile"],
        observed=True,
        sort=True,
    ):
        rows.append(
            {
                "model": str(model),
                "source": str(source),
                "physical_distance_quartile": str(quartile),
                "n_pairs": int(group["pair_id"].nunique()),
                "physical_distance_mean": float(group["physical_distance"].mean()),
                "interaction_mean": float(group["interaction"].mean()),
                "directional_win_rate": float(group["directional_win_rate"].mean()),
            }
        )
    source_rows = pd.DataFrame(rows)
    for (model, quartile), group in source_rows.groupby(
        ["model", "physical_distance_quartile"], sort=True
    ):
        rows.append(
            {
                "model": str(model),
                "source": "source_macro",
                "physical_distance_quartile": str(quartile),
                "n_pairs": int(group["n_pairs"].sum()),
                "physical_distance_mean": float(group["physical_distance_mean"].mean()),
                "interaction_mean": float(group["interaction_mean"].mean()),
                "directional_win_rate": float(group["directional_win_rate"].mean()),
            }
        )
    return rows


def _plot_results(
    summary_rows: Sequence[Mapping[str, Any]],
    quartile_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.DataFrame(summary_rows)
    absolute = summary[
        (summary["endpoint"] == "absolute_sensitivity")
        & (summary["source"] == "source_macro")
    ].copy()
    absolute = absolute.sort_values("mean_interaction")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    positions = np.arange(len(absolute))
    means = absolute["mean_interaction"].to_numpy(float)
    lower = means - absolute["ci95_low"].to_numpy(float)
    upper = absolute["ci95_high"].to_numpy(float) - means
    ax.errorbar(means, positions, xerr=np.vstack([lower, upper]), fmt="o", capsize=4)
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(positions, absolute["model"].tolist())
    ax.set_xlabel("Crossed-pair interaction (fusion-logit margin)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    forest_png = output_dir / "pairing_interaction_forest.png"
    forest_pdf = output_dir / "pairing_interaction_forest.pdf"
    fig.savefig(forest_png, dpi=300, bbox_inches="tight")
    fig.savefig(forest_pdf, bbox_inches="tight")
    plt.close(fig)

    quartiles = pd.DataFrame(quartile_rows)
    quartiles = quartiles[quartiles["source"] == "source_macro"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    order = ["Q1", "Q2", "Q3", "Q4"]
    for model, group in quartiles.groupby("model", sort=True):
        values = group.set_index("physical_distance_quartile").reindex(order)
        ax.plot(
            order,
            values["interaction_mean"].to_numpy(float),
            marker="o",
            label=str(model),
        )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Within-source GW physical-distance quartile")
    ax.set_ylabel("Mean crossed-pair interaction")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    trend_png = output_dir / "pairing_interaction_by_physical_distance.png"
    trend_pdf = output_dir / "pairing_interaction_by_physical_distance.pdf"
    fig.savefig(trend_png, dpi=300, bbox_inches="tight")
    fig.savefig(trend_pdf, bbox_inches="tight")
    plt.close(fig)
    return [path.name for path in (forest_png, forest_pdf, trend_png, trend_pdf)]


def _single_target_order() -> list[str]:
    return [
        "chirp_mass_detector",
        "mass_ratio",
        "chi_eff",
        "primary_spin_z",
        "abs_costheta",
        "log10_distance_gpc",
    ]


def _single_primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    mask = frame.apply(
        lambda row: bool(row["is_primary_caliper"])
        and _is_primary_scope(str(row["target_parameter"]), str(row["source"])),
        axis=1,
    )
    return frame[mask].copy()


def _plot_single_parameter_results(
    summary_rows: Sequence[Mapping[str, Any]],
    dose_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.DataFrame(summary_rows)
    primary = _single_primary_rows(summary)
    target_order = _single_target_order()
    model_order = ["Mixed Gallery v1", "Default MAGIKS", "Optical-only"]
    colors = {model: f"C{index}" for index, model in enumerate(model_order)}
    artifacts: list[Path] = []

    for metric, reference, xlabel, stem in (
        (
            "interaction",
            0.0,
            "Crossed-pair interaction (fusion-logit margin)",
            "single_parameter_interaction_forest",
        ),
        (
            "directional_win_rate",
            0.5,
            "Directional-win rate",
            "single_parameter_directional_win_forest",
        ),
    ):
        selected = primary[
            (primary["endpoint"] == "absolute_sensitivity")
            & (primary["metric"] == metric)
        ]
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        y_base = np.arange(len(target_order), dtype=float)
        for model_index, model in enumerate(model_order):
            model_rows = (
                selected[selected["model"] == model]
                .set_index("target_parameter")
                .reindex(target_order)
            )
            estimates = model_rows["estimate"].to_numpy(float)
            lower = estimates - model_rows["ci95_low"].to_numpy(float)
            upper = model_rows["ci95_high"].to_numpy(float) - estimates
            offset = (model_index - 1) * 0.18
            ax.errorbar(
                estimates,
                y_base + offset,
                xerr=np.vstack([lower, upper]),
                fmt="o",
                capsize=3,
                color=colors[model],
                label=model,
            )
        ax.axvline(reference, color="black", linewidth=1, linestyle="--")
        ax.set_yticks(y_base, target_order)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
            artifacts.append(path)
        plt.close(fig)

    comparison = summary[
        (summary["endpoint"] == "new_minus_baseline")
        & (summary["metric"] == "interaction_delta")
    ].copy()
    comparison = comparison[
        comparison.apply(
            lambda row: _is_primary_scope(
                str(row["target_parameter"]), str(row["source"])
            ),
            axis=1,
        )
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for target in target_order:
        group = comparison[comparison["target_parameter"] == target].sort_values(
            "caliper_iqr"
        )
        ax.plot(
            group["caliper_iqr"],
            group["estimate"],
            marker="o",
            label=target,
        )
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Maximum non-target GW difference (IQR)")
    ax.set_ylabel("Mixed Gallery v1 - Default MAGIKS interaction")
    ax.set_xticks(sorted(comparison["caliper_iqr"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"single_parameter_caliper_robustness.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        artifacts.append(path)
    plt.close(fig)

    dose = pd.DataFrame(dose_rows)
    dose = dose[
        dose.apply(
            lambda row: _is_primary_scope(
                str(row["target_parameter"]), str(row["source"])
            ),
            axis=1,
        )
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), sharex=True)
    for ax, target in zip(axes.flat, target_order):
        selected = dose[dose["target_parameter"] == target]
        for model in model_order:
            group = selected[selected["model"] == model].sort_values("dose_bin")
            ax.plot(
                group["dose_bin"],
                group["interaction_mean"],
                marker="o",
                color=colors[model],
                label=model,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(target)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Target-separation rank tertile")
    fig.supylabel("Mean crossed-pair interaction")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output_dir / f"single_parameter_dose_response.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        artifacts.append(path)
    plt.close(fig)
    return [path.name for path in artifacts]


def _script_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def run(
    config: str | Path,
    *,
    validate_only: bool = False,
    preflight_pairs: bool = False,
) -> None:
    config_path = Path(config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if str(raw.get("primary_metric", "")) == "directional_win_rate":
        from scripts.eval.eval_gw_kn_directional_bridge import (
            run_directional_bridge,
        )

        run_directional_bridge(
            config_path,
            raw,
            validate_only=validate_only,
            preflight_pairs=preflight_pairs,
        )
        return
    cfg = normalise_config(raw, config_path)
    if not Path(cfg["test_data_path"]).is_file():
        raise FileNotFoundError(cfg["test_data_path"])
    model_specs = base._build_model_specs(dict(raw), config_path.parent)
    if validate_only:
        print(json.dumps({"status": "valid", "config": str(config_path)}, indent=2))
        return

    _seed_all(cfg["seed"])

    print("Reading GW identities and optical sampling metadata...")
    metadata = _read_metadata(cfg["test_data_path"], cfg["comparison_window"])
    if cfg["pairing_mode"] == "single_parameter":
        event_pairs, curve_pairs, pairing_audit = _prepare_single_parameter_pairs(
            metadata, cfg
        )
    else:
        event_pairs, curve_pairs, pairing_audit = _prepare_pairs(metadata, cfg)
    pair_payload = {"event_pairs": event_pairs, "curve_pairs": curve_pairs}
    pair_digest = stable_digest(pair_payload)
    count_key = (
        "condition_id" if cfg["pairing_mode"] == "single_parameter" else "source_type"
    )
    pair_counts = {
        label: sum(1 for pair in event_pairs if str(pair[count_key]) == label)
        for label in sorted({str(pair[count_key]) for pair in event_pairs})
    }
    print(f"Built {len(event_pairs)} disjoint event pairs: {pair_counts}")
    print(f"Pair manifest SHA256: {pair_digest}")
    if preflight_pairs:
        print(
            json.dumps(
                {
                    "status": "pairing_preflight_valid",
                    "config": str(config_path),
                    "pairing_mode": cfg["pairing_mode"],
                    "n_event_pairs": len(event_pairs),
                    "n_curve_pairs": len(curve_pairs),
                    "pair_counts": pair_counts,
                    "pair_manifest_sha256": pair_digest,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    requested_device = cfg["device"]
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(requested_device)
    amp_dtype, amp_enabled = base._resolve_eval_amp(cfg["amp_dtype"], device)

    output_dir = Path(cfg["output_dir"])
    manifest = prepare_output_directory(
        output_dir,
        manifest={
            "experiment_id": str(
                cfg.get("experiment_id", "gw_kn_pairing_sensitivity_v1")
            ),
            "config_path": str(config_path),
            "requested_test_data_path": cfg["requested_test_data_path"],
            "resolved_test_data_path": cfg["test_data_path"],
            "pair_manifest_sha256": pair_digest,
            "script_sha256": _script_digest(),
            "seed": cfg["seed"],
            "comparison_window": cfg["comparison_window"],
            "brightness_policy": "native_unmodified_not_matched_not_ablated",
            "anchor_policy": "counterbalanced_pair_shared_coordinate_and_dt",
            "pairing_mode": cfg["pairing_mode"],
        },
        resume=False,
    )
    is_single_parameter = cfg["pairing_mode"] == "single_parameter"
    event_fields = SINGLE_PAIR_FIELDS if is_single_parameter else PAIR_FIELDS
    curve_fields = (
        SINGLE_CURVE_PAIR_FIELDS if is_single_parameter else CURVE_PAIR_FIELDS
    )
    write_csv_atomic(output_dir / "event_pairs.csv", event_pairs, event_fields)
    write_csv_atomic(output_dir / "curve_pairs.csv", curve_pairs, curve_fields)
    balance_rows: list[dict[str, Any]] = []
    if is_single_parameter:
        balance_rows = matching_balance_rows(event_pairs)
        write_csv_atomic(
            output_dir / "matching_balance.csv", balance_rows, MATCHING_BALANCE_FIELDS
        )
    write_json_atomic(output_dir / "pairing_audit.json", pairing_audit)

    selected_optical = np.unique(
        np.asarray(
            [
                int(value)
                for curve_pair in curve_pairs
                for value in (
                    curve_pair["optical_index_a"],
                    curve_pair["optical_index_b"],
                )
            ],
            dtype=np.int64,
        )
    )
    bank, compact_index = base.load_selected_positive_bank(
        cfg["test_data_path"],
        selected_optical,
        runtime_input_window_start=float(cfg["comparison_window"][0]),
        runtime_input_window_end=float(cfg["comparison_window"][1]),
    )
    if not torch.equal(
        bank["source_optical_indices"], torch.from_numpy(selected_optical)
    ):
        raise AssertionError("Selected optical bank identity changed unexpectedly")

    all_score_rows: list[dict[str, Any]] = []
    all_interaction_rows: list[dict[str, Any]] = []
    model_info: list[dict[str, Any]] = []
    ordered_specs = sorted(
        model_specs,
        key=lambda spec: 1 if spec["type"] == "optical" else 0,
    )
    for spec in ordered_specs:
        print(f"\nEvaluating {spec['name']} ({spec['type']})")
        kwargs = {
            "model_spec": spec,
            "cfg": cfg,
            "bank": bank,
            "compact_index": compact_index,
            "event_pairs": event_pairs,
            "curve_pairs": curve_pairs,
            "metadata": metadata,
            "device": device,
            "amp_dtype": amp_dtype,
            "amp_enabled": amp_enabled,
        }
        if spec["type"] == "multimodal":
            scores, interactions, info = _score_multimodal(**kwargs)
        elif spec["type"] == "optical":
            scores, interactions, info = _score_optical_null(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {spec['type']}")
        all_score_rows.extend(scores)
        all_interaction_rows.extend(interactions)
        model_info.append(info)

    optical_interactions = [
        abs(float(row["interaction"]))
        for row in all_interaction_rows
        if row["model"] == cfg["optical_null_model_name"]
    ]
    if (
        not optical_interactions
        or max(optical_interactions) > cfg["optical_null_tolerance"]
    ):
        raise AssertionError(
            "Optical-only crossed interaction is not a numerical zero: "
            f"max={max(optical_interactions, default=float('nan'))}"
        )

    dose_rows: list[dict[str, Any]] = []
    trend_rows: list[dict[str, Any]] = []
    if is_single_parameter:
        pair_metric_rows = aggregate_single_parameter_pair_metrics(
            all_interaction_rows, event_pairs
        )
        summary_rows = summarize_single_parameter_metrics(
            pair_metric_rows,
            new_model_name=cfg["new_model_name"],
            baseline_model_name=cfg["baseline_model_name"],
            primary_caliper_iqr=cfg["primary_other_parameter_caliper_iqr"],
            bootstrap_samples=cfg["bootstrap_samples"],
            permutation_samples=cfg["permutation_samples"],
            seed=cfg["seed"],
        )
        dose_rows, trend_rows = single_parameter_dose_response(
            pair_metric_rows,
            primary_caliper_iqr=cfg["primary_other_parameter_caliper_iqr"],
            n_bins=cfg["dose_response_bins"],
        )
        robustness_rows = [
            row
            for row in summary_rows
            if _is_primary_scope(row["target_parameter"], row["source"])
        ]
        write_csv_atomic(
            output_dir / "single_parameter_summary.csv",
            summary_rows,
            SINGLE_SUMMARY_FIELDS,
        )
        write_csv_atomic(
            output_dir / "caliper_robustness.csv",
            robustness_rows,
            SINGLE_SUMMARY_FIELDS,
        )
        write_csv_atomic(
            output_dir / "dose_response.csv", dose_rows, DOSE_RESPONSE_FIELDS
        )
        write_csv_atomic(output_dir / "parameter_trends.csv", trend_rows, TREND_FIELDS)
        plot_artifacts = _plot_single_parameter_results(
            summary_rows, dose_rows, output_dir
        )
        score_fields = SINGLE_SCORE_FIELDS
        interaction_fields = SINGLE_ANCHOR_INTERACTION_FIELDS
        pair_metric_fields = SINGLE_PAIR_METRIC_FIELDS
    else:
        pair_metric_rows = aggregate_pair_metrics(all_interaction_rows, event_pairs)
        summary_rows = summarize_pair_metrics(
            pair_metric_rows,
            new_model_name=cfg["new_model_name"],
            baseline_model_name=cfg["baseline_model_name"],
            bootstrap_samples=cfg["bootstrap_samples"],
            permutation_samples=cfg["permutation_samples"],
            seed=cfg["seed"],
        )
        quartile_rows = physical_distance_quartiles(pair_metric_rows)
        write_csv_atomic(
            output_dir / "sensitivity_summary.csv", summary_rows, SUMMARY_FIELDS
        )
        write_csv_atomic(
            output_dir / "physical_distance_quartiles.csv",
            quartile_rows,
            QUARTILE_FIELDS,
        )
        plot_artifacts = _plot_results(summary_rows, quartile_rows, output_dir)
        score_fields = SCORE_FIELDS
        interaction_fields = ANCHOR_INTERACTION_FIELDS
        pair_metric_fields = PAIR_METRIC_FIELDS

    writer = AtomicGzipCsvWriter(output_dir / "candidate_scores.csv.gz", score_fields)
    try:
        writer.writerows(all_score_rows)
        writer.commit()
    finally:
        writer.close()
    write_csv_atomic(
        output_dir / "anchor_interactions.csv",
        all_interaction_rows,
        interaction_fields,
    )
    write_csv_atomic(
        output_dir / "pair_metrics.csv", pair_metric_rows, pair_metric_fields
    )
    result = {
        "experiment_id": str(cfg.get("experiment_id", "gw_kn_pairing_sensitivity_v1")),
        "pairing_mode": cfg["pairing_mode"],
        "pair_manifest_sha256": pair_digest,
        "pair_counts": pair_counts,
        "n_curve_pairs": len(curve_pairs),
        "models": model_info,
        "summary": summary_rows,
        "protocol": {
            "score": "class_1_logit_minus_class_0_logit",
            "interaction": "0.5*((s_aa-s_ab)+(s_bb-s_ba))",
            "source_aggregation": "equal_weight_bns_nsbh",
            "brightness": (
                "Native values and errors are retained. Brightness is neither "
                "normalized, matched, nor independently ablated."
            ),
            "candidate_metadata": (
                "A and B coordinates/delays are used as counterbalanced anchors; "
                "each anchor is shared by all four crossed cells."
            ),
            "interpretation": (
                "Single-parameter mode conditions on the other recovered GW "
                "coordinates without altering native brightness, color, or temporal "
                "evolution. It measures conditional association sensitivity rather "
                "than a strict causal effect."
                if is_single_parameter
                else "Sensitivity includes joint GW relationships with brightness, "
                "color, and temporal evolution; it does not isolate intrinsic "
                "morphology from distance-brightness compatibility."
            ),
        },
        "config": cfg,
    }
    common_artifacts = [
        "event_pairs.csv",
        "curve_pairs.csv",
        "pairing_audit.json",
        "candidate_scores.csv.gz",
        "anchor_interactions.csv",
        "pair_metrics.csv",
    ]
    if is_single_parameter:
        result["dose_response"] = dose_rows
        result["parameter_trends"] = trend_rows
        summary_filename = "single_parameter_sensitivity_summary.json"
        artifacts = [
            *common_artifacts,
            "matching_balance.csv",
            "single_parameter_summary.csv",
            "caliper_robustness.csv",
            "dose_response.csv",
            "parameter_trends.csv",
            summary_filename,
            *plot_artifacts,
        ]
    else:
        summary_filename = "pairing_sensitivity_summary.json"
        artifacts = [
            *common_artifacts,
            "sensitivity_summary.csv",
            "physical_distance_quartiles.csv",
            summary_filename,
            *plot_artifacts,
        ]
    write_json_atomic(output_dir / summary_filename, result)
    mark_run_success(output_dir, manifest, artifacts)
    print(f"GW--KN {cfg['pairing_mode']} pairing sensitivity complete: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate crossed-pair GW--KN physical association sensitivity."
    )
    parser.add_argument("--config", required=True, help="Experiment JSON config")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths and model specifications without creating outputs",
    )
    parser.add_argument(
        "--preflight-pairs",
        action="store_true",
        help="Build and audit pair manifests without loading models or writing outputs",
    )
    args = parser.parse_args()
    if args.validate_only and args.preflight_pairs:
        parser.error("--validate-only and --preflight-pairs are mutually exclusive")
    run(
        args.config,
        validate_only=args.validate_only,
        preflight_pairs=args.preflight_pairs,
    )


if __name__ == "__main__":
    main()
