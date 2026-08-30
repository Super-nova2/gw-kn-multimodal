"""Directional-win-only single-parameter comparison with a physical bridge."""

from __future__ import annotations

import gc
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from scripts.eval import eval_gw_kn_pairing_sensitivity as v1
    from scripts.eval import eval_retrieval_comparison as base
    from scripts.eval.eval_run_io import (
        AtomicGzipCsvWriter,
        mark_run_success,
        prepare_output_directory,
        stable_digest,
        write_csv_atomic,
        write_json_atomic,
    )
    from scripts.eval.physics_ejecta_bridge import (
        PhysicsEjectaBridge,
        bhattacharyya_score,
        extract_lightcurve_features,
    )
except ModuleNotFoundError:
    from Model.scripts.eval import eval_gw_kn_pairing_sensitivity as v1
    from Model.scripts.eval import eval_retrieval_comparison as base
    from Model.scripts.eval.eval_run_io import (
        AtomicGzipCsvWriter,
        mark_run_success,
        prepare_output_directory,
        stable_digest,
        write_csv_atomic,
        write_json_atomic,
    )
    from Model.scripts.eval.physics_ejecta_bridge import (
        PhysicsEjectaBridge,
        bhattacharyya_score,
        extract_lightcurve_features,
    )

BRIDGE_NAME = "Physics Ejecta Bridge"
GW_BLIND_NAME = "GW-blind Control"
EXPECTED_PAIR_DIGEST = (
    "893bd33191190756b5987f1f8d9b0d54d6388777fced96a5f616fddb2533f1ee"
)

PLOT_PARAMETER_LABELS = {
    "chirp_mass_detector": (
        r"$\mathcal{M}_{c}^{\mathrm{det}}$ (detector-frame chirp mass)"
    ),
    "mass_ratio": r"$q$ (mass ratio)",
    "chi_eff": r"$\chi_{\mathrm{eff}}$ (effective spin)",
    "primary_spin_z": r"$\chi_{1z}$ (primary spin)",
    "abs_costheta": r"$|\cos\theta_{\mathrm{JN}}|$ (inclination)",
    "log10_distance_gpc": (r"$\log_{10}(d_{L}/\mathrm{Gpc})$ (luminosity distance)"),
}
DOSE_BIN_LABELS = {
    "T1_low": "Low",
    "T2_mid": "Medium",
    "T3_high": "High",
}
PLOT_LATEX_PREAMBLE = (
    r"\usepackage{txfonts}" r"\usepackage{fontspec}" r"\setmainfont{Times New Roman}"
)
PLOT_FONT_SCALE = 2.0
PLOT_AXIS_SCALE = 1.5
PLOT_LEGEND_SCALE = 1.5

DECISION_FIELDS = (
    "model",
    "model_type",
    *v1.CONDITION_FIELDS,
    "pair_id",
    "curve_pair_id",
    "source_type",
    "anchor",
    "score_aa",
    "score_ab",
    "score_ba",
    "score_bb",
    "margin_a",
    "margin_b",
    "win_a",
    "win_b",
    "directional_win_rate",
)
PAIR_DWR_FIELDS = (
    "model",
    "model_type",
    *v1.CONDITION_FIELDS,
    "pair_id",
    "source_type",
    "gw_a",
    "gw_b",
    "n_curve_pairs",
    "directional_win_rate",
    "target_delta_iqr",
    "other_parameter_l1_distance",
    "other_parameter_max_abs_difference",
    "nuisance_l1_distance",
    "nuisance_max_abs_difference",
)
DWR_SUMMARY_FIELDS = (
    "endpoint",
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
    "permutation_p",
    "holm_adjusted_p",
    "holm_reject_0_05",
    "median",
    "trimmed_mean_5pct",
    "fraction_above_0_5",
    "bootstrap_samples",
    "permutation_samples",
)
DOSE_FIELDS = (
    "target_parameter",
    "source",
    "model",
    "dose_bin",
    "n_pairs",
    "target_delta_min",
    "target_delta_median",
    "target_delta_max",
    "directional_win_rate",
)
TREND_FIELDS = (
    "target_parameter",
    "source",
    "model",
    "n_pairs",
    "spearman_target_delta_directional_win_rate",
)
NEIGHBOR_FIELDS = (
    "side",
    "source_type",
    "query_id",
    "rank",
    "reference_parent_gw_id",
    "distance",
    "weight",
    "cadence_fallback_level",
)


def _resolve_path(directory: Path, value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    return str(
        (directory / path).resolve() if not path.is_absolute() else path.resolve()
    )


def normalise_v2_config(
    raw: Mapping[str, Any], config_path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if str(raw.get("primary_metric", "")) != "directional_win_rate":
        raise ValueError("v2 primary_metric must be directional_win_rate")
    if bool(raw.get("compare_interaction", False)):
        raise ValueError("v2 does not support interaction comparison")
    models = [dict(model) for model in raw.get("models", [])]
    model_names = [str(model.get("name")) for model in models]
    names = set(model_names)
    if len(names) != len(model_names):
        raise ValueError("v2 model names must be unique")
    required = {
        str(raw.get("new_model_name", "Mixed Gallery v1")),
        str(raw.get("baseline_model_name", "Default MAGIKS")),
        BRIDGE_NAME,
        GW_BLIND_NAME,
    }
    if not required.issubset(names):
        raise ValueError(f"v2 models must contain at least {sorted(required)}")
    bridge_specs = [
        model for model in models if model.get("type") == "physics_ejecta_bridge"
    ]
    if len(bridge_specs) != 1 or str(bridge_specs[0]["name"]) != BRIDGE_NAME:
        raise ValueError("Physics Ejecta Bridge must use type physics_ejecta_bridge")
    bridge = bridge_specs[0]
    bridge["artifact_path"] = _resolve_path(
        config_path.parent, bridge.get("artifact_path")
    )
    if not bridge["artifact_path"]:
        raise ValueError("Physics Ejecta Bridge requires artifact_path")
    neural_models = [
        model for model in models if model.get("type") != "physics_ejecta_bridge"
    ]
    legacy_raw = dict(raw)
    legacy_raw["optical_null_model_name"] = GW_BLIND_NAME
    legacy_raw["new_model_name"] = str(raw.get("new_model_name", "Mixed Gallery v1"))
    legacy_raw["baseline_model_name"] = str(
        raw.get("baseline_model_name", "Default MAGIKS")
    )
    legacy_validation_names = {
        legacy_raw["new_model_name"],
        legacy_raw["baseline_model_name"],
        GW_BLIND_NAME,
    }
    legacy_raw["models"] = [
        model
        for model in neural_models
        if str(model.get("name")) in legacy_validation_names
    ]
    specs_raw = dict(raw)
    specs_raw["models"] = neural_models
    specs_raw["optical_null_model_name"] = GW_BLIND_NAME
    cfg = v1.normalise_config(legacy_raw, config_path)
    cfg["primary_metric"] = "directional_win_rate"
    cfg["compare_interaction"] = False
    cfg["bridge_model_name"] = BRIDGE_NAME
    cfg["gw_blind_model_name"] = GW_BLIND_NAME
    cfg["bridge_artifact_path"] = bridge["artifact_path"]
    cfg["expected_pair_manifest_sha256"] = str(
        raw.get("expected_pair_manifest_sha256", EXPECTED_PAIR_DIGEST)
    )
    default_comparisons = [
        {
            "model": cfg["new_model_name"],
            "baseline_model": BRIDGE_NAME,
            "family": "mixed_minus_bridge_directional_win_rate",
            "label": "Mixed - Bridge",
        },
        {
            "model": cfg["baseline_model_name"],
            "baseline_model": BRIDGE_NAME,
            "family": "default_minus_bridge_directional_win_rate",
            "label": "Default - Bridge",
        },
        {
            "model": cfg["new_model_name"],
            "baseline_model": cfg["baseline_model_name"],
            "family": "mixed_minus_default_directional_win_rate",
            "label": "Mixed - Default",
        },
    ]
    comparisons = [
        {
            "model": str(item["model"]),
            "baseline_model": str(item["baseline_model"]),
            "family": str(item["family"]),
            "label": str(
                item.get("label", f"{item['model']} - {item['baseline_model']}")
            ),
        }
        for item in raw.get("pairwise_comparisons", default_comparisons)
    ]
    if not comparisons or len({item["family"] for item in comparisons}) != len(
        comparisons
    ):
        raise ValueError("v2 pairwise comparison families must be non-empty and unique")
    for item in comparisons:
        if item["model"] not in names or item["baseline_model"] not in names:
            raise ValueError(f"Unknown v2 pairwise comparison: {item}")
        if item["model"] == item["baseline_model"]:
            raise ValueError(f"Self comparison is invalid: {item}")
    cfg["pairwise_comparisons"] = comparisons
    cfg["plot_model_order"] = [
        str(value) for value in raw.get("plot_model_order", model_names)
    ]
    if set(cfg["plot_model_order"]) != names or len(cfg["plot_model_order"]) != len(
        names
    ):
        raise ValueError("plot_model_order must contain every v2 model exactly once")
    plot_families = raw.get(
        "plot_pairwise_families", [item["family"] for item in comparisons]
    )
    cfg["plot_pairwise_comparisons"] = [
        item for item in comparisons if item["family"] in set(plot_families)
    ]
    if len(cfg["plot_pairwise_comparisons"]) != len(plot_families):
        raise ValueError("plot_pairwise_families contains an unknown family")
    cfg["robustness_model_order"] = [
        str(value)
        for value in raw.get("robustness_model_order", cfg["plot_model_order"][:3])
    ]
    if not set(cfg["robustness_model_order"]).issubset(names):
        raise ValueError("robustness_model_order contains an unknown model")
    if cfg["pairing_mode"] != "single_parameter":
        raise ValueError("Directional Bridge v2 requires pairing_mode=single_parameter")
    specs = base._build_model_specs(specs_raw, config_path.parent)
    return cfg, specs, bridge


def _win(value: float) -> float:
    if float(value) > 0.0:
        return 1.0
    if float(value) < 0.0:
        return 0.0
    return 0.5


def decision_row(
    *,
    model: str,
    model_type: str,
    pair: Mapping[str, Any],
    curve_pair_id: int,
    anchor: str,
    scores: Sequence[float],
) -> dict[str, Any]:
    score_aa, score_ab, score_ba, score_bb = [float(value) for value in scores]
    margin_a = score_aa - score_ab
    margin_b = score_bb - score_ba
    win_a = _win(margin_a)
    win_b = _win(margin_b)
    return {
        "model": str(model),
        "model_type": str(model_type),
        **{field: pair[field] for field in v1.CONDITION_FIELDS},
        "pair_id": str(pair["pair_id"]),
        "curve_pair_id": int(curve_pair_id),
        "source_type": str(pair["source_type"]),
        "anchor": str(anchor),
        "score_aa": score_aa,
        "score_ab": score_ab,
        "score_ba": score_ba,
        "score_bb": score_bb,
        "margin_a": margin_a,
        "margin_b": margin_b,
        "win_a": win_a,
        "win_b": win_b,
        "directional_win_rate": 0.5 * (win_a + win_b),
    }


def decisions_from_legacy(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        decision_row(
            model=str(row["model"]),
            model_type=str(row["model_type"]),
            pair=row,
            curve_pair_id=int(row["curve_pair_id"]),
            anchor=str(row["anchor"]),
            scores=(row["score_aa"], row["score_ab"], row["score_ba"], row["score_bb"]),
        )
        for row in rows
    ]


def aggregate_pair_dwr(
    decision_rows: Sequence[Mapping[str, Any]],
    event_pairs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(decision_rows)
    curve = frame.groupby(
        [
            "model",
            "model_type",
            *v1.CONDITION_FIELDS,
            "pair_id",
            "curve_pair_id",
            "source_type",
        ],
        as_index=False,
        sort=True,
    )["directional_win_rate"].mean()
    pair = curve.groupby(
        ["model", "model_type", *v1.CONDITION_FIELDS, "pair_id", "source_type"],
        as_index=False,
        sort=True,
    )["directional_win_rate"].mean()
    counts = curve.groupby(["model", "pair_id"]).size().rename("n_curve_pairs")
    pair = pair.merge(counts, on=["model", "pair_id"], validate="one_to_one")
    lookup = v1._pair_lookup(event_pairs)
    for field in (
        "gw_a",
        "gw_b",
        "target_delta_iqr",
        "other_parameter_l1_distance",
        "other_parameter_max_abs_difference",
        "nuisance_l1_distance",
        "nuisance_max_abs_difference",
    ):
        pair[field] = pair["pair_id"].map(
            lambda pair_id, key=field: lookup[str(pair_id)][key]
        )
    return pair[list(PAIR_DWR_FIELDS)].to_dict("records")


def _trimmed_mean(values: np.ndarray, fraction: float = 0.05) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    trim = math.floor(len(ordered) * fraction)
    selected = (
        ordered[trim : len(ordered) - trim]
        if trim and 2 * trim < len(ordered)
        else ordered
    )
    return float(np.mean(selected))


def _scope_values(
    frame: pd.DataFrame, column: str, source: str
) -> dict[str, np.ndarray]:
    if source == "source_macro":
        groups = {
            str(key): group[column].to_numpy(dtype=np.float64)
            for key, group in frame.groupby("source_type", sort=True)
        }
        if set(groups) != {"bns", "nsbh"}:
            raise ValueError("source_macro requires BNS and NSBH values")
        return groups
    values = frame.loc[frame["source_type"] == source, column].to_numpy(
        dtype=np.float64
    )
    return {source: values}


def resample_summary(
    frame: pd.DataFrame,
    *,
    column: str,
    source: str,
    null_value: float,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
    alternative: str,
) -> dict[str, Any]:
    groups = _scope_values(frame, column, source)
    if any(values.size == 0 for values in groups.values()):
        raise ValueError("Cannot resample an empty DWR group")
    observed = float(np.mean([values.mean() for values in groups.values()]))
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(bootstrap_samples), dtype=np.float64)
    permutation = np.empty(int(permutation_samples), dtype=np.float64)
    for start in range(0, int(bootstrap_samples), 250):
        stop = min(start + 250, int(bootstrap_samples))
        draws = []
        for values in groups.values():
            indices = rng.integers(0, values.size, size=(stop - start, values.size))
            draws.append(values[indices].mean(axis=1))
        bootstrap[start:stop] = np.mean(draws, axis=0)
    centered_observed = observed - float(null_value)
    for start in range(0, int(permutation_samples), 250):
        stop = min(start + 250, int(permutation_samples))
        draws = []
        for values in groups.values():
            centered = values - float(null_value)
            signs = rng.choice(
                np.asarray([-1.0, 1.0]), size=(stop - start, values.size)
            )
            draws.append((signs * centered).mean(axis=1))
        permutation[start:stop] = np.mean(draws, axis=0)
    if alternative == "greater":
        extreme = np.count_nonzero(permutation >= centered_observed)
    elif alternative == "two-sided":
        extreme = np.count_nonzero(np.abs(permutation) >= abs(centered_observed))
    else:
        raise ValueError(f"Unsupported alternative {alternative}")
    raw = np.concatenate(list(groups.values()))
    fraction = float(np.mean(raw > 0.5)) if np.isclose(null_value, 0.5) else ""
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "permutation_p": float((1 + extreme) / (int(permutation_samples) + 1)),
        "median": float(np.mean([np.median(values) for values in groups.values()])),
        "trimmed_mean_5pct": float(
            np.mean([_trimmed_mean(values) for values in groups.values()])
        ),
        "fraction_above_0_5": fraction,
    }


def _primary_source(target: str) -> str:
    if target in v1.SINGLE_PARAMETER_COMMON_TARGETS:
        return "source_macro"
    return "bns" if target == "chi_eff" else "nsbh"


def summarize_dwr(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    new_model: str,
    default_model: str,
    primary_caliper: float,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
    comparisons: Sequence[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(pair_rows)
    rows: list[dict[str, Any]] = []
    if comparisons is None:
        comparisons = (
            (new_model, BRIDGE_NAME, "mixed_minus_bridge_directional_win_rate"),
            (default_model, BRIDGE_NAME, "default_minus_bridge_directional_win_rate"),
            (new_model, default_model, "mixed_minus_default_directional_win_rate"),
        )
    for (target_value, caliper_value), target_frame in frame.groupby(
        ["target_parameter", "caliper_iqr"], sort=True
    ):
        target = str(target_value)
        caliper = float(caliper_value)
        primary = bool(np.isclose(caliper, primary_caliper, atol=1e-12, rtol=0.0))
        scopes = [str(value) for value in sorted(target_frame["source_type"].unique())]
        if target in v1.SINGLE_PARAMETER_COMMON_TARGETS:
            scopes = ["source_macro", *scopes]
        for model, model_frame in target_frame.groupby("model", sort=True):
            for source in scopes:
                family = ""
                if (
                    primary
                    and str(model) == BRIDGE_NAME
                    and source == _primary_source(target)
                ):
                    family = "bridge_absolute_directional_win_rate"
                if str(model) == GW_BLIND_NAME:
                    stats = {
                        "estimate": 0.5,
                        "ci95_low": 0.5,
                        "ci95_high": 0.5,
                        "permutation_p": "",
                        "median": 0.5,
                        "trimmed_mean_5pct": 0.5,
                        "fraction_above_0_5": 0.0,
                    }
                else:
                    stats = resample_summary(
                        model_frame,
                        column="directional_win_rate",
                        source=source,
                        null_value=0.5,
                        bootstrap_samples=bootstrap_samples,
                        permutation_samples=permutation_samples,
                        seed=v1._stable_seed(
                            seed, "dwr", target, caliper, model, source
                        ),
                        alternative="greater",
                    )
                rows.append(
                    {
                        "endpoint": "absolute_directional_win_rate",
                        "family": family,
                        "target_parameter": target,
                        "caliper_iqr": caliper,
                        "is_primary_caliper": primary,
                        "model": str(model),
                        "baseline_model": "",
                        "source": source,
                        "n_pairs": int(
                            model_frame["pair_id"].nunique()
                            if source == "source_macro"
                            else model_frame.loc[
                                model_frame["source_type"] == source, "pair_id"
                            ].nunique()
                        ),
                        **stats,
                        "null_value": 0.5,
                        "holm_adjusted_p": "",
                        "holm_reject_0_05": "",
                        "bootstrap_samples": int(bootstrap_samples),
                        "permutation_samples": int(permutation_samples),
                    }
                )
        for left, right, family_name in comparisons:
            left_frame = target_frame[target_frame["model"] == left]
            right_frame = target_frame[target_frame["model"] == right]
            merged = left_frame.merge(
                right_frame,
                on=[
                    "pair_id",
                    "source_type",
                    "condition_id",
                    "target_parameter",
                    "caliper_iqr",
                ],
                suffixes=("_left", "_right"),
                validate="one_to_one",
            )
            if len(merged) != len(left_frame) or len(merged) != len(right_frame):
                raise ValueError(f"Pair manifest mismatch for {left} versus {right}")
            delta = pd.DataFrame(
                {
                    "pair_id": merged["pair_id"],
                    "source_type": merged["source_type"],
                    "delta": merged["directional_win_rate_left"]
                    - merged["directional_win_rate_right"],
                }
            )
            for source in scopes:
                family = (
                    family_name if primary and source == _primary_source(target) else ""
                )
                stats = resample_summary(
                    delta,
                    column="delta",
                    source=source,
                    null_value=0.0,
                    bootstrap_samples=bootstrap_samples,
                    permutation_samples=permutation_samples,
                    seed=v1._stable_seed(
                        seed, "delta", target, caliper, left, right, source
                    ),
                    alternative="two-sided",
                )
                rows.append(
                    {
                        "endpoint": "paired_directional_win_rate_delta",
                        "family": family,
                        "target_parameter": target,
                        "caliper_iqr": caliper,
                        "is_primary_caliper": primary,
                        "model": left,
                        "baseline_model": right,
                        "source": source,
                        "n_pairs": int(
                            delta["pair_id"].nunique()
                            if source == "source_macro"
                            else delta.loc[
                                delta["source_type"] == source, "pair_id"
                            ].nunique()
                        ),
                        **stats,
                        "null_value": 0.0,
                        "holm_adjusted_p": "",
                        "holm_reject_0_05": "",
                        "bootstrap_samples": int(bootstrap_samples),
                        "permutation_samples": int(permutation_samples),
                    }
                )
    summary = pd.DataFrame(rows)
    for family in sorted(value for value in summary["family"].unique() if value):
        indices = summary.index[summary["family"] == family]
        if len(indices) != 6:
            raise ValueError(
                f"DWR Holm family {family} has {len(indices)} endpoints; expected 6"
            )
        adjusted = v1.holm_adjust(summary.loc[indices, "permutation_p"].to_numpy(float))
        summary.loc[indices, "holm_adjusted_p"] = adjusted
        summary.loc[indices, "holm_reject_0_05"] = adjusted <= 0.05
    return summary[list(DWR_SUMMARY_FIELDS)].to_dict("records")


def dose_response_dwr(
    pair_rows: Sequence[Mapping[str, Any]], *, primary_caliper: float, n_bins: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.DataFrame(pair_rows)
    frame = frame[
        np.isclose(frame["caliper_iqr"], primary_caliper, atol=1e-12, rtol=0.0)
    ].copy()
    labelled: list[pd.DataFrame] = []
    for (_, _), group in frame.groupby(["target_parameter", "source_type"], sort=True):
        unique = group.drop_duplicates("pair_id")[
            ["pair_id", "target_delta_iqr"]
        ].sort_values(["target_delta_iqr", "pair_id"], kind="stable")
        assignment: dict[str, str] = {}
        for bin_index, positions in enumerate(
            np.array_split(np.arange(len(unique)), n_bins), start=1
        ):
            suffix = (
                "low" if bin_index == 1 else "high" if bin_index == n_bins else "mid"
            )
            for pair_id in unique.iloc[positions]["pair_id"]:
                assignment[str(pair_id)] = f"T{bin_index}_{suffix}"
        part = group.copy()
        part["dose_bin"] = part["pair_id"].map(assignment)
        labelled.append(part)
    data = pd.concat(labelled, ignore_index=True)
    rows: list[dict[str, Any]] = []
    trends: list[dict[str, Any]] = []
    for (target, source, model), group in data.groupby(
        ["target_parameter", "source_type", "model"], sort=True
    ):
        trends.append(
            {
                "target_parameter": str(target),
                "source": str(source),
                "model": str(model),
                "n_pairs": int(group["pair_id"].nunique()),
                "spearman_target_delta_directional_win_rate": v1._spearman_rank_correlation(
                    group["target_delta_iqr"].to_numpy(float),
                    group["directional_win_rate"].to_numpy(float),
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
                    "directional_win_rate": float(dose["directional_win_rate"].mean()),
                }
            )
    source_rows = pd.DataFrame(rows)
    source_trends = pd.DataFrame(trends)
    for target in v1.SINGLE_PARAMETER_COMMON_TARGETS:
        for (model, dose_bin), group in source_rows[
            source_rows["target_parameter"] == target
        ].groupby(["model", "dose_bin"], sort=True):
            if set(group["source"]) != {"bns", "nsbh"}:
                raise ValueError(f"Incomplete dose source macro for {target}")
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
                    "directional_win_rate": float(group["directional_win_rate"].mean()),
                }
            )
        for model, group in source_trends[
            source_trends["target_parameter"] == target
        ].groupby("model", sort=True):
            if set(group["source"]) != {"bns", "nsbh"}:
                raise ValueError(f"Incomplete trend source macro for {target}")
            trends.append(
                {
                    "target_parameter": target,
                    "source": "source_macro",
                    "model": str(model),
                    "n_pairs": int(group["n_pairs"].sum()),
                    "spearman_target_delta_directional_win_rate": float(
                        group["spearman_target_delta_directional_win_rate"].mean()
                    ),
                }
            )
    return rows, trends


def _neighbor_rows(
    side: str,
    source: str,
    query_id: int,
    neighbors: Any,
    reference_parents: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "side": side,
            "source_type": source,
            "query_id": int(query_id),
            "rank": rank,
            "reference_parent_gw_id": int(reference_parents[index]),
            "distance": float(distance),
            "weight": float(weight),
            "cadence_fallback_level": int(neighbors.fallback_level),
        }
        for rank, (index, distance, weight) in enumerate(
            zip(neighbors.indices[:8], neighbors.distances[:8], neighbors.weights[:8]),
            start=1,
        )
    ]


def score_bridge(
    *,
    artifact_path: str,
    cfg: Mapping[str, Any],
    bank: Mapping[str, torch.Tensor],
    compact_index: Mapping[int, int],
    event_pairs: Sequence[Mapping[str, Any]],
    curve_pairs: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    bridge = PhysicsEjectaBridge(artifact_path)
    with h5py.File(cfg["test_data_path"], "r") as handle:
        psfflux_zp = float(handle.attrs["psfflux_zp"])
        lupt_b_njy = np.asarray(handle.attrs["lupt_b_njy"], dtype=np.float64)
    features, feature_mask, cadence = extract_lightcurve_features(
        np.asarray(bank["times"].cpu()),
        np.asarray(bank["values"].cpu()),
        np.asarray(bank["masks"].cpu()),
        np.asarray(bank["errors"].cpu()),
        psfflux_zp=psfflux_zp,
        lupt_b_njy=lupt_b_njy,
    )
    selected_optical = np.asarray(bank["source_optical_indices"].cpu(), dtype=np.int64)
    selected_parent = np.asarray(bank["gw_indices"].cpu(), dtype=np.int64)
    lc_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    lc_neighbor_rows: list[dict[str, Any]] = []
    for source in ("bns", "nsbh"):
        rows = np.asarray(
            [
                index
                for index, parent in enumerate(selected_parent.tolist())
                if str(metadata["gw_source_types"][parent]) == source
            ],
            dtype=np.int64,
        )
        if not rows.size:
            continue
        means, covariances, neighbors = bridge.lc_posteriors_batch(
            features[rows],
            feature_mask[rows],
            cadence[rows],
            source,
            device=str(device),
            batch_size=int(cfg.get("bridge_lc_batch_size", 64)),
        )
        reference_parents = bridge._key(source, "lc_parent_ids")
        for local, compact in enumerate(rows.tolist()):
            optical_index = int(selected_optical[compact])
            lc_cache[(source, optical_index)] = (means[local], covariances[local])
            lc_neighbor_rows.extend(
                _neighbor_rows(
                    "lc", source, optical_index, neighbors[local], reference_parents
                )
            )
    gw_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    gw_neighbor_rows: list[dict[str, Any]] = []
    for gw_id in sorted(
        {int(value) for pair in event_pairs for value in (pair["gw_a"], pair["gw_b"])}
    ):
        source = str(metadata["gw_source_types"][gw_id])
        mean, covariance, neighbors = bridge.gw_posterior(
            metadata["gw_scalars"][gw_id], source
        )
        gw_cache[gw_id] = (mean, covariance)
        gw_neighbor_rows.extend(
            _neighbor_rows(
                "gw", source, gw_id, neighbors, bridge._key(source, "gw_parent_ids")
            )
        )
    curves_by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for curve_pair in curve_pairs:
        curves_by_pair[str(curve_pair["pair_id"])].append(curve_pair)
    score_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for pair in tqdm(event_pairs, desc="  Cross-scoring Physics Ejecta Bridge"):
        source = str(pair["source_type"])
        gw_a = int(pair["gw_a"])
        gw_b = int(pair["gw_b"])
        for curve_pair in sorted(
            curves_by_pair[str(pair["pair_id"])],
            key=lambda row: int(row["curve_pair_id"]),
        ):
            optical_a = int(curve_pair["optical_index_a"])
            optical_b = int(curve_pair["optical_index_b"])
            scores = (
                bhattacharyya_score(*gw_cache[gw_a], *lc_cache[(source, optical_a)]),
                bhattacharyya_score(*gw_cache[gw_a], *lc_cache[(source, optical_b)]),
                bhattacharyya_score(*gw_cache[gw_b], *lc_cache[(source, optical_a)]),
                bhattacharyya_score(*gw_cache[gw_b], *lc_cache[(source, optical_b)]),
            )
            for anchor in ("a", "b"):
                anchor_optical = int(curve_pair[f"optical_index_{anchor}"])
                anchor_gw = int(pair[f"gw_{anchor}"])
                coordinate = np.asarray(
                    metadata["coordinates"][anchor_optical], dtype=np.float32
                )
                dt_days = abs(
                    float(metadata["first_detection_mjd"][anchor_optical])
                    - float(metadata["event_time_mjd"][anchor_gw])
                )
                score_rows.extend(
                    v1._score_rows_for_block(
                        model_name=BRIDGE_NAME,
                        model_type="physics_ejecta_bridge",
                        pair=pair,
                        curve_pair=curve_pair,
                        anchor=anchor,
                        anchor_coordinate=coordinate,
                        anchor_dt=dt_days,
                        scores=scores,
                    )
                )
                decision_rows.append(
                    decision_row(
                        model=BRIDGE_NAME,
                        model_type="physics_ejecta_bridge",
                        pair=pair,
                        curve_pair_id=int(curve_pair["curve_pair_id"]),
                        anchor=anchor,
                        scores=scores,
                    )
                )
    info = {
        "name": BRIDGE_NAME,
        "type": "physics_ejecta_bridge",
        "artifact_path": str(Path(artifact_path).resolve()),
        "artifact_version": bridge.manifest["artifact_version"],
        "test_truth_fields_used_for_scoring": [],
        "score": "negative_gaussian_bhattacharyya_distance",
    }
    bridge.arrays.close()
    return score_rows, decision_rows, gw_neighbor_rows, lc_neighbor_rows, info


def plot_dwr_results(
    summary_rows: Sequence[Mapping[str, Any]],
    dose_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    model_order: Sequence[str],
    pairwise_comparisons: Sequence[Mapping[str, str]],
    robustness_models: Sequence[str],
) -> list[str]:
    import matplotlib

    matplotlib.use("pgf", force=True)
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "xelatex",
            "pgf.rcfonts": False,
            "pgf.preamble": PLOT_LATEX_PREAMBLE,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 12 * PLOT_FONT_SCALE,
            "axes.labelsize": 12 * PLOT_AXIS_SCALE,
            "axes.titlesize": 14 * PLOT_AXIS_SCALE,
            "xtick.labelsize": 11 * PLOT_AXIS_SCALE,
            "ytick.labelsize": 11 * PLOT_AXIS_SCALE,
            "legend.fontsize": 10 * PLOT_LEGEND_SCALE,
        }
    )
    import matplotlib.pyplot as plt

    summary = pd.DataFrame(summary_rows)
    primary = summary[
        summary["is_primary_caliper"].astype(bool)
        & summary.apply(
            lambda row: str(row["source"])
            == _primary_source(str(row["target_parameter"])),
            axis=1,
        )
    ].copy()
    targets = [
        "chirp_mass_detector",
        "mass_ratio",
        "chi_eff",
        "primary_spin_z",
        "abs_costheta",
        "log10_distance_gpc",
    ]
    target_labels = [PLOT_PARAMETER_LABELS[target] for target in targets]
    models = list(model_order)
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#9467bd",
        "#2ca02c",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]
    colors = [palette[index % len(palette)] for index in range(len(models))]
    color_map = dict(zip(models, colors))
    artifacts: list[str] = []
    absolute = primary[primary["endpoint"] == "absolute_directional_win_rate"]
    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = np.linspace(-0.24, 0.24, len(models))
    for offset, model, color in zip(offsets, models, colors):
        rows = absolute[absolute["model"] == model].set_index("target_parameter")
        y = np.arange(len(targets)) + offset
        estimate = np.asarray(
            [float(rows.loc[target, "estimate"]) for target in targets]
        )
        low = np.asarray([float(rows.loc[target, "ci95_low"]) for target in targets])
        high = np.asarray([float(rows.loc[target, "ci95_high"]) for target in targets])
        ax.errorbar(
            estimate,
            y,
            xerr=[estimate - low, high - estimate],
            fmt="o",
            capsize=3,
            color=color,
            label=model,
        )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(targets)), target_labels)
    ax.set_xlabel("Directional-win rate")
    ax.legend(loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        name = f"directional_win_with_physics_bridge.{suffix}"
        fig.savefig(
            output_dir / name, dpi=300 if suffix == "png" else None, bbox_inches="tight"
        )
        artifacts.append(name)
    plt.close(fig)
    delta = primary[primary["endpoint"] == "paired_directional_win_rate_delta"].copy()
    comparisons = list(pairwise_comparisons)
    fig, ax = plt.subplots(figsize=(10, 6))
    offsets = np.linspace(-0.18, 0.18, len(comparisons))
    for offset, comparison in zip(offsets, comparisons):
        model = str(comparison["model"])
        baseline = str(comparison["baseline_model"])
        label = str(comparison["label"])
        color = color_map.get(model, "#333333")
        rows = delta[
            (delta["model"] == model) & (delta["baseline_model"] == baseline)
        ].set_index("target_parameter")
        y = np.arange(len(targets)) + offset
        estimate = np.asarray(
            [float(rows.loc[target, "estimate"]) for target in targets]
        )
        low = np.asarray([float(rows.loc[target, "ci95_low"]) for target in targets])
        high = np.asarray([float(rows.loc[target, "ci95_high"]) for target in targets])
        ax.errorbar(
            estimate,
            y,
            xerr=[estimate - low, high - estimate],
            fmt="o",
            capsize=3,
            color=color,
            label=label,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(np.arange(len(targets)), target_labels)
    ax.set_xlabel("Paired directional-win difference")
    ax.legend(loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        name = f"directional_win_model_differences.{suffix}"
        fig.savefig(
            output_dir / name, dpi=300 if suffix == "png" else None, bbox_inches="tight"
        )
        artifacts.append(name)
    plt.close(fig)
    robustness = summary[
        (summary["endpoint"] == "absolute_directional_win_rate")
        & summary.apply(
            lambda row: str(row["source"])
            == _primary_source(str(row["target_parameter"])),
            axis=1,
        )
        & summary["model"].isin(robustness_models)
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, target in zip(axes.flat, targets):
        for model in robustness_models:
            color = color_map.get(model, "#333333")
            rows = robustness[
                (robustness["target_parameter"] == target)
                & (robustness["model"] == model)
            ].sort_values("caliper_iqr")
            ax.plot(
                rows["caliper_iqr"],
                rows["estimate"] - 0.5,
                marker="o",
                color=color,
                label=model,
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(PLOT_PARAMETER_LABELS[target])
    axes[0, 0].legend(fontsize=8 * PLOT_LEGEND_SCALE)
    fig.supxlabel("Maximum non-target GW difference (IQR)")
    fig.supylabel("Directional-win - 0.5")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        name = f"directional_win_caliper_robustness.{suffix}"
        fig.savefig(
            output_dir / name, dpi=300 if suffix == "png" else None, bbox_inches="tight"
        )
        artifacts.append(name)
    plt.close(fig)
    dose = pd.DataFrame(dose_rows)
    dose = dose[
        dose.apply(
            lambda row: str(row["source"])
            == _primary_source(str(row["target_parameter"])),
            axis=1,
        )
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, target in zip(axes.flat, targets):
        for model in robustness_models:
            color = color_map.get(model, "#333333")
            rows = dose[
                (dose["target_parameter"] == target) & (dose["model"] == model)
            ].sort_values("dose_bin")
            ax.plot(
                rows["dose_bin"].map(DOSE_BIN_LABELS),
                rows["directional_win_rate"],
                marker="o",
                color=color,
                label=model,
            )
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title(PLOT_PARAMETER_LABELS[target])
    axes[0, 0].legend(fontsize=8 * PLOT_LEGEND_SCALE)
    fig.supxlabel("Target-separation rank tertile")
    fig.supylabel("Directional-win rate")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        name = f"directional_win_dose_response.{suffix}"
        fig.savefig(
            output_dir / name, dpi=300 if suffix == "png" else None, bbox_inches="tight"
        )
        artifacts.append(name)
    plt.close(fig)
    return artifacts


def run_directional_bridge(
    config_path: Path,
    raw: Mapping[str, Any],
    *,
    validate_only: bool,
    preflight_pairs: bool,
) -> None:
    cfg, model_specs, _bridge_spec = normalise_v2_config(raw, config_path)
    if not Path(cfg["test_data_path"]).is_file():
        raise FileNotFoundError(cfg["test_data_path"])
    PhysicsEjectaBridge(cfg["bridge_artifact_path"]).arrays.close()
    if validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "config": str(config_path),
                    "primary_metric": "directional_win_rate",
                },
                indent=2,
            )
        )
        return
    v1._seed_all(cfg["seed"])
    metadata = v1._read_metadata(cfg["test_data_path"], cfg["comparison_window"])
    event_pairs, curve_pairs, pairing_audit = v1._prepare_single_parameter_pairs(
        metadata, cfg
    )
    pair_digest = stable_digest(
        {"event_pairs": event_pairs, "curve_pairs": curve_pairs}
    )
    if pair_digest != cfg["expected_pair_manifest_sha256"]:
        raise AssertionError(f"Pair digest changed: {pair_digest}")
    pair_counts = {
        condition: sum(1 for pair in event_pairs if pair["condition_id"] == condition)
        for condition in sorted({pair["condition_id"] for pair in event_pairs})
    }
    if preflight_pairs:
        print(
            json.dumps(
                {
                    "status": "pairing_preflight_valid",
                    "n_event_pairs": len(event_pairs),
                    "n_curve_pairs": len(curve_pairs),
                    "pair_manifest_sha256": pair_digest,
                    "pair_counts": pair_counts,
                },
                indent=2,
            )
        )
        return
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    amp_dtype, amp_enabled = base._resolve_eval_amp(cfg["amp_dtype"], device)
    output_dir = Path(cfg["output_dir"])
    manifest = prepare_output_directory(
        output_dir,
        manifest={
            "experiment_id": str(raw["experiment_id"]),
            "config_path": str(config_path),
            "pair_manifest_sha256": pair_digest,
            "primary_metric": "directional_win_rate",
            "interaction_comparison": False,
            "seed": cfg["seed"],
        },
    )
    write_csv_atomic(output_dir / "event_pairs.csv", event_pairs, v1.SINGLE_PAIR_FIELDS)
    write_csv_atomic(
        output_dir / "curve_pairs.csv", curve_pairs, v1.SINGLE_CURVE_PAIR_FIELDS
    )
    write_csv_atomic(
        output_dir / "matching_balance.csv",
        v1.matching_balance_rows(event_pairs),
        v1.MATCHING_BALANCE_FIELDS,
    )
    write_json_atomic(output_dir / "pairing_audit.json", pairing_audit)
    selected_optical = np.unique(
        np.asarray(
            [
                int(value)
                for pair in curve_pairs
                for value in (pair["optical_index_a"], pair["optical_index_b"])
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
    all_scores: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    model_info: list[dict[str, Any]] = []
    for spec in sorted(
        model_specs, key=lambda item: 1 if item["type"] == "optical" else 0
    ):
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
            scores, legacy_rows, info = v1._score_multimodal(**kwargs)
        elif spec["type"] == "optical":
            scores, legacy_rows, info = v1._score_optical_null(**kwargs)
        else:
            raise ValueError(f"Unsupported v2 neural/control type {spec['type']}")
        all_scores.extend(scores)
        all_decisions.extend(decisions_from_legacy(legacy_rows))
        model_info.append(info)
    bridge_scores, bridge_decisions, gw_neighbors, lc_neighbors, bridge_info = (
        score_bridge(
            artifact_path=cfg["bridge_artifact_path"],
            cfg=cfg,
            bank=bank,
            compact_index=compact_index,
            event_pairs=event_pairs,
            curve_pairs=curve_pairs,
            metadata=metadata,
            device=device,
        )
    )
    all_scores.extend(bridge_scores)
    all_decisions.extend(bridge_decisions)
    model_info.append(bridge_info)
    pair_rows = aggregate_pair_dwr(all_decisions, event_pairs)
    blind = [
        float(row["directional_win_rate"])
        for row in pair_rows
        if row["model"] == GW_BLIND_NAME
    ]
    if not blind or max(abs(value - 0.5) for value in blind) > 0.0:
        raise AssertionError("GW-blind Control must have directional-win exactly 0.5")
    summary_rows = summarize_dwr(
        pair_rows,
        new_model=cfg["new_model_name"],
        default_model=cfg["baseline_model_name"],
        primary_caliper=cfg["primary_other_parameter_caliper_iqr"],
        bootstrap_samples=cfg["bootstrap_samples"],
        permutation_samples=cfg["permutation_samples"],
        seed=cfg["seed"],
        comparisons=[
            (item["model"], item["baseline_model"], item["family"])
            for item in cfg["pairwise_comparisons"]
        ],
    )
    robustness_rows = [
        row
        for row in summary_rows
        if row["endpoint"] == "absolute_directional_win_rate"
        and row["source"] == _primary_source(row["target_parameter"])
    ]
    dose_rows, trend_rows = dose_response_dwr(
        pair_rows,
        primary_caliper=cfg["primary_other_parameter_caliper_iqr"],
        n_bins=cfg["dose_response_bins"],
    )
    score_writer = AtomicGzipCsvWriter(
        output_dir / "candidate_scores.csv.gz", v1.SINGLE_SCORE_FIELDS
    )
    try:
        score_writer.writerows(all_scores)
        score_writer.commit()
    finally:
        score_writer.close()
    write_csv_atomic(
        output_dir / "anchor_pairing_decisions.csv", all_decisions, DECISION_FIELDS
    )
    write_csv_atomic(
        output_dir / "pair_directional_wins.csv", pair_rows, PAIR_DWR_FIELDS
    )
    write_csv_atomic(
        output_dir / "directional_win_summary.csv", summary_rows, DWR_SUMMARY_FIELDS
    )
    write_csv_atomic(
        output_dir / "caliper_robustness_directional_win.csv",
        robustness_rows,
        DWR_SUMMARY_FIELDS,
    )
    write_csv_atomic(
        output_dir / "dose_response_directional_win.csv", dose_rows, DOSE_FIELDS
    )
    write_csv_atomic(
        output_dir / "parameter_trends_directional_win.csv", trend_rows, TREND_FIELDS
    )
    gw_writer = AtomicGzipCsvWriter(
        output_dir / "bridge_gw_neighbors.csv.gz", NEIGHBOR_FIELDS
    )
    lc_writer = AtomicGzipCsvWriter(
        output_dir / "bridge_lc_neighbors.csv.gz", NEIGHBOR_FIELDS
    )
    try:
        gw_writer.writerows(gw_neighbors)
        gw_writer.commit()
        lc_writer.writerows(lc_neighbors)
        lc_writer.commit()
    finally:
        gw_writer.close()
        lc_writer.close()
    plot_artifacts = plot_dwr_results(
        summary_rows,
        dose_rows,
        output_dir,
        model_order=cfg["plot_model_order"],
        pairwise_comparisons=cfg["plot_pairwise_comparisons"],
        robustness_models=cfg["robustness_model_order"],
    )
    result = {
        "experiment_id": str(raw["experiment_id"]),
        "pairing_mode": "single_parameter",
        "primary_metric": "directional_win_rate",
        "interaction_comparison": False,
        "pair_manifest_sha256": pair_digest,
        "pair_counts": pair_counts,
        "n_curve_pairs": len(curve_pairs),
        "models": model_info,
        "summary": summary_rows,
        "dose_response": dose_rows,
        "parameter_trends": trend_rows,
        "protocol": {
            "directional_win": "mean(1[margin>0] + 0.5*1[margin=0]) over the two query directions",
            "aggregation": "anchors then curve pairs then independent event pairs",
            "source_aggregation": "equal_weight_bns_nsbh",
            "brightness": "native_unmodified_not_normalized",
            "physics_bridge_role": "achievable_physics_reference_not_bayes_ceiling",
            "test_truth_fields_used_for_bridge_scoring": [],
            "pairwise_comparisons": cfg["pairwise_comparisons"],
        },
        "config": cfg,
    }
    write_json_atomic(output_dir / "directional_win_summary.json", result)
    artifacts = [
        "event_pairs.csv",
        "curve_pairs.csv",
        "matching_balance.csv",
        "pairing_audit.json",
        "candidate_scores.csv.gz",
        "anchor_pairing_decisions.csv",
        "pair_directional_wins.csv",
        "bridge_gw_neighbors.csv.gz",
        "bridge_lc_neighbors.csv.gz",
        "directional_win_summary.csv",
        "caliper_robustness_directional_win.csv",
        "dose_response_directional_win.csv",
        "parameter_trends_directional_win.csv",
        "directional_win_summary.json",
        *plot_artifacts,
    ]
    mark_run_success(output_dir, manifest, artifacts)
    del bank
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"GW--KN directional Bridge comparison complete: {output_dir}")
