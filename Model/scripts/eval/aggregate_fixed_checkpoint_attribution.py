#!/usr/bin/env python3
"""Aggregate paired fixed-checkpoint attribution runs with hierarchical bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval.eval_run_io import (  # noqa: E402
    stable_digest,
    write_csv_atomic,
    write_json_atomic,
)
from scripts.eval.run_fixed_checkpoint_attribution import (  # noqa: E402
    MATCHED_CONDITIONS,
    OPERATIONAL_CONDITIONS,
    RANDOM_MATCHED_CONDITIONS,
)


METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "positive_score",
    "score_margin",
)
COMPARISON_FIELDS = (
    "task",
    "condition",
    "reference_condition",
    "source_type",
    "gallery_size",
    "metric",
    "n_seeds",
    "n_pairs",
    "condition_mean",
    "reference_mean",
    "mean_delta",
    "ci_low",
    "ci_high",
    "p_value",
    "p_value_bh",
    "equivalence_margin",
    "decision",
    "physics_expectation",
)
PER_SEED_FIELDS = (
    "task",
    "condition",
    "source_type",
    "gallery_size",
    "metric",
    "seed",
    "n_rows",
    "value",
)
CORRELATION_FIELDS = (
    "task",
    "condition",
    "seed",
    "source_type",
    "gallery_size",
    "mismatch",
    "n_candidates",
    "n_query_trials",
    "n_gw",
    "correlation_method",
    "spearman_rho",
    "p_value",
)
FACTORIAL_DEFINITIONS = (
    {
        "combined": "distance_perm__brightness_norm",
        "gw_only": "distance_perm",
        "optical_only": "brightness_norm",
        "question": "distance_x_brightness",
    },
    {
        "combined": "spin1z_perm__time_shuffle",
        "gw_only": "spin1z_perm",
        "optical_only": "time_shuffle",
        "question": "primary_spin_x_temporal_evolution",
    },
    {
        "combined": "inclination_abs_perm__per_band_norm",
        "gw_only": "inclination_abs_perm",
        "optical_only": "per_band_norm",
        "question": "inclination_x_color_amplitude",
    },
    {
        "combined": "inclination_abs_perm__time_shuffle",
        "gw_only": "inclination_abs_perm",
        "optical_only": "time_shuffle",
        "question": "inclination_x_temporal_evolution",
    },
)
FACTORIAL_FIELDS = (
    "task",
    "question",
    "combined_condition",
    "gw_only_condition",
    "optical_only_condition",
    "reference_condition",
    "source_type",
    "gallery_size",
    "metric",
    "n_seeds",
    "n_pairs",
    "reference_mean",
    "gw_only_mean",
    "optical_only_mean",
    "combined_mean",
    "interaction",
    "ci_low",
    "ci_high",
    "p_value",
    "p_value_bh",
    "equivalence_margin",
    "decision",
)
GALLERY_STRATEGY_FIELDS = (
    "source_type",
    "gallery_size",
    "metric",
    "n_seeds",
    "n_pairs",
    "nuisance_nearest_mean",
    "random_same_source_mean",
    "mean_delta_nearest_minus_random",
    "ci_low",
    "ci_high",
    "p_value",
    "p_value_bh",
    "equivalence_margin",
    "decision",
)
DISTRACTOR_PHYSICAL_FIELDS = (
    "task",
    "seed",
    "source_type",
    "gallery_size",
    "mismatch",
    "n_candidates",
    "mean",
    "median",
    "p90",
    "zero_fraction",
)
DISTRACTOR_NUISANCE_FIELDS = (
    "task",
    "seed",
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


def _validated_run(run_dir: Path) -> tuple[Dict[str, Any], Dict[str, Any]]:
    success_path = run_dir / "_SUCCESS.json"
    manifest_path = run_dir / "run_manifest.json"
    if not success_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete attribution run: {run_dir}")
    success = _load_json(success_path)
    manifest = _load_json(manifest_path)
    unsigned = dict(manifest)
    digest = unsigned.pop("manifest_digest", None)
    if not digest or stable_digest(unsigned) != digest:
        raise ValueError(f"Invalid manifest digest: {run_dir}")
    if success.get("manifest_digest") != digest or success.get("status") != "complete":
        raise ValueError(f"Invalid success marker: {run_dir}")
    for artifact in success.get("artifacts", []):
        relative = Path(str(artifact))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe artifact path in {success_path}: {artifact}")
        if not (run_dir / relative).is_file():
            raise FileNotFoundError(run_dir / relative)
    return manifest, success


def discover_runs(input_roots: Sequence[Path]) -> list[Dict[str, Any]]:
    """Discover complete unique task/condition/seed runs below input roots."""
    runs: list[Dict[str, Any]] = []
    identities: set[tuple[str, str, int]] = set()
    for root in input_roots:
        for success_path in sorted(root.rglob("_SUCCESS.json")):
            run_dir = success_path.parent
            manifest, _ = _validated_run(run_dir)
            identity = (
                str(manifest["task"]),
                str(manifest["condition"]),
                int(manifest["seed"]),
            )
            if identity in identities:
                raise ValueError(f"Duplicate attribution run identity: {identity}")
            identities.add(identity)
            gallery = _load_json(run_dir / "gallery_identity.json")
            result = _load_json(run_dir / "attribution_result.json")
            runs.append(
                {
                    "run_dir": run_dir,
                    "manifest": manifest,
                    "gallery": gallery,
                    "result": result,
                }
            )
    if not runs:
        raise FileNotFoundError("No complete attribution runs found.")
    return runs


def validate_run_matrix(runs: Sequence[Mapping[str, Any]]) -> None:
    """Reject partial condition/seed matrices and cross-run code or input drift."""
    observed_tasks = {str(run["manifest"]["task"]) for run in runs}
    current_tasks = {
        "operational_nonkn",
        "kn_nuisance_matched",
        "kn_same_source_random",
    }
    legacy_tasks = {"operational_nonkn", "kn_nuisance_matched"}
    if observed_tasks == current_tasks:
        expected = {
            "operational_nonkn": {item["name"] for item in OPERATIONAL_CONDITIONS},
            "kn_nuisance_matched": {item["name"] for item in MATCHED_CONDITIONS},
            "kn_same_source_random": {
                item["name"] for item in RANDOM_MATCHED_CONDITIONS
            },
        }
    elif observed_tasks == legacy_tasks:
        factorial_names = {item["combined"] for item in FACTORIAL_DEFINITIONS}
        expected = {
            "operational_nonkn": {item["name"] for item in OPERATIONAL_CONDITIONS},
            "kn_nuisance_matched": {
                item["name"]
                for item in MATCHED_CONDITIONS
                if item["name"] not in factorial_names
            },
        }
    else:
        raise ValueError(
            "Expected the legacy two-task or current three-task attribution "
            f"matrix, found {sorted(observed_tasks)}"
        )
    seed_sets: Dict[tuple[str, str], set[int]] = {}
    for run in runs:
        manifest = run["manifest"]
        key = (str(manifest["task"]), str(manifest["condition"]))
        seed_sets.setdefault(key, set()).add(int(manifest["seed"]))
    all_seeds = set().union(*seed_sets.values())
    for task, expected_conditions in expected.items():
        for seed in sorted(all_seeds):
            observed = {
                condition
                for (observed_task, condition), seeds in seed_sets.items()
                if observed_task == task and seed in seeds
            }
            if observed != expected_conditions:
                missing = sorted(expected_conditions.difference(observed))
                extra = sorted(observed.difference(expected_conditions))
                raise ValueError(
                    f"Incomplete condition matrix for {task}/seed={seed}: "
                    f"missing={missing}, extra={extra}"
                )

    invariants = {
        "code_digest": {str(run["manifest"]["code_digest"]) for run in runs},
        "test_data_path": {
            str(run["manifest"]["test_data_path"]) for run in runs
        },
        "resolved_checkpoint": {
            str(run["result"]["resolved_checkpoint"]) for run in runs
        },
        "resolved_config": {
            str(run["result"]["resolved_config"]) for run in runs
        },
    }
    drift = {name: values for name, values in invariants.items() if len(values) != 1}
    if drift:
        raise ValueError(
            "Cross-run attribution drift detected: "
            + ", ".join(f"{name}={len(values)} values" for name, values in drift.items())
        )


def _load_outcomes(
    runs: Sequence[Mapping[str, Any]], expected_trials: int
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    gallery_digests: Dict[tuple[str, int], set[str]] = {}
    for run in runs:
        run_dir = Path(run["run_dir"])
        frame = pd.read_csv(run_dir / "retrieval_outcomes.csv.gz")
        required = {
            "task",
            "condition",
            "seed",
            "trial",
            "gw_id",
            "source_type",
            "gallery_size",
            *METRICS,
            "coverage",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing outcome columns in {run_dir}: {sorted(missing)}")
        if frame.empty or frame[list(required)].isnull().any().any():
            raise ValueError(f"Missing/NaN outcome values in {run_dir}")
        identity_values = {
            "task": frame["task"].astype(str).unique().tolist(),
            "condition": frame["condition"].astype(str).unique().tolist(),
            "seed": frame["seed"].astype(int).unique().tolist(),
        }
        expected_identity = {
            "task": str(run["manifest"]["task"]),
            "condition": str(run["manifest"]["condition"]),
            "seed": int(run["manifest"]["seed"]),
        }
        for name, values in identity_values.items():
            if values != [expected_identity[name]]:
                raise ValueError(
                    f"Outcome {name} does not match manifest in {run_dir}: {values}"
                )
        if len(frame) != int(run["gallery"]["n_galleries"]):
            raise ValueError(
                f"Outcome row count does not match gallery identity in {run_dir}"
            )
        if str(run["result"]["gallery_identity_sha256"]) != str(
            run["gallery"]["sha256"]
        ):
            raise ValueError(f"Result/gallery digest mismatch in {run_dir}")
        if not np.allclose(frame["coverage"].to_numpy(), 1.0):
            raise ValueError(f"Coverage is below 100% in {run_dir}")
        key_columns = ["seed", "trial", "gw_id", "gallery_size"]
        if frame.duplicated(key_columns).any():
            raise ValueError(f"Duplicate per-query outcome keys in {run_dir}")
        if int(frame["trial"].nunique()) != int(expected_trials):
            raise ValueError(
                f"Expected {expected_trials} trials in {run_dir}, "
                f"found {frame['trial'].nunique()}"
            )
        task = str(frame["task"].iloc[0])
        seed = int(frame["seed"].iloc[0])
        digest = str(run["gallery"]["sha256"])
        gallery_digests.setdefault((task, seed), set()).add(digest)
        frames.append(frame)
    for key, values in gallery_digests.items():
        if len(values) != 1:
            raise ValueError(f"Gallery identity differs across conditions for {key}")
    return pd.concat(frames, ignore_index=True)


def _comparison_reference(task: str, condition: str) -> str | None:
    if task == "operational_nonkn":
        if condition in {"dt_zero", "dt_shared", "coord_shared", "controlled"}:
            return "native"
        if condition != "native":
            return "controlled"
        return None
    return None if condition == "baseline" else "baseline"


def _physics_expectation(task: str, condition: str, source: str) -> str:
    if task != "kn_nuisance_matched":
        return "diagnostic"
    if condition == "inclination_sign_flip":
        return "null_even_inclination_symmetry"
    if source == "bns" and condition in {"spin1z_perm", "spin2z_perm"}:
        return "simulator_null"
    if source == "nsbh" and condition == "spin2z_perm":
        return "simulator_null"
    if source == "nsbh" and condition == "spin1z_perm":
        return "expected_sensitive_bh_spin"
    if condition in {
        "mass1_perm",
        "mass2_perm",
        "inclination_abs_perm",
        "distance_perm",
    }:
        return "expected_sensitive"
    return "diagnostic"


def _merge_paired(
    outcomes: pd.DataFrame,
    *,
    task: str,
    condition: str,
    reference: str,
    source: str,
    gallery_size: int,
    metric: str,
) -> pd.DataFrame:
    subset = outcomes[
        (outcomes["task"] == task)
        & (outcomes["gallery_size"] == gallery_size)
    ]
    if source != "all":
        subset = subset[subset["source_type"] == source]
    keys = ["seed", "trial", "gw_id", "gallery_size", "source_type"]
    left = subset[subset["condition"] == condition][keys + [metric]].rename(
        columns={metric: "condition_value"}
    )
    right = subset[subset["condition"] == reference][keys + [metric]].rename(
        columns={metric: "reference_value"}
    )
    paired = left.merge(right, on=keys, validate="one_to_one")
    if len(paired) != len(left) or len(paired) != len(right):
        raise ValueError(
            f"Incomplete pairing for {task}/{condition}/{reference}/{source}/g{gallery_size}"
        )
    paired["delta"] = paired["condition_value"] - paired["reference_value"]
    return paired


def hierarchical_paired_bootstrap(
    paired: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, float]:
    """Resample GW events, then paired trials, and weight seeds equally."""
    if paired.empty:
        raise ValueError("Cannot bootstrap an empty paired table.")
    if int(n_bootstrap) <= 0:
        raise ValueError("n_bootstrap must be positive.")
    rng = np.random.default_rng(seed)
    by_seed: Dict[int, np.ndarray] = {}
    for seed_value, seed_frame in paired.groupby("seed", sort=True):
        trial_counts = seed_frame.groupby("gw_id", sort=True).size().unique()
        if trial_counts.size != 1:
            raise ValueError("Every GW must have the same number of paired trials.")
        by_seed[int(seed_value)] = np.stack(
            [
                group.sort_values("trial")["delta"].to_numpy(dtype=np.float64)
                for _, group in seed_frame.groupby("gw_id", sort=True)
            ],
            axis=0,
        )
    point_seed = [float(np.mean(matrix)) for matrix in by_seed.values()]
    point = float(np.mean(point_seed))
    samples = np.empty(int(n_bootstrap), dtype=np.float64)
    bootstrap_batch_size = 128
    for start in range(0, int(n_bootstrap), bootstrap_batch_size):
        end = min(start + bootstrap_batch_size, int(n_bootstrap))
        batch_size = end - start
        seed_means = np.empty((batch_size, len(by_seed)), dtype=np.float64)
        for seed_idx, matrix in enumerate(by_seed.values()):
            n_gw, n_trials = matrix.shape
            sampled_gw = rng.integers(
                0, n_gw, size=(batch_size, n_gw), endpoint=False
            )
            sampled_rows = matrix[sampled_gw]
            sampled_trials = rng.integers(
                0,
                n_trials,
                size=(batch_size, n_gw, n_trials),
                endpoint=False,
            )
            resampled = np.take_along_axis(
                sampled_rows, sampled_trials, axis=2
            )
            seed_means[:, seed_idx] = resampled.mean(axis=(1, 2))
        samples[start:end] = seed_means.mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    p_value = min(
        1.0,
        2.0
        * min(
            float(np.mean(samples <= 0.0)),
            float(np.mean(samples >= 0.0)),
        ),
    )
    return {
        "mean_delta": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": p_value,
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values in original order."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def build_comparisons(
    outcomes: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    equivalence_margin: float,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    counter = 0
    for task in sorted(outcomes["task"].unique().tolist()):
        task_frame = outcomes[outcomes["task"] == task]
        conditions = sorted(task_frame["condition"].unique().tolist())
        sources = ["all", *sorted(task_frame["source_type"].unique().tolist())]
        galleries = sorted(task_frame["gallery_size"].unique().tolist())
        for condition in conditions:
            reference = _comparison_reference(task, condition)
            if reference is None or reference not in conditions:
                continue
            for source in sources:
                for gallery_size in galleries:
                    for metric in METRICS:
                        paired = _merge_paired(
                            outcomes,
                            task=task,
                            condition=condition,
                            reference=reference,
                            source=source,
                            gallery_size=int(gallery_size),
                            metric=metric,
                        )
                        stats = hierarchical_paired_bootstrap(
                            paired,
                            n_bootstrap=n_bootstrap,
                            seed=bootstrap_seed + counter,
                        )
                        counter += 1
                        rows.append(
                            {
                                "task": task,
                                "condition": condition,
                                "reference_condition": reference,
                                "source_type": source,
                                "gallery_size": int(gallery_size),
                                "metric": metric,
                                "n_seeds": int(paired["seed"].nunique()),
                                "n_pairs": len(paired),
                                "condition_mean": float(
                                    paired.groupby("seed")["condition_value"].mean().mean()
                                ),
                                "reference_mean": float(
                                    paired.groupby("seed")["reference_value"].mean().mean()
                                ),
                                **stats,
                                "equivalence_margin": float(equivalence_margin),
                                "physics_expectation": _physics_expectation(
                                    task, condition, source
                                ),
                            }
                        )
    adjusted = benjamini_hochberg([row["p_value"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted.tolist()):
        row["p_value_bh"] = float(p_adjusted)
        if (
            row["ci_low"] >= -equivalence_margin
            and row["ci_high"] <= equivalence_margin
        ):
            row["decision"] = "equivalent_within_margin"
        elif row["ci_low"] > 0.0 or row["ci_high"] < 0.0:
            row["decision"] = "detectable_effect"
        else:
            row["decision"] = "inconclusive"
    return rows


def _four_way_factorial_frame(
    outcomes: pd.DataFrame,
    *,
    definition: Mapping[str, str],
    source: str,
    gallery_size: int,
    metric: str,
) -> pd.DataFrame:
    subset = outcomes[
        (outcomes["task"] == "kn_nuisance_matched")
        & (outcomes["gallery_size"] == int(gallery_size))
    ]
    if source != "all":
        subset = subset[subset["source_type"] == source]
    keys = ["seed", "trial", "gw_id", "gallery_size", "source_type"]
    names = {
        "reference_value": "baseline",
        "gw_only_value": str(definition["gw_only"]),
        "optical_only_value": str(definition["optical_only"]),
        "combined_value": str(definition["combined"]),
    }
    merged: pd.DataFrame | None = None
    expected_rows: int | None = None
    for output_name, condition in names.items():
        part = subset[subset["condition"] == condition][keys + [metric]].rename(
            columns={metric: output_name}
        )
        if expected_rows is None:
            expected_rows = len(part)
        elif len(part) != expected_rows:
            raise ValueError(
                f"Incomplete factorial rows for {definition['combined']}/{source}/"
                f"g{gallery_size}/{metric}"
            )
        merged = (
            part
            if merged is None
            else merged.merge(part, on=keys, validate="one_to_one")
        )
    if merged is None or expected_rows is None or len(merged) != expected_rows:
        raise ValueError(f"Incomplete factorial pairing for {definition['combined']}")
    merged["delta"] = (
        merged["reference_value"]
        - merged["gw_only_value"]
        - merged["optical_only_value"]
        + merged["combined_value"]
    )
    return merged


def build_factorial_interactions(
    outcomes: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    equivalence_margin: float,
) -> list[Dict[str, Any]]:
    """Estimate paired GW-by-optical difference-in-differences interactions."""
    available = set(
        outcomes.loc[
            outcomes["task"] == "kn_nuisance_matched", "condition"
        ].astype(str)
    )
    definitions = [
        item for item in FACTORIAL_DEFINITIONS if item["combined"] in available
    ]
    if not definitions:
        return []
    matched = outcomes[outcomes["task"] == "kn_nuisance_matched"]
    sources = ["all", *sorted(matched["source_type"].unique().tolist())]
    galleries = sorted(matched["gallery_size"].unique().tolist())
    rows: list[Dict[str, Any]] = []
    counter = 0
    for definition in definitions:
        for source in sources:
            for gallery_size in galleries:
                for metric in METRICS:
                    paired = _four_way_factorial_frame(
                        outcomes,
                        definition=definition,
                        source=source,
                        gallery_size=int(gallery_size),
                        metric=metric,
                    )
                    stats = hierarchical_paired_bootstrap(
                        paired,
                        n_bootstrap=n_bootstrap,
                        seed=bootstrap_seed + counter,
                    )
                    counter += 1
                    means = paired.groupby("seed")[[
                        "reference_value",
                        "gw_only_value",
                        "optical_only_value",
                        "combined_value",
                    ]].mean().mean()
                    rows.append(
                        {
                            "task": "kn_nuisance_matched",
                            "question": definition["question"],
                            "combined_condition": definition["combined"],
                            "gw_only_condition": definition["gw_only"],
                            "optical_only_condition": definition["optical_only"],
                            "reference_condition": "baseline",
                            "source_type": source,
                            "gallery_size": int(gallery_size),
                            "metric": metric,
                            "n_seeds": int(paired["seed"].nunique()),
                            "n_pairs": len(paired),
                            "reference_mean": float(means["reference_value"]),
                            "gw_only_mean": float(means["gw_only_value"]),
                            "optical_only_mean": float(means["optical_only_value"]),
                            "combined_mean": float(means["combined_value"]),
                            "interaction": stats.pop("mean_delta"),
                            **stats,
                            "equivalence_margin": float(equivalence_margin),
                        }
                    )
    adjusted = benjamini_hochberg([row["p_value"] for row in rows])
    for row, adjusted_value in zip(rows, adjusted.tolist()):
        row["p_value_bh"] = float(adjusted_value)
        if (
            row["ci_low"] >= -equivalence_margin
            and row["ci_high"] <= equivalence_margin
        ):
            row["decision"] = "equivalent_within_margin"
        elif adjusted_value < 0.05 and (
            row["ci_low"] > 0.0 or row["ci_high"] < 0.0
        ):
            row["decision"] = "detectable_interaction_after_bh"
        else:
            row["decision"] = "inconclusive"
    return rows


def build_gallery_strategy_comparisons(
    outcomes: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
    equivalence_margin: float,
) -> list[Dict[str, Any]]:
    """Compare nuisance-nearest and random-same-source baseline galleries."""
    if "kn_same_source_random" not in set(outcomes["task"].astype(str)):
        return []
    sources = ["all", *sorted(outcomes["source_type"].unique().tolist())]
    galleries = sorted(
        outcomes.loc[
            outcomes["task"] == "kn_nuisance_matched", "gallery_size"
        ].unique().tolist()
    )
    keys = ["seed", "trial", "gw_id", "gallery_size", "source_type"]
    rows: list[Dict[str, Any]] = []
    counter = 0
    for source in sources:
        for gallery_size in galleries:
            for metric in METRICS:
                subset = outcomes[outcomes["gallery_size"] == int(gallery_size)]
                if source != "all":
                    subset = subset[subset["source_type"] == source]
                nearest = subset[
                    (subset["task"] == "kn_nuisance_matched")
                    & (subset["condition"] == "baseline")
                ][keys + [metric]].rename(columns={metric: "nearest_value"})
                random_frame = subset[
                    (subset["task"] == "kn_same_source_random")
                    & (subset["condition"] == "baseline")
                ][keys + [metric]].rename(columns={metric: "random_value"})
                paired = nearest.merge(random_frame, on=keys, validate="one_to_one")
                if len(paired) != len(nearest) or len(paired) != len(random_frame):
                    raise ValueError(
                        f"Incomplete gallery-strategy pairing for {source}/"
                        f"g{gallery_size}/{metric}"
                    )
                paired["delta"] = paired["nearest_value"] - paired["random_value"]
                stats = hierarchical_paired_bootstrap(
                    paired,
                    n_bootstrap=n_bootstrap,
                    seed=bootstrap_seed + counter,
                )
                counter += 1
                means = paired.groupby("seed")[["nearest_value", "random_value"]].mean().mean()
                rows.append(
                    {
                        "source_type": source,
                        "gallery_size": int(gallery_size),
                        "metric": metric,
                        "n_seeds": int(paired["seed"].nunique()),
                        "n_pairs": len(paired),
                        "nuisance_nearest_mean": float(means["nearest_value"]),
                        "random_same_source_mean": float(means["random_value"]),
                        "mean_delta_nearest_minus_random": stats.pop("mean_delta"),
                        **stats,
                        "equivalence_margin": float(equivalence_margin),
                    }
                )
    adjusted = benjamini_hochberg([row["p_value"] for row in rows])
    for row, adjusted_value in zip(rows, adjusted.tolist()):
        row["p_value_bh"] = float(adjusted_value)
        if (
            row["ci_low"] >= -equivalence_margin
            and row["ci_high"] <= equivalence_margin
        ):
            row["decision"] = "equivalent_within_margin"
        elif adjusted_value < 0.05 and (
            row["ci_low"] > 0.0 or row["ci_high"] < 0.0
        ):
            row["decision"] = "detectable_after_bh"
        else:
            row["decision"] = "inconclusive"
    return rows


def _distractor_physical_similarity(
    runs: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    mismatch_columns = (
        "delta_effective_mej_dyn",
        "delta_effective_mej_wind",
        "delta_effective_mej_total",
        "delta_abs_costheta",
    )
    for run in runs:
        result = run["result"]
        if result["task"] not in {
            "kn_nuisance_matched",
            "kn_same_source_random",
        } or result["condition"]["name"] != "baseline":
            continue
        frame = pd.read_csv(Path(run["run_dir"]) / "candidate_scores.csv.gz")
        frame = frame[frame["is_positive"] == 0]
        for (source, gallery_size), group in frame.groupby(
            ["source_type", "gallery_size"], sort=True
        ):
            for mismatch in mismatch_columns:
                values = group[mismatch].to_numpy(dtype=np.float64)
                rows.append(
                    {
                        "task": result["task"],
                        "seed": int(result["seed"]),
                        "source_type": source,
                        "gallery_size": int(gallery_size),
                        "mismatch": mismatch,
                        "n_candidates": int(values.size),
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "p90": float(np.percentile(values, 90.0)),
                        "zero_fraction": float(np.mean(values == 0.0)),
                    }
                )
    return rows


def _distractor_nuisance_balance(
    runs: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for run in runs:
        result = run["result"]
        if result["task"] not in {
            "kn_nuisance_matched",
            "kn_same_source_random",
        } or result["condition"]["name"] != "baseline":
            continue
        frame = pd.read_csv(Path(run["run_dir"]) / "match_balance.csv")
        for item in frame.to_dict("records"):
            rows.append(
                {
                    "task": result["task"],
                    "seed": int(result["seed"]),
                    **item,
                }
            )
    return rows


def _per_seed_metrics(outcomes: pd.DataFrame) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    sources = ["all", *sorted(outcomes["source_type"].unique().tolist())]
    for source in sources:
        source_frame = outcomes if source == "all" else outcomes[outcomes["source_type"] == source]
        for keys, frame in source_frame.groupby(
            ["task", "condition", "gallery_size", "seed"], sort=True
        ):
            task, condition, gallery_size, seed = keys
            for metric in METRICS:
                rows.append(
                    {
                        "task": task,
                        "condition": condition,
                        "source_type": source,
                        "gallery_size": int(gallery_size),
                        "metric": metric,
                        "seed": int(seed),
                        "n_rows": len(frame),
                        "value": float(frame[metric].mean()),
                    }
                )
    return rows


def _within_query_spearman(
    frame: pd.DataFrame, mismatch: str
) -> Dict[str, float | int]:
    """Average query-conditional rank correlations with GW as inference unit."""
    query_rows: list[tuple[int, float]] = []
    for (gw_id, _trial), group in frame.groupby(["gw_id", "trial"], sort=True):
        if group["score"].nunique() < 2 or group[mismatch].nunique() < 2:
            continue
        rho, _ = spearmanr(group["score"], group[mismatch])
        if np.isfinite(rho):
            query_rows.append((int(gw_id), float(rho)))
    if not query_rows:
        return {
            "n_query_trials": 0,
            "n_gw": 0,
            "spearman_rho": float("nan"),
            "p_value": float("nan"),
        }
    query_frame = pd.DataFrame(query_rows, columns=["gw_id", "rho"])
    gw_rho = query_frame.groupby("gw_id", sort=True)["rho"].mean().to_numpy()
    if gw_rho.size < 2 or np.allclose(gw_rho, 0.0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(gw_rho, alternative="two-sided").pvalue)
    return {
        "n_query_trials": len(query_frame),
        "n_gw": int(gw_rho.size),
        "spearman_rho": float(np.mean(gw_rho)),
        "p_value": p_value,
    }


def _physical_correlations(runs: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    mismatch_columns = (
        "delta_effective_mej_dyn",
        "delta_effective_mej_wind",
        "delta_effective_mej_total",
        "delta_abs_costheta",
    )
    for run in runs:
        result = run["result"]
        if result["task"] not in {
            "kn_nuisance_matched",
            "kn_same_source_random",
        }:
            continue
        path = Path(run["run_dir"]) / "candidate_scores.csv.gz"
        frame = pd.read_csv(path)
        frame = frame[frame["is_positive"] == 0]
        for (source, gallery_size), group in frame.groupby(
            ["source_type", "gallery_size"], sort=True
        ):
            for mismatch in mismatch_columns:
                stats = _within_query_spearman(group, mismatch)
                rows.append(
                    {
                        "task": result["task"],
                        "condition": result["condition"]["name"],
                        "seed": result["seed"],
                        "source_type": source,
                        "gallery_size": int(gallery_size),
                        "mismatch": mismatch,
                        "n_candidates": len(group),
                        "correlation_method": (
                            "mean within-query Spearman; GW-mean Wilcoxon"
                        ),
                        **stats,
                    }
                )
    return rows


def _gate_checks(outcomes: pd.DataFrame) -> list[Dict[str, Any]]:
    checks: list[Dict[str, Any]] = []
    operational = outcomes[outcomes["task"] == "operational_nonkn"]
    for seed in sorted(operational["seed"].unique().tolist()):
        frame = operational[
            (operational["seed"] == seed)
            & (operational["source_type"].isin(["bns", "nsbh"]))
        ]
        native = frame[frame["condition"] == "native"]["recall_at_1"].mean()
        dt_zero = frame[frame["condition"] == "dt_zero"]["recall_at_1"].mean()
        checks.append(
            {
                "check": "dt_zero_not_better_than_native_r1",
                "seed": int(seed),
                "passed": bool(dt_zero <= native),
                "native_r1": float(native),
                "dt_zero_r1": float(dt_zero),
            }
        )
    return checks


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Create the aggregation directory or explicitly allow atomic file replacement."""
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Aggregation output is not a directory: {output_dir}")
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite aggregation: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-trials", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260728)
    parser.add_argument("--equivalence-margin", type=float, default=0.01)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Allow atomically replacing aggregation files in an existing "
            "output directory. Existing unrelated files are preserved."
        ),
    )
    args = parser.parse_args()

    input_roots = [Path(value).expanduser().resolve() for value in args.input_root]
    output_dir = Path(args.output_dir).expanduser().resolve()
    prepare_output_dir(output_dir, overwrite=args.overwrite)

    runs = discover_runs(input_roots)
    validate_run_matrix(runs)
    outcomes = _load_outcomes(runs, args.expected_trials)
    comparisons = build_comparisons(
        outcomes,
        n_bootstrap=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        equivalence_margin=args.equivalence_margin,
    )
    factorial = build_factorial_interactions(
        outcomes,
        n_bootstrap=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed + 1000000,
        equivalence_margin=args.equivalence_margin,
    )
    gallery_strategy = build_gallery_strategy_comparisons(
        outcomes,
        n_bootstrap=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed + 2000000,
        equivalence_margin=args.equivalence_margin,
    )
    per_seed = _per_seed_metrics(outcomes)
    correlations = _physical_correlations(runs)
    distractor_physical = _distractor_physical_similarity(runs)
    distractor_nuisance = _distractor_nuisance_balance(runs)
    gate_checks = _gate_checks(outcomes)

    write_csv_atomic(
        output_dir / "attribution_comparisons.csv",
        comparisons,
        COMPARISON_FIELDS,
    )
    write_csv_atomic(
        output_dir / "per_seed_metrics.csv", per_seed, PER_SEED_FIELDS
    )
    write_csv_atomic(
        output_dir / "physical_score_correlations.csv",
        correlations,
        CORRELATION_FIELDS,
    )
    write_csv_atomic(
        output_dir / "factorial_interactions.csv",
        factorial,
        FACTORIAL_FIELDS,
    )
    write_csv_atomic(
        output_dir / "gallery_strategy_comparisons.csv",
        gallery_strategy,
        GALLERY_STRATEGY_FIELDS,
    )
    write_csv_atomic(
        output_dir / "distractor_physical_similarity.csv",
        distractor_physical,
        DISTRACTOR_PHYSICAL_FIELDS,
    )
    write_csv_atomic(
        output_dir / "distractor_nuisance_balance.csv",
        distractor_nuisance,
        DISTRACTOR_NUISANCE_FIELDS,
    )
    write_json_atomic(
        output_dir / "aggregation_summary.json",
        {
            "input_roots": [str(path) for path in input_roots],
            "n_runs": len(runs),
            "seeds": sorted(int(value) for value in outcomes["seed"].unique()),
            "tasks": sorted(outcomes["task"].unique().tolist()),
            "n_outcomes": len(outcomes),
            "n_comparisons": len(comparisons),
            "n_factorial_interactions": len(factorial),
            "n_gallery_strategy_comparisons": len(gallery_strategy),
            "bootstrap_replicates": args.bootstrap_replicates,
            "equal_seed_weighting": True,
            "equivalence_margin": args.equivalence_margin,
            "multiple_testing": "Benjamini-Hochberg across all reported comparisons",
            "gate_checks": gate_checks,
        },
    )
    print(f"Wrote attribution aggregation to {output_dir}")


if __name__ == "__main__":
    main()
