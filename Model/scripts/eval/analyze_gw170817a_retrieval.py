#!/usr/bin/env python3
"""Summarize the crossed GW170817A LSST scenario retrieval experiment."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd

METRICS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")


def _decode(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def load_optical_metadata(h5_path: Path) -> pd.DataFrame:
    with h5py.File(h5_path, "r") as handle:
        optical = handle["events/optical_data"]
        parent = np.asarray(optical["parent_gw_idx"], dtype=np.int64)
        gw_times = np.asarray(handle["events/gw_data/event_time_mjd"], dtype=float)
        frame = pd.DataFrame(
            {
                "positive_optical_index": np.arange(parent.size, dtype=np.int64),
                "scenario_id": np.asarray(optical["scenario_id"], dtype=np.int64),
                "coordinate_id": np.asarray(optical["coordinate_id"], dtype=np.int64),
                "is_true_position": np.asarray(optical["is_true_position"], dtype=bool),
                "skymap_credible_level": np.asarray(
                    optical["skymap_credible_level"], dtype=float
                ),
                "n_observations": np.asarray(optical["n_observations"], dtype=np.int64),
                "too_nobs": np.asarray(optical["too_nobs"], dtype=np.int64),
                "first_detection_latency_days": (
                    np.asarray(optical["first_detection_mjd"], dtype=float)
                    - gw_times[parent]
                ),
            }
        )
        frame["dataset_mode"] = str(_decode(handle.attrs.get("dataset_mode", "")))
    if frame["positive_optical_index"].duplicated().any():
        raise ValueError("HDF5 optical indices are not unique.")
    return frame


def load_and_collapse(outcomes_path: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    with gzip.open(outcomes_path, "rt", newline="") as stream:
        outcomes = pd.read_csv(stream)
    required = {"model", "gallery_size", "positive_optical_index", "repeat", *METRICS}
    missing = sorted(required.difference(outcomes.columns))
    if missing:
        raise ValueError(f"Outcome CSV is missing required columns: {missing}")
    joined = outcomes.merge(
        metadata, on="positive_optical_index", how="left", validate="many_to_one"
    )
    if joined[["scenario_id", "coordinate_id"]].isna().any().any():
        raise ValueError(
            "Outcome CSV contains positive indices absent from the event HDF5."
        )

    keys = ["model", "gallery_size", "scenario_id", "coordinate_id"]
    repeat_counts = joined.groupby(keys, observed=True)["repeat"].nunique()
    if repeat_counts.nunique() != 1:
        raise ValueError(
            "Every model/gallery/scenario/coordinate cell must have equal repeat coverage."
        )
    collapsed = joined.groupby(keys, as_index=False, observed=True).agg(
        **{metric: (metric, "mean") for metric in METRICS},
        is_true_position=("is_true_position", "first"),
        skymap_credible_level=("skymap_credible_level", "first"),
        n_observations=("n_observations", "first"),
        too_nobs=("too_nobs", "first"),
        first_detection_latency_days=("first_detection_latency_days", "first"),
        repeats=("repeat", "nunique"),
    )
    return collapsed


def _balanced_matrix(
    group: pd.DataFrame, metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scenarios = np.sort(group["scenario_id"].unique())
    coordinates = np.sort(group["coordinate_id"].unique())
    matrix = group.pivot(
        index="scenario_id", columns="coordinate_id", values=metric
    ).reindex(index=scenarios, columns=coordinates)
    if matrix.isna().any().any():
        raise ValueError(
            "The event analysis requires a complete scenario x coordinate panel."
        )
    return matrix.to_numpy(dtype=float), scenarios, coordinates


def two_way_bootstrap(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    draws: Tuple[np.ndarray, np.ndarray] | None = None,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Independently resample scenario rows and coordinate columns with replacement."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) <= 0:
        raise ValueError("values must be a non-empty scenario x coordinate matrix.")
    if draws is None:
        rng = np.random.default_rng(seed)
        scenario_draws = rng.integers(
            values.shape[0], size=(n_bootstrap, values.shape[0])
        )
        coordinate_draws = rng.integers(
            values.shape[1], size=(n_bootstrap, values.shape[1])
        )
    else:
        scenario_draws, coordinate_draws = draws
        if (
            scenario_draws.shape[0] != n_bootstrap
            or coordinate_draws.shape[0] != n_bootstrap
        ):
            raise ValueError("Bootstrap draw count does not match n_bootstrap.")
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        samples[index] = values[
            np.ix_(scenario_draws[index], coordinate_draws[index])
        ].mean()
    return samples, (scenario_draws, coordinate_draws)


def _interval(samples: np.ndarray) -> Tuple[float, float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def summarize_overall(
    collapsed: pd.DataFrame, *, n_bootstrap: int, seed: int, reference_model: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    delta_rows = []
    for gallery_size in sorted(collapsed["gallery_size"].unique()):
        gallery = collapsed[collapsed["gallery_size"] == gallery_size]
        models = list(dict.fromkeys(gallery["model"].tolist()))
        if reference_model not in models:
            raise ValueError(
                f"Reference model {reference_model!r} is absent at gallery size {gallery_size}."
            )
        for metric_index, metric in enumerate(METRICS):
            matrices: Dict[str, np.ndarray] = {}
            bootstraps: Dict[str, np.ndarray] = {}
            shared_draws = None
            for model_index, model in enumerate(models):
                matrix, _, _ = _balanced_matrix(
                    gallery[gallery["model"] == model], metric
                )
                matrices[model] = matrix
                samples, shared_draws = two_way_bootstrap(
                    matrix,
                    n_bootstrap=n_bootstrap,
                    seed=seed + 1009 * metric_index + int(gallery_size),
                    draws=shared_draws,
                )
                bootstraps[model] = samples
                low, high = _interval(samples)
                overall_rows.append(
                    {
                        "model": model,
                        "gallery_size": int(gallery_size),
                        "metric": metric,
                        "micro_mean": float(matrix.mean()),
                        "scenario_macro_mean": float(matrix.mean(axis=1).mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "n_scenarios": int(matrix.shape[0]),
                        "n_coordinates": int(matrix.shape[1]),
                    }
                )
            reference = matrices[reference_model]
            for baseline in models:
                if baseline == reference_model:
                    continue
                if matrices[baseline].shape != reference.shape:
                    raise ValueError(
                        "Paired model matrices do not share the same crossed panel."
                    )
                delta_samples = bootstraps[reference_model] - bootstraps[baseline]
                low, high = _interval(delta_samples)
                delta_rows.append(
                    {
                        "reference_model": reference_model,
                        "baseline_model": baseline,
                        "gallery_size": int(gallery_size),
                        "metric": metric,
                        "delta": float((reference - matrices[baseline]).mean()),
                        "ci95_low": low,
                        "ci95_high": high,
                        "probability_delta_gt_zero": float(
                            np.mean(delta_samples > 0.0)
                        ),
                    }
                )
    return pd.DataFrame(overall_rows), pd.DataFrame(delta_rows)


def _quartile_labels(series: pd.Series, prefix: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Cannot stratify {prefix}: no finite values.")
    cuts = np.quantile(finite, [0.25, 0.5, 0.75])
    bins = np.searchsorted(cuts, values, side="right") + 1
    labels = np.asarray([f"{prefix}_Q{value}" for value in bins], dtype=object)
    labels[~np.isfinite(values)] = f"{prefix}_missing"
    return pd.Series(labels, index=series.index)


def summarize_conditions(collapsed: pd.DataFrame) -> pd.DataFrame:
    frame = collapsed.copy()
    frame["credible_band"] = np.where(
        frame["skymap_credible_level"] <= 0.5, "credible_le_0.5", "credible_gt_0.5"
    )
    frame["nobs_quartile"] = _quartile_labels(frame["n_observations"], "nobs")
    frame["latency_quartile"] = _quartile_labels(
        frame["first_detection_latency_days"], "latency"
    )
    frame["too_nobs_group"] = frame["too_nobs"].astype(str)
    rows = []
    for condition in (
        "credible_band",
        "nobs_quartile",
        "latency_quartile",
        "too_nobs_group",
    ):
        grouped = frame.groupby(["model", "gallery_size", condition], observed=True)
        for keys, group in grouped:
            model, gallery_size, level = keys
            for metric in METRICS:
                rows.append(
                    {
                        "model": model,
                        "gallery_size": int(gallery_size),
                        "condition": condition,
                        "level": str(level),
                        "metric": metric,
                        "mean": float(group[metric].mean()),
                        "n_cells": int(len(group)),
                    }
                )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--test-h5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-model", default="Full")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=170817)
    parser.add_argument("--expected-scenarios", type=int, default=10)
    parser.add_argument("--expected-coordinates", type=int, default=50)
    parser.add_argument("--expected-repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    metadata = load_optical_metadata(args.test_h5)
    collapsed = load_and_collapse(args.outcomes, metadata)
    observed_shape = (
        int(collapsed["scenario_id"].nunique()),
        int(collapsed["coordinate_id"].nunique()),
        sorted(int(value) for value in collapsed["repeats"].unique()),
    )
    expected_shape = (
        int(args.expected_scenarios),
        int(args.expected_coordinates),
        [int(args.expected_repeats)],
    )
    if observed_shape != expected_shape:
        raise ValueError(
            f"Expected scenario/coordinate/repeat panel {expected_shape}, got {observed_shape}."
        )
    overall, deltas = summarize_overall(
        collapsed,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        reference_model=args.reference_model,
    )
    conditions = summarize_conditions(collapsed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collapsed.to_csv(args.output_dir / "collapsed_coordinate_metrics.csv", index=False)
    overall.to_csv(args.output_dir / "overall_metrics.csv", index=False)
    deltas.to_csv(args.output_dir / "paired_deltas.csv", index=False)
    conditions.to_csv(args.output_dir / "condition_metrics.csv", index=False)
    payload = {
        "analysis": "gw170817a_lsst_scenarios_two_way_bootstrap_v1",
        "test_h5": str(args.test_h5),
        "outcomes": str(args.outcomes),
        "reference_model": args.reference_model,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "n_scenarios": int(collapsed["scenario_id"].nunique()),
        "n_coordinates": int(collapsed["coordinate_id"].nunique()),
        "repeats_per_cell": sorted(
            int(value) for value in collapsed["repeats"].unique()
        ),
        "overall": overall.to_dict(orient="records"),
        "paired_deltas": deltas.to_dict(orient="records"),
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote GW170817A retrieval summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
