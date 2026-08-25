#!/usr/bin/env python3
"""Evaluate GW retrieval in galleries mixing KN and non-KN distractors."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import random
import sys
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

from mixed_retrieval import (
    MIXED_SCOPES,
    allocate_mixed_negative_counts,
    build_mixed_galleries,
    mixed_gallery_identity_digest,
    mixed_score_outcome,
    select_source_balanced_queries,
)
from retrieval_gallery import (
    build_synthetic_time_sky_candidate_sequences,
    plot_retrieval_curves,
)

from scripts.eval import eval_retrieval_comparison as base
from scripts.eval import mixed_retrieval_analysis as analysis

METRIC_COLUMNS = analysis.METRIC_COLUMNS

SOURCE_AGGREGATIONS = ("pooled", "source_macro")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _parse_sizes(value: Any) -> list[int]:
    if isinstance(value, str):
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        values = [int(item) for item in value]
    sizes = sorted(set(values))
    if not sizes or min(sizes) < 3:
        raise ValueError("mixed gallery_sizes must contain values >= 3")
    return sizes


def _resolve_path(cfg_dir: Path, value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    path = (cfg_dir / path).resolve() if not path.is_absolute() else path.resolve()
    jobfs_dir = os.environ.get("JOBFS_DIR")
    if jobfs_dir and (Path(jobfs_dir) / path.name).is_file():
        return str(Path(jobfs_dir) / path.name)
    return str(path)


def _normalise_config(raw: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    cfg_dir = config_path.parent
    cfg = dict(raw)
    cfg["requested_test_data_path"] = str(cfg.get("test_data_path", ""))
    cfg["requested_neg_data_path"] = str(cfg.get("neg_data_path", ""))
    cfg["test_data_path"] = _resolve_path(cfg_dir, cfg.get("test_data_path"))
    cfg["neg_data_path"] = _resolve_path(cfg_dir, cfg.get("neg_data_path"))
    cfg["output_dir"] = _resolve_path(cfg_dir, cfg.get("output_dir"))
    cfg["gallery_sizes"] = _parse_sizes(
        cfg.get("gallery_sizes", [16, 32, 64, 128, 500, 1000])
    )
    cfg["gallery_trials"] = int(cfg.get("gallery_trials", 10))
    cfg["queries_per_source"] = int(cfg.get("queries_per_source", 256))
    cfg["seed"] = int(cfg.get("seed", 42))
    cfg["kn_fraction"] = float(cfg.get("kn_fraction", 0.25))
    cfg["kn_candidate_mode"] = str(cfg.get("kn_candidate_mode", "kn_random")).lower()
    cfg["n_neg_samples"] = int(cfg.get("n_neg_samples", 500000))
    cfg["neg_group"] = str(cfg.get("neg_group", "ELASTICC/optical_data"))
    cfg["batch_size"] = int(cfg.get("batch_size", 512))
    cfg["num_workers"] = int(cfg.get("num_workers", 2))
    cfg["opt_ref_start"] = float(cfg.get("opt_ref_start", -0.1))
    cfg["opt_ref_end"] = float(cfg.get("opt_ref_end", 0.2))
    cfg["opt_n_ref"] = int(cfg.get("opt_n_ref", 64))
    cfg["device"] = str(cfg.get("device", "cuda"))
    cfg["amp_dtype"] = str(cfg.get("amp_dtype", "bf16"))
    cfg["nonkn_training_empirical_fraction"] = float(
        cfg.get("nonkn_training_empirical_fraction", 0.5)
    )
    cfg["nonkn_uniform_time_window_days"] = float(
        cfg.get("nonkn_uniform_time_window_days", 30.0)
    )
    cfg["candidate_credible_level_max"] = float(
        cfg.get("candidate_credible_level_max", 0.9)
    )
    cfg["conditions"] = [
        str(item)
        for item in cfg.get("conditions", ["training_aligned", "positive_shared"])
    ]
    unknown = set(cfg["conditions"]) - {"training_aligned", "positive_shared"}
    if unknown:
        raise ValueError(f"Unsupported conditions: {sorted(unknown)}")
    cfg["primary_condition"] = str(cfg.get("primary_condition", "training_aligned"))
    cfg["primary_scope"] = str(cfg.get("primary_scope", "all"))
    if cfg["primary_condition"] not in cfg["conditions"]:
        raise ValueError("primary_condition must be included in conditions")
    if cfg["primary_scope"] not in MIXED_SCOPES:
        raise ValueError(f"primary_scope must be one of {MIXED_SCOPES}")
    cfg["bootstrap_samples"] = int(cfg.get("bootstrap_samples", 5000))
    cfg["bootstrap_seed"] = int(cfg.get("bootstrap_seed", int(cfg["seed"]) + 17))
    if cfg["bootstrap_samples"] < 1:
        raise ValueError("bootstrap_samples must be >= 1")

    primary = dict(cfg.get("primary_metric", {}))
    primary["name"] = str(primary.get("name", "training_selection_score"))
    primary["source_aggregation"] = str(
        primary.get("source_aggregation", "source_macro")
    )
    primary["metric_weights"] = {
        str(key): float(value)
        for key, value in primary.get(
            "metric_weights", {"mrr": 0.8, "recall_at_1": 0.2}
        ).items()
    }
    primary["gallery_size_weights"] = {
        int(key): float(value)
        for key, value in primary.get(
            "gallery_size_weights",
            {100: 0.10, 500: 0.15, 1000: 0.20, 2000: 0.25, 5000: 0.30},
        ).items()
    }
    if primary["source_aggregation"] != "source_macro":
        raise ValueError("primary_metric.source_aggregation must be source_macro")
    if set(primary["metric_weights"]) - set(METRIC_COLUMNS):
        raise ValueError("primary_metric contains an unsupported metric")
    if not set(primary["gallery_size_weights"]).issubset(cfg["gallery_sizes"]):
        raise ValueError("primary_metric gallery sizes must be evaluated")
    for key in ("metric_weights", "gallery_size_weights"):
        if not np.isclose(sum(primary[key].values()), 1.0):
            raise ValueError(f"primary_metric.{key} weights must sum to 1")
    cfg["primary_metric"] = primary

    redshift = dict(cfg.get("redshift_analysis", {}))
    redshift["enabled"] = bool(redshift.get("enabled", False))
    if redshift["enabled"]:
        redshift["catalogs"] = base._normalize_redshift_catalog_paths(
            redshift.get("catalogs"), cfg_dir
        )
        redshift["bin_edges"], redshift["bin_labels"] = (
            base._normalize_redshift_bin_config(
                redshift.get("bin_edges"), redshift.get("bin_labels")
            )
        )
        redshift["primary_gallery_size"] = int(
            redshift.get("primary_gallery_size", 1000)
        )
        if redshift["primary_gallery_size"] not in cfg["gallery_sizes"]:
            raise ValueError("redshift primary_gallery_size must be evaluated")
        redshift["aggregations"] = [
            str(value)
            for value in redshift.get("aggregations", list(SOURCE_AGGREGATIONS))
        ]
        unknown_aggregations = set(redshift["aggregations"]) - set(SOURCE_AGGREGATIONS)
        if unknown_aggregations:
            raise ValueError(
                f"Unsupported redshift aggregations: {sorted(unknown_aggregations)}"
            )
        redshift["min_queries_per_bin_pooled"] = int(
            redshift.get("min_queries_per_bin_pooled", 30)
        )
        redshift["min_queries_per_bin_source"] = int(
            redshift.get("min_queries_per_bin_source", 10)
        )
        redshift["validate_scalars"] = bool(redshift.get("validate_scalars", True))
    cfg["redshift_analysis"] = redshift
    if not cfg["test_data_path"] or not cfg["neg_data_path"] or not cfg["output_dir"]:
        raise ValueError("test_data_path, neg_data_path and output_dir are required")
    if not 0.0 <= cfg["nonkn_training_empirical_fraction"] <= 1.0:
        raise ValueError("nonkn_training_empirical_fraction must be in [0, 1]")
    return cfg


def _load_kn_metadata(test_path: str) -> dict[str, Any]:
    with h5py.File(test_path, "r") as handle:
        opt = handle["events/optical_data"]
        gw = handle["events/gw_data"]
        parent = np.asarray(opt["parent_gw_idx"][:], dtype=np.int64)
        first = np.asarray(opt["first_detection_mjd"][:], dtype=np.float64)
        event_time = np.asarray(gw["event_time_mjd"][:], dtype=np.float64)
        source_raw = gw["source_type"][:]
        scalars = np.asarray(gw["scalars"][:], dtype=np.float64)
    source = [
        (
            item.decode("utf-8", errors="ignore").strip().lower()
            if isinstance(item, (bytes, np.bytes_))
            else str(item).strip().lower()
        )
        for item in source_raw
    ]
    positive_map: dict[int, list[int]] = {}
    for row, gw_id in enumerate(parent.tolist()):
        positive_map.setdefault(int(gw_id), []).append(int(row))
    return {
        "parent": parent,
        "first_detection_mjd": first,
        "event_time_mjd": event_time,
        "source_types": source,
        "gw_scalars": scalars,
        "positive_map": {
            key: np.asarray(value, dtype=np.int64)
            for key, value in positive_map.items()
        },
    }


def _optical_nuisance_features(
    bank: Mapping[str, torch.Tensor], metadata: Mapping[str, Any]
) -> np.ndarray:
    times = bank["times"].numpy()
    masks = bank["masks"].numpy()
    parent = np.asarray(metadata["parent"], dtype=np.int64)
    scalars = np.asarray(metadata["gw_scalars"], dtype=np.float64)
    first = np.asarray(metadata["first_detection_mjd"], dtype=np.float64)
    event = np.asarray(metadata["event_time_mjd"], dtype=np.float64)
    features = np.zeros((parent.size, 5), dtype=np.float64)
    for idx, gw_id in enumerate(parent.tolist()):
        valid = masks[idx] > 0
        valid_rows = np.any(valid, axis=-1)
        observed_times = times[idx, valid_rows]
        span = float(np.ptp(observed_times)) if observed_times.size > 1 else 0.0
        features[idx] = (
            np.log10(max(float(scalars[gw_id, 5]), 1e-8)),
            abs(float(first[idx]) - float(event[gw_id])),
            np.log1p(np.count_nonzero(valid)),
            np.count_nonzero(np.any(valid, axis=0)) / float(masks.shape[-1]),
            np.log1p(max(span, 0.0)),
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite KN nuisance features")
    return features


def _apply_training_aligned_nonkn_times(
    sequences: Mapping[tuple[int, int], dict[str, np.ndarray]],
    empirical_dt: np.ndarray,
    *,
    fraction: float,
    window_days: float,
    seed: int,
) -> None:
    empirical = np.asarray(empirical_dt, dtype=np.float64)
    empirical = empirical[np.isfinite(empirical) & (empirical >= 0.0)]
    if empirical.size == 0 and fraction > 0.0:
        raise ValueError("No finite empirical KN delays available")
    for (trial, gw_id), sequence in sequences.items():
        n_values = int(np.asarray(sequence["candidate_indices"]).size)
        rng = np.random.default_rng(
            int(seed) + 433494437 * int(trial) + 2971215073 * int(gw_id)
        )
        n_empirical = int(np.floor(n_values * float(fraction) + 0.5))
        values = np.concatenate(
            [
                rng.choice(empirical, size=n_empirical, replace=True),
                rng.uniform(0.0, float(window_days), size=n_values - n_empirical),
            ]
        ).astype(np.float32)
        sequence["abs_dt_days"] = values[rng.permutation(n_values)]


def _candidate_dt(metadata: Mapping[str, Any], indices: np.ndarray) -> np.ndarray:
    parent = np.asarray(metadata["parent"], dtype=np.int64)[indices]
    first = np.asarray(metadata["first_detection_mjd"], dtype=np.float64)[indices]
    event = np.asarray(metadata["event_time_mjd"], dtype=np.float64)[parent]
    return np.abs(first - event).astype(np.float32)


def _condition_candidate_features(
    condition: str,
    *,
    positive_coordinate: np.ndarray,
    positive_dt_days: np.ndarray,
    kn_dt_days: np.ndarray,
    nonkn_synthetic_coordinates: np.ndarray,
    nonkn_training_aligned_dt_days: np.ndarray,
    n_kn: int,
    n_nonkn: int,
) -> dict[str, np.ndarray | None]:
    """Return candidate sky/time features for one evaluation condition."""
    positive_coordinate = np.asarray(positive_coordinate, dtype=np.float32).reshape(
        1, 2
    )
    positive_dt_days = np.asarray(positive_dt_days, dtype=np.float32).reshape(1)
    if condition == "positive_shared":
        return {
            "positive_coordinates": positive_coordinate,
            "kn_coordinates": np.repeat(positive_coordinate, int(n_kn), axis=0),
            "nonkn_coordinates": np.repeat(positive_coordinate, int(n_nonkn), axis=0),
            "kn_dt_days": np.full(int(n_kn), positive_dt_days[0], dtype=np.float32),
            "nonkn_dt_days": np.full(
                int(n_nonkn), positive_dt_days[0], dtype=np.float32
            ),
        }
    if condition != "training_aligned":
        raise ValueError(f"Unsupported condition: {condition}")
    return {
        "positive_coordinates": None,
        "kn_coordinates": np.repeat(positive_coordinate, int(n_kn), axis=0),
        "nonkn_coordinates": np.asarray(nonkn_synthetic_coordinates, dtype=np.float32),
        "kn_dt_days": np.asarray(kn_dt_days, dtype=np.float32),
        "nonkn_dt_days": np.asarray(nonkn_training_aligned_dt_days, dtype=np.float32),
    }


def _validate_redshift_query_metadata(
    query_gw: Sequence[int],
    source_types: Sequence[str],
    redshift_metadata: Mapping[int, Mapping[str, Any]],
    *,
    bin_edges: Sequence[float],
    bin_labels: Sequence[str],
    min_pooled: int,
    min_per_source: int,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Validate query redshift coverage and return per-query metadata and bin counts."""
    query_rows: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    outside: list[tuple[int, float]] = []
    for gw_id in query_gw:
        if int(gw_id) not in redshift_metadata:
            missing.append(int(gw_id))
            continue
        redshift = float(redshift_metadata[int(gw_id)]["redshift"])
        bin_index = base._find_redshift_bin(redshift, list(bin_edges))
        if not np.isfinite(redshift) or bin_index < 0:
            outside.append((int(gw_id), redshift))
            continue
        query_rows[int(gw_id)] = {
            "redshift": redshift,
            "redshift_bin_index": int(bin_index),
            "redshift_bin_label": str(bin_labels[bin_index]),
            "source": str(source_types[int(gw_id)]).lower(),
        }
    if missing:
        raise ValueError(
            f"Missing redshift metadata for {len(missing)} selected GW IDs; "
            f"examples={missing[:5]}"
        )
    if outside:
        raise ValueError(
            f"{len(outside)} selected GW redshifts fall outside configured bins; "
            f"examples={outside[:5]}"
        )

    counts: list[dict[str, Any]] = []
    for bin_index, label in enumerate(bin_labels):
        in_bin = [
            row for row in query_rows.values() if row["redshift_bin_index"] == bin_index
        ]
        pooled = len(in_bin)
        counts.append(
            {
                "redshift_bin_index": int(bin_index),
                "redshift_bin_label": str(label),
                "source": "pooled",
                "n_queries": int(pooled),
            }
        )
        if pooled < int(min_pooled):
            raise ValueError(
                f"Redshift bin {label!r} has {pooled} pooled queries; "
                f"minimum is {min_pooled}"
            )
        for source in sorted({row["source"] for row in query_rows.values()}):
            source_count = sum(row["source"] == source for row in in_bin)
            counts.append(
                {
                    "redshift_bin_index": int(bin_index),
                    "redshift_bin_label": str(label),
                    "source": source,
                    "n_queries": int(source_count),
                }
            )
            if source_count < int(min_per_source):
                raise ValueError(
                    f"Redshift bin {label!r} has {source_count} {source} queries; "
                    f"minimum is {min_per_source}"
                )
    return query_rows, counts


def _load_query_redshifts(
    cfg: Mapping[str, Any],
    query_gw: Sequence[int],
    source_types: Sequence[str],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    redshift_cfg = cfg["redshift_analysis"]
    if not redshift_cfg.get("enabled", False):
        return {}, []
    metadata = base._build_redshift_metadata_from_catalogs(
        cfg["test_data_path"],
        redshift_cfg["catalogs"],
        validate_scalars=redshift_cfg["validate_scalars"],
    )
    return _validate_redshift_query_metadata(
        query_gw,
        source_types,
        metadata,
        bin_edges=redshift_cfg["bin_edges"],
        bin_labels=redshift_cfg["bin_labels"],
        min_pooled=redshift_cfg["min_queries_per_bin_pooled"],
        min_per_source=redshift_cfg["min_queries_per_bin_source"],
    )


def _random_baseline(n_candidates: int) -> dict[str, float]:
    n = int(n_candidates)
    return {
        "recall_at_1": 1.0 / n,
        "recall_at_5": min(5, n) / n,
        "recall_at_10": min(10, n) / n,
        "mrr": float(np.sum(1.0 / np.arange(1, n + 1))) / n,
    }


def _manifest_payload(
    galleries: Mapping[tuple[int, int, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for (size, trial, gw_id), spec in sorted(galleries.items()):
        rows.append(
            {"gallery_size": size, "trial": trial, "gw_id": gw_id, **dict(spec)}
        )
    return rows


def _write_gzip_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump(value, handle, default=_json_default, separators=(",", ":"))
    os.replace(temporary, path)


def _prepare_output(path: Path, *, resume: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [item for item in path.iterdir() if not item.name.endswith(".tmp")]
    if existing and not resume:
        raise FileExistsError(
            f"Output directory is non-empty: {path}. Use a new output_dir or resume=true."
        )


def _append_outcomes(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    condition: str,
    gallery_sizes: list[int],
    trial: int,
    gw_id: int,
    source: str,
    galleries: Mapping[tuple[int, int, int], Mapping[str, Any]],
    positive_score: float,
    kn_scores: np.ndarray,
    nonkn_scores: np.ndarray,
) -> None:
    for size in gallery_sizes:
        spec = galleries[(size, trial, gw_id)]
        n_kn = int(spec["n_kn_negative"])
        n_nonkn = int(spec["n_nonkn_negative"])
        outcome = mixed_score_outcome(
            positive_score, kn_scores[:n_kn], nonkn_scores[:n_nonkn]
        )
        common = {
            "model": model_name,
            "condition": condition,
            "gallery_size": int(size),
            "trial": int(trial),
            "gw_id": int(gw_id),
            "source": source,
            "positive_score": outcome["positive_score"],
            "best_kn_score": outcome["best_kn_score"],
            "best_nonkn_score": outcome["best_nonkn_score"],
            "kn_margin": outcome["kn_margin"],
            "nonkn_margin": outcome["nonkn_margin"],
            "top1_type": outcome["top1_type"],
            "n_kn_negative": n_kn,
            "n_nonkn_negative": n_nonkn,
        }
        for scope in MIXED_SCOPES:
            rows.append({**common, "scope": scope, **outcome[scope]})


def _aggregate(outcomes: pd.DataFrame, *, kn_fraction: float) -> pd.DataFrame:
    frames = []
    for source_label, frame in [
        ("all", outcomes),
        *outcomes.groupby("source", sort=True),
    ]:
        grouped = frame.groupby(
            ["model", "condition", "scope", "gallery_size"], sort=True
        )
        metrics = grouped[
            list(METRIC_COLUMNS) + ["rank", "kn_margin", "nonkn_margin"]
        ].mean()
        metrics["n_outcomes"] = grouped.size()
        top1 = (
            frame.assign(top1_correct=(frame["top1_type"] == "positive").astype(float))
            .groupby(["model", "condition", "scope", "gallery_size"])["top1_correct"]
            .mean()
        )
        metrics["top1_correct_fraction"] = top1
        metrics = metrics.reset_index()
        metrics["source"] = str(source_label)
        frames.append(metrics)
    result = pd.concat(frames, ignore_index=True)
    baselines = []
    for row in result.itertuples(index=False):
        n_kn, n_nonkn = allocate_mixed_negative_counts(row.gallery_size, kn_fraction)
        n = {"all": row.gallery_size, "kn_only": 1 + n_kn, "nonkn_only": 1 + n_nonkn}[
            row.scope
        ]
        baselines.append(_random_baseline(n))
    for metric in METRIC_COLUMNS:
        result[f"random_{metric}"] = [item[metric] for item in baselines]
    return result


def _plot_with_legacy_retrieval_plotter(
    metrics: pd.DataFrame, output_dir: Path
) -> list[str]:
    """Render each mixed condition/scope with the established curve plotter."""
    metric_fields = (
        ("R@1", "recall_at_1", "random_recall_at_1"),
        ("R@5", "recall_at_5", "random_recall_at_5"),
        ("R@10", "recall_at_10", "random_recall_at_10"),
        ("MRR", "mrr", "random_mrr"),
    )
    all_source = metrics[metrics["source"].eq("all")]
    relative_paths: list[str] = []
    for condition in sorted(all_source["condition"].unique()):
        for scope in MIXED_SCOPES:
            panel = all_source[
                all_source["condition"].eq(condition) & all_source["scope"].eq(scope)
            ]
            curve_rows: list[dict[str, Any]] = []
            for row in panel.to_dict(orient="records"):
                for metric_label, metric_field, _random_field in metric_fields:
                    curve_rows.append(
                        {
                            "method": str(row["model"]),
                            "gallery_size_target": int(row["gallery_size"]),
                            "gallery_size_actual": float(row["gallery_size"]),
                            "coverage": 1.0,
                            "fill_ratio_mean": 1.0,
                            "full_coverage": 1.0,
                            "metric_name": metric_label,
                            "metric_value": float(row[metric_field]),
                        }
                    )
            random_rows = panel.groupby("gallery_size", as_index=False).first()
            for row in random_rows.to_dict(orient="records"):
                for metric_label, _metric_field, random_field in metric_fields:
                    curve_rows.append(
                        {
                            "method": "Random ranking",
                            "gallery_size_target": int(row["gallery_size"]),
                            "gallery_size_actual": float(row["gallery_size"]),
                            "coverage": 1.0,
                            "fill_ratio_mean": 1.0,
                            "full_coverage": 1.0,
                            "metric_name": metric_label,
                            "metric_value": float(row[random_field]),
                        }
                    )
            plot_dir = output_dir / "plots" / str(condition) / str(scope)
            plot_retrieval_curves(curve_rows, plot_dir)
            for filename in ("retrieval_curves.png", "retrieval_curves.pdf"):
                plot_path = plot_dir / filename
                if not plot_path.is_file():
                    raise RuntimeError(
                        f"Legacy retrieval plotter did not create {plot_path}"
                    )
                relative_paths.append(str(plot_path.relative_to(output_dir)))
    return relative_paths


def _paired_bootstrap(
    outcomes: pd.DataFrame,
    *,
    new_name: str,
    baseline_name: str,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    collapsed = analysis.collapse_trials(outcomes)
    return analysis.paired_bootstrap(
        collapsed,
        new_name=new_name,
        baseline_name=baseline_name,
        n_bootstrap=n_bootstrap,
        seed=seed,
    ).to_dict(orient="records")


def run(config_path: Path) -> None:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if str(raw.get("evaluation_mode", "")) != "mixed_kn_nonkn":
        raise ValueError("evaluation_mode must be 'mixed_kn_nonkn'")
    cfg = _normalise_config(raw, config_path)
    output_dir = Path(cfg["output_dir"])
    _prepare_output(output_dir, resume=bool(cfg.get("resume", False)))

    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    amp_dtype, amp_enabled = base._resolve_eval_amp(cfg["amp_dtype"], device)
    comparison_window = (cfg["opt_ref_start"], cfg["opt_ref_end"])

    print("Loading KN metadata and source-balanced queries...")
    metadata = _load_kn_metadata(cfg["test_data_path"])
    query_gw = select_source_balanced_queries(
        metadata["positive_map"].keys(),
        metadata["source_types"],
        queries_per_source=cfg["queries_per_source"],
        seed=seed,
    )
    query_redshifts, redshift_bin_counts = _load_query_redshifts(
        cfg, query_gw, metadata["source_types"]
    )
    all_kn_indices = np.arange(metadata["parent"].size, dtype=np.int64)
    kn_bank, remap = base.load_selected_positive_bank(
        cfg["test_data_path"],
        all_kn_indices,
        runtime_input_window_start=comparison_window[0],
        runtime_input_window_end=comparison_window[1],
    )
    if any(source != compact for source, compact in remap.items()):
        raise AssertionError("Full KN bank must preserve source row indices")

    print("Loading held-out non-KN pool...")
    nonkn_bank = base.load_negative_optical_samples(
        cfg["neg_data_path"],
        cfg["neg_group"],
        n_samples=cfg["n_neg_samples"],
        seed=seed,
        runtime_input_window_start=comparison_window[0],
        runtime_input_window_end=comparison_window[1],
        negative_sample_strategy=str(
            cfg.get("negative_sample_strategy", "block_random")
        ),
        negative_sample_block_rows=cfg.get("negative_sample_block_rows"),
        negative_sample_shuffle=True,
    )
    sequences, _, _ = build_synthetic_time_sky_candidate_sequences(
        test_data_path=cfg["test_data_path"],
        unique_gw_ids=query_gw,
        neg_optical_data=nonkn_bank,
        gallery_sizes=cfg["gallery_sizes"],
        n_trials=cfg["gallery_trials"],
        seed=seed,
        time_window_days=cfg["nonkn_uniform_time_window_days"],
        credible_level_max=cfg["candidate_credible_level_max"],
    )
    empirical_dt = np.abs(
        metadata["first_detection_mjd"] - metadata["event_time_mjd"][metadata["parent"]]
    )
    _apply_training_aligned_nonkn_times(
        sequences,
        empirical_dt,
        fraction=cfg["nonkn_training_empirical_fraction"],
        window_days=cfg["nonkn_uniform_time_window_days"],
        seed=seed,
    )
    nuisance = (
        _optical_nuisance_features(kn_bank, metadata)
        if cfg["kn_candidate_mode"] == "kn_nuisance_matched"
        else None
    )
    galleries = build_mixed_galleries(
        gw_positive_indices=metadata["positive_map"],
        query_gw_ids=query_gw,
        optical_parent_gw_idx=metadata["parent"],
        nonkn_candidate_sequences=sequences,
        gallery_sizes=cfg["gallery_sizes"],
        n_trials=cfg["gallery_trials"],
        seed=seed,
        kn_fraction=cfg["kn_fraction"],
        kn_candidate_mode=cfg["kn_candidate_mode"],
        gw_source_types=metadata["source_types"],
        nuisance_features=nuisance,
    )
    digest = mixed_gallery_identity_digest(galleries)
    _write_gzip_json(
        output_dir / "mixed_gallery_manifest.json.gz", _manifest_payload(galleries)
    )

    model_specs = base._build_model_specs(raw, config_path.parent)
    outcomes: list[dict[str, Any]] = []
    max_size = max(cfg["gallery_sizes"])
    for model_spec in model_specs:
        name = str(model_spec["name"])
        model_type = str(model_spec["type"])
        print(f"\nEvaluating {name} ({model_type})")
        if model_type == "optical":
            model, _ = base.load_optical_model(
                model_spec["resolved_checkpoint"], device
            )
            kn_all_scores = base.score_gallery_optical_only(
                model,
                all_kn_indices,
                kn_bank["times"],
                kn_bank["values"],
                kn_bank["masks"],
                kn_bank["errors"],
                device,
                n_ref=cfg["opt_n_ref"],
                ref_start=comparison_window[0],
                ref_end=comparison_window[1],
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            nonkn_indices = np.arange(nonkn_bank["times"].shape[0], dtype=np.int64)
            nonkn_all_scores = base.score_gallery_optical_only(
                model,
                nonkn_indices,
                nonkn_bank["times"],
                nonkn_bank["values"],
                nonkn_bank["masks"],
                nonkn_bank["errors"],
                device,
                n_ref=cfg["opt_n_ref"],
                ref_start=comparison_window[0],
                ref_end=comparison_window[1],
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            for trial in range(cfg["gallery_trials"]):
                for gw_id in query_gw:
                    spec = galleries[(max_size, trial, gw_id)]
                    pos = float(kn_all_scores[int(spec["positive_index"])])
                    kn_scores = kn_all_scores[
                        np.asarray(spec["kn_negative_indices"], dtype=np.int64)
                    ]
                    nonkn_scores = nonkn_all_scores[
                        np.asarray(spec["nonkn_negative_indices"], dtype=np.int64)
                    ]
                    for condition in cfg["conditions"]:
                        _append_outcomes(
                            outcomes,
                            model_name=name,
                            condition=condition,
                            gallery_sizes=cfg["gallery_sizes"],
                            trial=trial,
                            gw_id=gw_id,
                            source=metadata["source_types"][gw_id],
                            galleries=galleries,
                            positive_score=pos,
                            kn_scores=kn_scores,
                            nonkn_scores=nonkn_scores,
                        )
        elif model_type == "multimodal":
            model, model_args, saved_args = base.load_multimodal_bundle(
                model_spec["resolved_checkpoint"],
                model_spec["resolved_config"],
                device,
                test_data_path=cfg["test_data_path"],
                neg_data_path=cfg["neg_data_path"],
                comparison_window=comparison_window,
                nonkn_cls_base_field=str(
                    cfg.get("nonkn_cls_base_field", "zero_time_mjd_cls_base")
                ),
            )
            scoring = base._resolve_multimodal_scoring(model_spec, saved_args)
            if scoring != "logits":
                raise ValueError(
                    f"Mixed evaluation currently requires logits scoring; {name} resolved to {scoring}"
                )
            kn_embeddings = base.extract_optical_candidate_embeddings(
                model,
                kn_bank,
                device,
                n_ref=int(model_args.get("n_ref", 64)),
                ref_start=float(model_args["ref_start"]),
                ref_end=float(model_args["ref_end"]),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
                desc="  Encoding KN bank",
            )
            nonkn_embeddings = base.extract_negative_gallery_embeddings(
                model,
                nonkn_bank,
                device,
                n_ref=int(model_args.get("n_ref", 64)),
                ref_start=float(model_args["ref_start"]),
                ref_end=float(model_args["ref_end"]),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            query_cache = base._build_gallery_query_cache(
                model,
                query_gw,
                cfg["test_data_path"],
                device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            dual = base._is_dual_fusion_model(model)
            iterator = (
                (trial, gw_id)
                for trial in range(cfg["gallery_trials"])
                for gw_id in query_gw
            )
            for trial, gw_id in tqdm(
                iterator,
                total=cfg["gallery_trials"] * len(query_gw),
                desc=f"  Scoring {name}",
            ):
                spec = galleries[(max_size, trial, gw_id)]
                pos_idx = int(spec["positive_index"])
                kn_idx = np.asarray(spec["kn_negative_indices"], dtype=np.int64)
                nonkn_idx = np.asarray(spec["nonkn_negative_indices"], dtype=np.int64)
                pos_coord = kn_bank["coordinates"][pos_idx].numpy().astype(np.float32)
                pos_dt = _candidate_dt(metadata, np.asarray([pos_idx], dtype=np.int64))
                kn_dt = _candidate_dt(metadata, kn_idx)
                for condition in cfg["conditions"]:
                    features = _condition_candidate_features(
                        condition,
                        positive_coordinate=pos_coord,
                        positive_dt_days=pos_dt,
                        kn_dt_days=kn_dt,
                        nonkn_synthetic_coordinates=spec["nonkn_synthetic_coordinates"],
                        nonkn_training_aligned_dt_days=spec[
                            "nonkn_training_aligned_dt_days"
                        ],
                        n_kn=kn_idx.size,
                        n_nonkn=nonkn_idx.size,
                    )
                    pos_scores = base._score_candidate_bank_with_logits(
                        model,
                        query_cache[gw_id],
                        np.asarray([pos_idx]),
                        kn_embeddings,
                        device,
                        dual,
                        candidate_coords=features["positive_coordinates"],
                        candidate_abs_dt_days=pos_dt,
                        amp_dtype=amp_dtype,
                        amp_enabled=amp_enabled,
                    )
                    kn_scores = base._score_candidate_bank_with_logits(
                        model,
                        query_cache[gw_id],
                        kn_idx,
                        kn_embeddings,
                        device,
                        dual,
                        candidate_coords=features["kn_coordinates"],
                        candidate_abs_dt_days=features["kn_dt_days"],
                        amp_dtype=amp_dtype,
                        amp_enabled=amp_enabled,
                    )
                    nonkn_scores = base._score_candidate_bank_with_logits(
                        model,
                        query_cache[gw_id],
                        nonkn_idx,
                        nonkn_embeddings,
                        device,
                        dual,
                        candidate_coords=features["nonkn_coordinates"],
                        candidate_abs_dt_days=features["nonkn_dt_days"],
                        amp_dtype=amp_dtype,
                        amp_enabled=amp_enabled,
                    )
                    _append_outcomes(
                        outcomes,
                        model_name=name,
                        condition=condition,
                        gallery_sizes=cfg["gallery_sizes"],
                        trial=trial,
                        gw_id=gw_id,
                        source=metadata["source_types"][gw_id],
                        galleries=galleries,
                        positive_score=float(pos_scores[0]),
                        kn_scores=kn_scores,
                        nonkn_scores=nonkn_scores,
                    )
        else:
            raise ValueError(f"Unsupported mixed-eval model type: {model_type}")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    outcome_frame = pd.DataFrame(outcomes)
    if query_redshifts:
        outcome_frame["redshift"] = outcome_frame["gw_id"].map(
            lambda gw_id: query_redshifts[int(gw_id)]["redshift"]
        )
        outcome_frame["redshift_bin_index"] = outcome_frame["gw_id"].map(
            lambda gw_id: query_redshifts[int(gw_id)]["redshift_bin_index"]
        )
        outcome_frame["redshift_bin_label"] = outcome_frame["gw_id"].map(
            lambda gw_id: query_redshifts[int(gw_id)]["redshift_bin_label"]
        )
    outcome_frame.to_csv(
        output_dir / "mixed_retrieval_outcomes.csv.gz", index=False, compression="gzip"
    )
    metrics = _aggregate(outcome_frame, kn_fraction=cfg["kn_fraction"])
    metrics.to_csv(output_dir / "mixed_retrieval_metrics.csv", index=False)
    processed = analysis.postprocess(outcome_frame, cfg, output_dir)
    result = {
        "evaluation_mode": "mixed_kn_nonkn",
        "experiment_id": str(cfg.get("experiment_id", "")),
        "protocol": {
            "primary_analysis": {
                "condition": cfg["primary_condition"],
                "scope": cfg["primary_scope"],
                "candidate_mode": cfg["kn_candidate_mode"],
                "source_aggregation": cfg["primary_metric"]["source_aggregation"],
                "primary_metric": cfg["primary_metric"],
            },
            "condition_semantics": {
                "training_aligned": {
                    "positive_sky": "native",
                    "kn_sky": "positive_shared",
                    "nonkn_sky": "synthetic_within_credible_region",
                    "kn_time": "parent_relative",
                    "nonkn_time": "empirical_uniform_mixture",
                },
                "positive_shared": {
                    "role": "supplementary_strict_control",
                    "candidate_sky": "positive_shared",
                    "candidate_time": "positive_shared",
                },
            },
            "intentional_training_differences": [
                "held-out ELASTICC test candidates replace ELASTICC2 training candidates",
                "training_aligned non-KN sky is sampled within the query credible region",
                "evaluation sweeps multiple gallery sizes",
            ],
            "inference": {
                "bootstrap_unit": "gw_id_after_trial_mean",
                "bootstrap_samples": cfg["bootstrap_samples"],
                "bootstrap_seed": cfg["bootstrap_seed"],
                "source_resampling": "stratified",
            },
        },
        "config": cfg,
        "experiment_digest": base.stable_digest(raw),
        "code_digest": base.source_tree_digest(MODEL_DIR),
        "input_config": str(config_path),
        "models": [
            {
                key: spec.get(key)
                for key in (
                    "name",
                    "type",
                    "scoring",
                    "resolved_checkpoint",
                    "resolved_config",
                )
            }
            for spec in model_specs
        ],
        "data_files": {
            "test_data": {
                "requested_path": cfg["requested_test_data_path"],
                "runtime_path": cfg["test_data_path"],
                "size_bytes": os.stat(cfg["test_data_path"]).st_size,
                "mtime_ns": os.stat(cfg["test_data_path"]).st_mtime_ns,
                "hdf5_groups": ["events/gw_data", "events/optical_data"],
            },
            "negative_data": {
                "requested_path": cfg["requested_neg_data_path"],
                "runtime_path": cfg["neg_data_path"],
                "size_bytes": os.stat(cfg["neg_data_path"]).st_size,
                "mtime_ns": os.stat(cfg["neg_data_path"]).st_mtime_ns,
                "hdf5_group": cfg["neg_group"],
                "role": "held_out_test_candidates",
            },
        },
        "gallery_identity_digest": digest,
        "n_queries": len(query_gw),
        "query_gw_ids": query_gw,
        "query_source_counts": {
            label: sum(metadata["source_types"][gw_id] == label for gw_id in query_gw)
            for label in sorted({metadata["source_types"][gw_id] for gw_id in query_gw})
        },
        "metrics": metrics.to_dict(orient="records"),
        "redshift_query_bin_counts": redshift_bin_counts,
        "training_effect": processed["training_effect"].iloc[0].to_dict(),
        "paired_bootstrap": processed["paired_bootstrap"].to_dict(orient="records"),
        "artifacts": {
            "manifest": "mixed_gallery_manifest.json.gz",
            "outcomes": "mixed_retrieval_outcomes.csv.gz",
            "metrics": "mixed_retrieval_metrics.csv",
            **processed["artifacts"],
        },
    }
    temporary = output_dir / "mixed_retrieval_comparison.json.tmp"
    temporary.write_text(
        json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
    )
    os.replace(temporary, output_dir / "mixed_retrieval_comparison.json")
    print(f"Mixed retrieval comparison complete: {output_dir}")


def run_postprocess_only(config_path: Path) -> None:
    """Regenerate GW-level statistics and plots from saved outcome rows."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if str(raw.get("evaluation_mode", "")) != "mixed_kn_nonkn":
        raise ValueError("evaluation_mode must be 'mixed_kn_nonkn'")
    cfg = _normalise_config(raw, config_path)
    output_dir = Path(cfg["output_dir"])
    outcome_path = output_dir / "mixed_retrieval_outcomes.csv.gz"
    if not outcome_path.is_file():
        raise FileNotFoundError(
            f"Postprocess-only requires saved outcomes: {outcome_path}"
        )
    outcomes = pd.read_csv(outcome_path)
    redshift_cfg = cfg["redshift_analysis"]
    if redshift_cfg.get("enabled", False) and "redshift" not in outcomes.columns:
        metadata = _load_kn_metadata(cfg["test_data_path"])
        query_gw = sorted(outcomes["gw_id"].astype(int).unique().tolist())
        query_redshifts, _ = _load_query_redshifts(
            cfg, query_gw, metadata["source_types"]
        )
        outcomes["redshift"] = outcomes["gw_id"].map(
            lambda gw_id: query_redshifts[int(gw_id)]["redshift"]
        )
        outcomes["redshift_bin_index"] = outcomes["gw_id"].map(
            lambda gw_id: query_redshifts[int(gw_id)]["redshift_bin_index"]
        )
        outcomes["redshift_bin_label"] = outcomes["gw_id"].map(
            lambda gw_id: query_redshifts[int(gw_id)]["redshift_bin_label"]
        )

    processed = analysis.postprocess(outcomes, cfg, output_dir)
    summary = {
        "experiment_id": str(cfg.get("experiment_id", "")),
        "mode": "postprocess_only",
        "input_outcomes": str(outcome_path),
        "training_effect": processed["training_effect"].iloc[0].to_dict(),
        "artifacts": processed["artifacts"],
    }
    summary_path = output_dir / "mixed_retrieval_postprocess.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    os.replace(temporary, summary_path)

    result_path = output_dir / "mixed_retrieval_comparison.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["training_effect"] = summary["training_effect"]
        result.setdefault("artifacts", {}).update(processed["artifacts"])
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, indent=2, default=_json_default), encoding="utf-8"
        )
        os.replace(temporary, result_path)
    print(f"Mixed retrieval postprocessing complete: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Regenerate statistics and plots from saved outcome rows.",
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    if args.postprocess_only:
        run_postprocess_only(config_path)
    else:
        run(config_path)


if __name__ == "__main__":
    main()
