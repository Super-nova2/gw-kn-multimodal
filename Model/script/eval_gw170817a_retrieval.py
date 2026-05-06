#!/usr/bin/env python3
"""Dedicated GW170817A LSST redshift retrieval evaluation.

This script intentionally does not extend ``eval_retrieval_comparison.py``.
It reuses helper functions from the existing evaluation stack and keeps the
GW170817A redshift aggregation and plots in this standalone entry point.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from retrieval_gallery import (  # noqa: E402
    _plot_method_label,
    aggregate_gallery_outcomes,
    build_comparison_model_specs,
    build_curve_rows,
    build_prefixed_gallery_specs,
    build_synthetic_time_sky_candidate_sequences,
    build_time_sky_candidate_sequences,
    plot_retrieval_coverage,
    plot_retrieval_curves,
    score_all_galleries_skymap,
)
from test_evaluate import (  # noqa: E402
    _resolve_eval_amp,
    load_negative_optical_samples,
    load_gw_event_time_mjd_table,
)
import eval_retrieval_comparison as base_eval  # noqa: E402


TABLE_METRIC_LABELS = ["R@1", "R@5", "R@10", "MRR"]
TABLE_METRIC_KEYS = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(base_dir: Path, value: Any) -> Optional[str]:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _parse_gallery_sizes(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(part.strip()) for part in str(value or "10,100,500").split(",") if part.strip()]


def _parse_n_neg_samples(value: Any) -> int:
    if value in (None, "", "none", "null", "all", "full"):
        return -1
    out = int(value)
    return -1 if out <= 0 else out


def normalize_config(raw_cfg: Mapping[str, Any], cfg_path: Path) -> Dict[str, Any]:
    cfg_dir = cfg_path.parent
    test_data_path = _resolve_path(cfg_dir, raw_cfg.get("test_data_path"))
    neg_data_path = _resolve_path(cfg_dir, raw_cfg.get("neg_data_path"))
    output_dir = _resolve_path(cfg_dir, raw_cfg.get("output_dir")) or str(
        (MODEL_DIR / "eval_results" / "gw170817a_lsst_retrieval").resolve()
    )
    if not test_data_path:
        raise ValueError("Config is missing test_data_path")
    return {
        "input_config": str(cfg_path),
        "test_data_path": test_data_path,
        "neg_data_path": neg_data_path,
        "neg_group": str(raw_cfg.get("neg_group", "ELASTICC/optical_data")),
        "output_dir": output_dir,
        "device": str(raw_cfg.get("device", "cuda")),
        "seed": int(raw_cfg.get("seed", 42)),
        "batch_size": int(raw_cfg.get("batch_size", 512)),
        "num_workers": int(raw_cfg.get("num_workers", 2)),
        "gallery_sizes": _parse_gallery_sizes(raw_cfg.get("gallery_sizes", "10,100,500")),
        "gallery_trials": int(raw_cfg.get("gallery_trials", 5)),
        "gallery_candidate_mode": str(raw_cfg.get("gallery_candidate_mode", "time_sky_hard")).strip().lower(),
        "gallery_candidate_time_window_days": float(raw_cfg.get("gallery_candidate_time_window_days", 30.0)),
        "gallery_candidate_credible_level_max": float(raw_cfg.get("gallery_candidate_credible_level_max", 0.9)),
        "gallery_include_undersized": bool(raw_cfg.get("gallery_include_undersized", True)),
        "max_kn_per_redshift_bin": int(raw_cfg.get("max_kn_per_redshift_bin", 0)) or None,
        "n_neg_samples": _parse_n_neg_samples(raw_cfg.get("n_neg_samples", -1)),
        "nonkn_cls_base_field": str(raw_cfg.get("nonkn_cls_base_field", "zero_time_mjd_cls_base")),
        "comparison_window": [float(v) for v in raw_cfg.get("comparison_window", [-0.1, 0.2])],
        "amp_dtype": str(raw_cfg.get("amp_dtype", "auto")),
    }


def load_redshift_metadata(test_data_path: str) -> Dict[int, Dict[str, Any]]:
    with h5py.File(test_data_path, "r") as f:
        if "events/gw_data/redshift" not in f or "events/gw_data/redshift_bin" not in f:
            raise KeyError("GW170817A retrieval HDF5 must contain redshift and redshift_bin fields.")
        redshift = np.asarray(f["events/gw_data/redshift"][:], dtype=np.float64)
        redshift_bin = np.asarray(f["events/gw_data/redshift_bin"][:], dtype=np.int64)
    return {
        int(idx): {
            "redshift": float(redshift[idx]),
            "redshift_bin": int(redshift_bin[idx]),
        }
        for idx in range(redshift.shape[0])
    }


def subsample_kn_per_redshift_bin(
    gw_positive_indices: Dict[int, Any],
    redshift_metadata: Dict[int, Dict[str, Any]],
    max_per_bin: int,
    seed: int,
) -> Tuple[Dict[int, Any], List[int]]:
    """Cap KN events per redshift bin by random subsampling.

    Args:
        gw_positive_indices: GW ID -> optical index array mapping.
        redshift_metadata: GW ID -> {"redshift_bin": int, "redshift": float}.
        max_per_bin: Maximum number of KN GW events to keep per redshift bin.
        seed: Random seed for reproducibility.

    Returns:
        (filtered_gw_positive_indices, filtered_all_test_gw_ids)
    """
    rng = np.random.default_rng(int(seed))
    all_gw_ids = sorted(gw_positive_indices.keys())

    # Group GW IDs by redshift_bin
    bins: Dict[int, List[int]] = {}
    for gw_id in all_gw_ids:
        meta = redshift_metadata.get(int(gw_id))
        if meta is None:
            continue
        bin_id = int(meta["redshift_bin"])
        bins.setdefault(bin_id, []).append(int(gw_id))

    # Track per-bin counts before/after
    bin_summary_lines = []
    total_before = 0
    total_after = 0

    kept_gw_ids: set = set()
    for bin_id in sorted(bins):
        gw_ids = bins[bin_id]
        n_before = len(gw_ids)
        total_before += n_before
        if n_before > max_per_bin:
            selected = rng.choice(gw_ids, size=max_per_bin, replace=False).tolist()
            n_after = len(selected)
        else:
            selected = gw_ids
            n_after = n_before
        kept_gw_ids.update(selected)
        total_after += n_after
        z_vals = [float(redshift_metadata[g]["redshift"]) for g in gw_ids]
        z_mean = float(np.mean(z_vals)) if z_vals else 0.0
        flag = " (capped)" if n_before > max_per_bin else ""
        bin_summary_lines.append(
            f"  bin={bin_id} z~{z_mean:.3f}: {n_before} -> {n_after}{flag}"
        )

    print(
        f"Per-redshift KN subsample (max={max_per_bin}): "
        f"{total_before} -> {total_after} total events"
    )
    for line in bin_summary_lines:
        print(line)

    filtered_indices = {
        gw_id: gw_positive_indices[gw_id]
        for gw_id in kept_gw_ids
    }
    filtered_gw_ids = sorted(kept_gw_ids)
    return filtered_indices, filtered_gw_ids


def aggregate_redshift_metrics(
    *,
    outcomes: Mapping[Tuple[int, int, int], Mapping[str, Any]],
    gallery_sizes: Sequence[int],
    n_trials: int,
    unique_gw: Sequence[int],
    redshift_metadata: Mapping[int, Mapping[str, Any]],
    method_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gallery_size in [int(size) for size in gallery_sizes]:
        buckets: Dict[int, Dict[str, Any]] = {}
        for trial in range(int(n_trials)):
            for gw_id in [int(gw) for gw in unique_gw]:
                key = (gallery_size, int(trial), gw_id)
                if key not in outcomes or gw_id not in redshift_metadata:
                    continue
                outcome = outcomes[key]
                meta = redshift_metadata[gw_id]
                bin_id = int(meta["redshift_bin"])
                bucket = buckets.setdefault(
                    bin_id,
                    {
                        "redshifts": [],
                        "rank": [],
                        "coverage": [],
                        "full_coverage": [],
                        "actual_sizes": [],
                    },
                )
                rank = int(outcome["rank"])
                actual_size = int(outcome.get("actual_gallery_size", gallery_size))
                coverage_met = bool(outcome.get("coverage_met", actual_size >= gallery_size))
                fill_ratio = min(1.0, max(0.0, float(actual_size) / float(max(int(gallery_size), 1))))
                bucket["redshifts"].append(float(meta["redshift"]))
                bucket["rank"].append(rank)
                bucket["coverage"].append(fill_ratio)
                bucket["full_coverage"].append(1.0 if coverage_met else 0.0)
                bucket["actual_sizes"].append(actual_size)

        for bin_id in sorted(buckets):
            bucket = buckets[bin_id]
            ranks = np.asarray(bucket["rank"], dtype=np.int64)
            rows.append(
                {
                    "method": str(method_name),
                    "redshift_bin": int(bin_id),
                    "redshift": float(np.mean(bucket["redshifts"])),
                    "gallery_size": int(gallery_size),
                    "n_queries": int(ranks.size),
                    "recall_at_1": float(np.mean(ranks < 1)) if ranks.size else 0.0,
                    "recall_at_5": float(np.mean(ranks < 5)) if ranks.size else 0.0,
                    "recall_at_10": float(np.mean(ranks < 10)) if ranks.size else 0.0,
                    "mrr": float(np.mean(1.0 / (ranks.astype(np.float64) + 1.0))) if ranks.size else 0.0,
                    "coverage": float(np.mean(bucket["coverage"])) if bucket["coverage"] else 0.0,
                    "fill_ratio_mean": float(np.mean(bucket["coverage"])) if bucket["coverage"] else 0.0,
                    "full_coverage": float(np.mean(bucket["full_coverage"])) if bucket["full_coverage"] else 0.0,
                    "effective_gallery_size_mean": float(np.mean(bucket["actual_sizes"])) if bucket["actual_sizes"] else 0.0,
                }
            )
    return rows


def plot_redshift_metrics(rows: Sequence[Mapping[str, Any]], output_dir: Path | str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = list(rows)
    if not rows:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in rows})
    metrics = [("recall_at_1", "R@1"), ("recall_at_10", "R@10"), ("mrr", "MRR")]
    all_gallery_sizes = sorted({int(row["gallery_size"]) for row in rows})

    for gallery_size in all_gallery_sizes:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
        for ax, (key, label) in zip(axes, metrics):
            for method in methods:
                method_rows = sorted(
                    [row for row in rows if str(row["method"]) == method and int(row["gallery_size"]) == gallery_size],
                    key=lambda row: float(row["redshift"]),
                )
                if method_rows:
                    ax.plot(
                        [float(row["redshift"]) for row in method_rows],
                        [float(row[key]) for row in method_rows],
                        marker="o",
                        linewidth=2,
                        label=_plot_method_label(method),
                    )
            ax.set_xlabel("Redshift")
            ax.set_ylabel(label)
            ax.set_title(f"{label} vs Redshift  (gallery_size={gallery_size})")
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(out / f"redshift_retrieval_metrics_g{gallery_size}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_redshift_coverage(rows: Sequence[Mapping[str, Any]], output_dir: Path | str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = list(rows)
    if not rows:
        return
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    methods = sorted({str(row["method"]) for row in rows})
    all_gallery_sizes = sorted({int(row["gallery_size"]) for row in rows})

    for gallery_size in all_gallery_sizes:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for method in methods:
            method_rows = sorted(
                [row for row in rows if str(row["method"]) == method and int(row["gallery_size"]) == gallery_size],
                key=lambda row: float(row["redshift"]),
            )
            if method_rows:
                ax.plot(
                    [float(row["redshift"]) for row in method_rows],
                    [float(row["coverage"]) for row in method_rows],
                    marker="o",
                    linewidth=2,
                    label=_plot_method_label(method),
                )
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Mean Fill Ratio")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Mean Gallery Fill Ratio vs Redshift  (gallery_size={gallery_size})")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out / f"redshift_coverage_g{gallery_size}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_redshift_csv(rows: Sequence[Mapping[str, Any]], output_path: Path | str) -> None:
    import csv

    rows = list(rows)
    if not rows:
        return
    fieldnames = [
        "method",
        "redshift_bin",
        "redshift",
        "gallery_size",
        "n_queries",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr",
        "coverage",
        "fill_ratio_mean",
        "full_coverage",
        "effective_gallery_size_mean",
    ]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _neg_target(n_neg_samples: int) -> Optional[int]:
    return int(n_neg_samples) if int(n_neg_samples) > 0 else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dedicated GW170817A LSST redshift retrieval evaluation.")
    parser.add_argument("--config", required=True, help="Path to retrieval_gw170817a_lsst.json")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config).expanduser().resolve()
    raw_cfg = _load_json(cfg_path)
    cfg = normalize_config(raw_cfg, cfg_path)
    model_specs = build_comparison_model_specs(raw_cfg, cfg_path.parent)
    _seed_all(int(cfg["seed"]))

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    amp_dtype, amp_enabled = _resolve_eval_amp(cfg["amp_dtype"], device)
    comparison_window = (float(cfg["comparison_window"][0]), float(cfg["comparison_window"][1]))
    gallery_sizes = list(cfg["gallery_sizes"])
    n_trials = int(cfg["gallery_trials"])

    redshift_metadata = load_redshift_metadata(cfg["test_data_path"])
    gw_positive_indices, all_test_gw_ids = base_eval.load_test_positive_index_map(cfg["test_data_path"])

    max_per_bin = cfg.get("max_kn_per_redshift_bin")
    if max_per_bin:
        gw_positive_indices, all_test_gw_ids = subsample_kn_per_redshift_bin(
            gw_positive_indices, redshift_metadata, max_per_bin, int(cfg["seed"])
        )

    neg_optical_data = load_negative_optical_samples(
        cfg["neg_data_path"],
        cfg["neg_group"],
        n_samples=_neg_target(cfg["n_neg_samples"]),
        seed=int(cfg["seed"]),
        require_zero_time_mjd_base=False,
        require_zero_time_mjd_cls_base=False,
        nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
        runtime_input_window_start=comparison_window[0],
        runtime_input_window_end=comparison_window[1],
    )
    gallery_candidate_mode = str(cfg["gallery_candidate_mode"]).strip().lower()
    if gallery_candidate_mode == "time_sky_hard":
        candidate_sequences, gallery_gw_skymaps, _gallery_gw_times = build_time_sky_candidate_sequences(
            test_data_path=cfg["test_data_path"],
            unique_gw_ids=all_test_gw_ids,
            neg_optical_data=neg_optical_data,
            n_trials=n_trials,
            seed=int(cfg["seed"]),
            time_window_days=float(cfg["gallery_candidate_time_window_days"]),
            credible_level_max=float(cfg["gallery_candidate_credible_level_max"]),
            zero_time_field=cfg["nonkn_cls_base_field"],
        )
    elif gallery_candidate_mode == "synthetic_time_sky_hard":
        candidate_sequences, gallery_gw_skymaps, _gallery_gw_times = build_synthetic_time_sky_candidate_sequences(
            test_data_path=cfg["test_data_path"],
            unique_gw_ids=all_test_gw_ids,
            neg_optical_data=neg_optical_data,
            gallery_sizes=gallery_sizes,
            n_trials=n_trials,
            seed=int(cfg["seed"]),
            time_window_days=float(cfg["gallery_candidate_time_window_days"]),
            credible_level_max=float(cfg["gallery_candidate_credible_level_max"]),
        )
    else:
        raise ValueError(
            f"Unsupported gallery_candidate_mode='{gallery_candidate_mode}'. "
            "Expected one of {'time_sky_hard', 'synthetic_time_sky_hard'}."
        )
    galleries, unique_gw = build_prefixed_gallery_specs(
        gw_positive_indices=gw_positive_indices,
        candidate_sequences=candidate_sequences,
        gallery_sizes=gallery_sizes,
        n_trials=n_trials,
        seed=int(cfg["seed"]),
        include_undersized=bool(cfg["gallery_include_undersized"]),
    )
    selected_positive_indices = np.unique(
        np.asarray([int(spec["positive_index"]) for spec in galleries.values()], dtype=np.int64)
    )
    positive_bank, positive_remap = base_eval.load_selected_positive_bank(
        cfg["test_data_path"], selected_positive_indices,
        runtime_input_window_start=comparison_window[0],
        runtime_input_window_end=comparison_window[1],
    )
    galleries = base_eval.remap_gallery_positive_indices(galleries, positive_remap)
    gw_event_time_mjd_table = load_gw_event_time_mjd_table(cfg["test_data_path"], device, required=False)

    model_results: OrderedDict[str, Dict[str, Any]] = OrderedDict()
    curve_rows: List[Dict[str, Any]] = []
    redshift_rows: List[Dict[str, Any]] = []

    for model_spec in model_specs:
        name = str(model_spec["name"])
        model_type = str(model_spec["type"])
        print(f"Evaluating {name} ({model_type})")
        if model_type == "skymap":
            outcomes = score_all_galleries_skymap(
                positive_bank={"opt_coords": positive_bank["opt_coords"]},
                galleries=galleries,
                gw_skymaps=gallery_gw_skymaps,
            )
        elif model_type == "optical":
            model, ckpt_args = base_eval.load_optical_model(model_spec["resolved_checkpoint"], device)
            ranks = base_eval.score_all_galleries_optical(
                model,
                positive_bank,
                neg_optical_data,
                galleries,
                device=device,
                n_ref=int(ckpt_args.get("n_ref", 64)),
                ref_start=comparison_window[0],
                ref_end=comparison_window[1],
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            outcomes = base_eval.enrich_gallery_outcomes(ranks, galleries)
            del model
        elif model_type == "multimodal":
            model, runtime_model_args, saved_args = base_eval.load_multimodal_bundle(
                model_spec["resolved_checkpoint"],
                model_spec["resolved_config"],
                device,
                test_data_path=cfg["test_data_path"],
                neg_data_path=cfg["neg_data_path"],
                comparison_window=comparison_window,
                nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
            )
            embeddings = base_eval.extract_optical_candidate_embeddings(
                model,
                positive_bank,
                device,
                n_ref=int(runtime_model_args.get("n_ref", 64)),
                ref_start=float(runtime_model_args["ref_start"]),
                ref_end=float(runtime_model_args["ref_end"]),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
                desc="  Extracting GW170817A positive query embeddings",
            )
            negative_embeddings = base_eval.extract_negative_gallery_embeddings(
                model,
                neg_optical_data,
                device,
                n_ref=int(runtime_model_args.get("n_ref", 64)),
                ref_start=float(runtime_model_args["ref_start"]),
                ref_end=float(runtime_model_args["ref_end"]),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            scoring_mode = base_eval._resolve_multimodal_scoring(model_spec, saved_args)
            if scoring_mode == "contrastive":
                ranks = base_eval.score_all_galleries_contrastive(
                    model,
                    embeddings,
                    negative_embeddings,
                    galleries,
                    unique_gw,
                    test_data_path=cfg["test_data_path"],
                    device=device,
                    model_args=runtime_model_args,
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                )
            else:
                ranks = base_eval.score_all_galleries_multimodal(
                    model,
                    embeddings,
                    negative_embeddings,
                    galleries,
                    unique_gw,
                    test_data_path=cfg["test_data_path"],
                    device=device,
                    model_args=runtime_model_args,
                    gw_event_time_mjd_table=gw_event_time_mjd_table,
                    amp_dtype=amp_dtype,
                    amp_enabled=amp_enabled,
                )
            outcomes = base_eval.enrich_gallery_outcomes(ranks, galleries)
            del model, embeddings, negative_embeddings
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        retrieval_metrics, retrieval_by_source, coverage_stats = aggregate_gallery_outcomes(
            outcomes=outcomes,
            gallery_sizes=gallery_sizes,
            n_trials=n_trials,
            unique_gw=unique_gw,
            gw_source_map={int(gw_id): "gw170817a" for gw_id in unique_gw},
        )
        model_results[name] = {
            "type": model_type,
            "scoring": str(model_spec.get("scoring")),
            "resolved_checkpoint": model_spec.get("resolved_checkpoint"),
            "resolved_config": model_spec.get("resolved_config"),
            "retrieval": retrieval_metrics,
            "retrieval_by_source": retrieval_by_source,
            "effective_gallery_size_stats": coverage_stats,
        }
        curve_rows.extend(
            build_curve_rows(
                method_name=name,
                gallery_sizes=gallery_sizes,
                retrieval_metrics=retrieval_metrics,
                coverage_stats=coverage_stats,
            )
        )
        redshift_rows.extend(
            aggregate_redshift_metrics(
                outcomes=outcomes,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                unique_gw=unique_gw,
                redshift_metadata=redshift_metadata,
                method_name=name,
            )
        )
        torch.cuda.empty_cache()
        gc.collect()

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_retrieval_curves(curve_rows, output_dir)
    plot_retrieval_coverage(curve_rows, output_dir)
    plot_redshift_metrics(redshift_rows, output_dir)
    plot_redshift_coverage(redshift_rows, output_dir)
    write_redshift_csv(redshift_rows, output_dir / "redshift_metrics.csv")
    output = {
        "curve_rows": curve_rows,
        "redshift_rows": redshift_rows,
        "models": dict(model_results),
        "config": cfg,
    }
    with (output_dir / "gw170817a_retrieval.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(output), f, indent=2, ensure_ascii=False)
    print(f"Wrote GW170817A retrieval outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
