#!/usr/bin/env python3
"""Merge a single-model retrieval run into an existing comparison result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval.eval_retrieval_comparison import (
    aggregate_redshift_macro_metrics,
    plot_redshift_macro_metrics,
    plot_redshift_metrics,
    write_redshift_csv,
    write_redshift_macro_csv,
)
from retrieval_gallery import plot_retrieval_curves

GALLERY_CONFIG_KEYS = (
    "seed",
    "gallery_sizes",
    "gallery_trials",
    "gallery_candidate_mode",
    "gallery_candidate_time_window_days",
    "gallery_candidate_credible_level_max",
    "gallery_include_undersized",
    "positive_selection",
    "n_neg_samples",
    "negative_sample_strategy",
)

def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def _assert_same_galleries(base: Dict[str, Any], supplement: Dict[str, Any]) -> None:
    for key in GALLERY_CONFIG_KEYS:
        if base["config"].get(key) != supplement["config"].get(key):
            raise ValueError(f"Gallery config mismatch for {key!r}.")
    for key in ("gallery_positive_summary", "selected_positive_summary"):
        if base.get(key) != supplement.get(key):
            raise ValueError(f"Gallery realization mismatch for {key!r}.")

def merge_results(base_path: Path, supplement_path: Path, output_dir: Path) -> Path:
    base = _load(base_path)
    supplement = _load(supplement_path)
    _assert_same_galleries(base, supplement)
    if len(supplement.get("models", {})) != 1:
        raise ValueError("Supplement must contain exactly one model.")

    method, model_result = next(iter(supplement["models"].items()))
    if method in base.get("models", {}):
        raise ValueError(f"Method already exists in base result: {method}")

    merged = dict(base)
    merged["models"] = dict(base["models"])
    merged["models"][method] = model_result
    merged["curve_rows"] = list(base.get("curve_rows", [])) + list(supplement.get("curve_rows", []))
    merged["redshift_rows"] = list(base.get("redshift_rows", [])) + list(supplement.get("redshift_rows", []))
    merged["redshift_macro_rows"] = aggregate_redshift_macro_metrics(merged["redshift_rows"])
    merged["table"] = dict(base["table"])
    merged["table"]["rows"] = list(base["table"]["rows"]) + list(supplement["table"]["rows"])
    merged["config"] = dict(base["config"])
    merged["config"]["output_dir"] = str(output_dir)
    merged["config"]["models"] = list(base["config"].get("models", [])) + list(
        supplement["config"].get("models", [])
    )
    merged["supplement_provenance"] = {
        "base_result": str(base_path),
        "single_model_result": str(supplement_path),
        "added_method": method,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ablation_comparison.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, ensure_ascii=False)

    plot_retrieval_curves(merged["curve_rows"], output_dir)
    if merged["redshift_rows"]:
        write_redshift_csv(merged["redshift_rows"], output_dir / "redshift_metrics.csv")
        write_redshift_macro_csv(
            merged["redshift_macro_rows"], output_dir / "redshift_macro_metrics_log10_weighted.csv"
        )
        plot_redshift_metrics(merged["redshift_rows"], output_dir)
        plot_redshift_macro_metrics(merged["redshift_macro_rows"], output_dir)
    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(merge_results(args.base.resolve(), args.supplement.resolve(), args.output_dir.resolve()))

if __name__ == "__main__":
    main()
