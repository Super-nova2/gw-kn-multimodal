#!/usr/bin/env python3
"""Regenerate paper retrieval plots from existing retrieval JSON outputs.

This redraws the plot files only; it does not rerun model evaluation.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPO_ROOT.parent
MODEL_DIR = REPO_ROOT / "Model"
MODEL_SCRIPT_DIR = MODEL_DIR / "script"
for p in (str(REPO_ROOT), str(MODEL_DIR), str(MODEL_SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval_gallery import (  # noqa: E402
    plot_retrieval_coverage,
    plot_retrieval_curves,
)
from eval_retrieval_comparison import (  # noqa: E402
    plot_redshift_coverage as plot_ablation_redshift_coverage,
    plot_redshift_macro_metrics as plot_ablation_redshift_macro_metrics,
    plot_redshift_metrics as plot_ablation_redshift_metrics,
)
from eval_gw170817a_retrieval import (  # noqa: E402
    plot_redshift_coverage as plot_gw170817a_redshift_coverage,
    plot_redshift_macro_metrics as plot_gw170817a_redshift_macro_metrics,
    plot_redshift_metrics as plot_gw170817a_redshift_metrics,
)

PlotFn = Callable[[Sequence[Mapping[str, object]], Path], None]

CASES = [
    {
        "name": "retrieval_comparison_v11",
        "result_dir": MODEL_DIR / "eval_results" / "retrieval_comparison_v11",
        "json_name": "ablation_comparison.json",
        "paper_dir": PROJECT_ROOT / "paper_draft" / "figures" / "results" / "retrieval_comparison",
        "plot_redshift_metrics": plot_ablation_redshift_metrics,
        "plot_redshift_coverage": plot_ablation_redshift_coverage,
        "plot_redshift_macro_metrics": plot_ablation_redshift_macro_metrics,
    },
    {
        "name": "gw170817a_lsst_retrieval",
        "result_dir": MODEL_DIR / "eval_results" / "gw170817a_lsst_retrieval",
        "json_name": "gw170817a_retrieval.json",
        "paper_dir": PROJECT_ROOT / "paper_draft" / "figures" / "results" / "gw170817a_lsst_retrieval",
        "plot_redshift_metrics": plot_gw170817a_redshift_metrics,
        "plot_redshift_coverage": plot_gw170817a_redshift_coverage,
        "plot_redshift_macro_metrics": plot_gw170817a_redshift_macro_metrics,
    },
]

PAPER_FILES = [
    "retrieval_curves.pdf",
    "redshift_macro_metrics_log10_weighted.pdf",
]


def _copy_paper_files(result_dir: Path, paper_dir: Path) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    for filename in PAPER_FILES:
        src = result_dir / filename
        if not src.exists():
            print(f"  WARN: {src} not found; not copied")
            continue
        dst = paper_dir / filename
        shutil.copy2(src, dst)
        print(f"  Copied {src} -> {dst}")


def main() -> int:
    for case in CASES:
        result_dir = Path(case["result_dir"])
        json_path = result_dir / str(case["json_name"])
        if not json_path.exists():
            print(f"SKIP {case['name']}: {json_path} not found")
            continue

        print(f"Loading {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        curve_rows = data.get("curve_rows", [])
        redshift_rows = data.get("redshift_rows", [])
        redshift_macro_rows = data.get("redshift_macro_rows", [])

        if curve_rows:
            print(f"  {len(curve_rows)} curve rows: regenerating retrieval curves/coverage")
            plot_retrieval_curves(curve_rows, result_dir)
            plot_retrieval_coverage(curve_rows, result_dir)
        if redshift_rows:
            print(f"  {len(redshift_rows)} redshift rows: regenerating redshift diagnostics")
            case["plot_redshift_metrics"](redshift_rows, result_dir)
            case["plot_redshift_coverage"](redshift_rows, result_dir)
        if redshift_macro_rows:
            print(f"  {len(redshift_macro_rows)} redshift macro rows: regenerating paper macro plot")
            case["plot_redshift_macro_metrics"](redshift_macro_rows, result_dir)

        _copy_paper_files(result_dir, Path(case["paper_dir"]))
        print(f"  Done: {case['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
