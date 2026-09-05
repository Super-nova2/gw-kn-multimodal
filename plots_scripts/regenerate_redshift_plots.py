#!/usr/bin/env python3
"""Regenerate paper retrieval plots from existing retrieval JSON outputs.

This redraws the plot files only; it does not rerun model evaluation.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = REPO_ROOT.parent
MODEL_DIR = REPO_ROOT / "Model"
MODEL_SCRIPT_DIR = MODEL_DIR / "scripts" / "eval"
for p in (str(REPO_ROOT), str(MODEL_DIR), str(MODEL_SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_retrieval_comparison import (  # noqa: I001
    plot_redshift_macro_metrics as plot_ablation_redshift_macro_metrics,
)
from retrieval_gallery import plot_retrieval_curves


CASES = [
    {
        "name": "retrieval_comparison_mixed_gallery_v1_all_models",
        "result_dir": MODEL_DIR / "eval_results" / "retrieval_comparison_mixed_gallery_v1_all_models",
        "json_name": "ablation_comparison.json",
        "paper_dir": PROJECT_ROOT / "paper_apj" / "figures" / "results" / "retrieval_comparison",
    },
]

PAPER_METHODS = {
    "Full (HPO v7 + Mixed Gallery)",
    "w/o Contrastive Loss",
    "w/o Retrieval Loss",
    "w/o Cross-Attn",
    "w/o Fusion",
    "Optical-only",
    "Fink Random Forest",
    "Skymap-only",
}

PAPER_FILES = [
    "retrieval_curves.pdf",
    "redshift_macro_metrics_log10_weighted.pdf",
]


def _copy_paper_files(render_dir: Path, paper_dir: Path) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    for filename in PAPER_FILES:
        src = render_dir / filename
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

        curve_rows = [
            row for row in data.get("curve_rows", [])
            if str(row.get("method")) in PAPER_METHODS
        ]
        redshift_macro_rows = [
            row for row in data.get("redshift_macro_rows", [])
            if str(row.get("method")) in PAPER_METHODS
        ]

        with tempfile.TemporaryDirectory(prefix="magiks_retrieval_figures_") as tmp:
            render_dir = Path(tmp)
            if curve_rows:
                print(f"  {len(curve_rows)} curve rows: regenerating paper retrieval curves")
                plot_retrieval_curves(curve_rows, render_dir)
            if redshift_macro_rows:
                print(f"  {len(redshift_macro_rows)} redshift macro rows: regenerating paper macro plot")
                plot_ablation_redshift_macro_metrics(redshift_macro_rows, render_dir)
            _copy_paper_files(render_dir, Path(case["paper_dir"]))
        print(f"  Done: {case['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
