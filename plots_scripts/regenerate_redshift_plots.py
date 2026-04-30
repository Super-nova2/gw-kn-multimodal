#!/usr/bin/env python3
"""Regenerate redshift metrics/coverage plots from an existing retrieval JSON,
without re-running the full evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_gw170817a_retrieval import (  # noqa: E402
    plot_redshift_coverage,
    plot_redshift_metrics,
)

RESULT_DIRS = [
    "/fred/oz016/bgao_kn/gw-kn-multimodal/Model/eval_results/gw170817a_lsst_retrieval_balanced",
    "/fred/oz016/bgao_kn/gw-kn-multimodal/Model/eval_results/gw170817a_lsst_retrieval",
]


def main() -> int:
    for result_dir in RESULT_DIRS:
        json_path = Path(result_dir) / "gw170817a_retrieval.json"
        if not json_path.exists():
            print(f"SKIP: {json_path} not found")
            continue
        print(f"Loading {json_path}")
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        redshift_rows = data.get("redshift_rows", [])
        if not redshift_rows:
            print(f"  No redshift_rows found, skipping")
            continue
        print(f"  {len(redshift_rows)} redshift rows, regenerating plots...")
        plot_redshift_metrics(redshift_rows, result_dir)
        plot_redshift_coverage(redshift_rows, result_dir)
        print(f"  Done: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
