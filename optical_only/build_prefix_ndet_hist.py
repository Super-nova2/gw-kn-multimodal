#!/usr/bin/env python3
"""
Build a real-stream detection-count histogram for prefix training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CANDIDATE_COLUMNS = (
    "n_detections",
    "n_detections_snr5",
    "n_det",
    "actual_n_det_snr5",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build empirical n_det histogram for prefix training.")
    parser.add_argument("--input", type=str, required=True, help="CSV or parquet file from real-stream inference.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--column", type=str, default=None, help="Optional explicit n_det column name.")
    parser.add_argument("--min_det", type=int, default=2)
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format for {path}; expected CSV or parquet.")


def resolve_column(df: pd.DataFrame, explicit: str | None) -> str:
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f"Requested column '{explicit}' not found.")
        return explicit
    for name in DEFAULT_CANDIDATE_COLUMNS:
        if name in df.columns:
            return name
    raise KeyError(
        "Could not find an n_det column. Tried: "
        + ", ".join(DEFAULT_CANDIDATE_COLUMNS)
    )


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = load_table(input_path)
    column = resolve_column(df, args.column)
    det = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
    det = det[np.isfinite(det)]
    det = np.rint(det).astype(np.int64, copy=False)
    det = det[det >= int(args.min_det)]
    if det.size == 0:
        raise ValueError(f"No valid detection counts >= {int(args.min_det)} found in {input_path}.")

    unique, counts = np.unique(det, return_counts=True)
    probs = counts.astype(np.float64) / float(np.sum(counts))

    payload = {
        "source_path": str(input_path),
        "source_column": str(column),
        "min_det": int(args.min_det),
        "n_objects": int(det.size),
        "counts": {str(int(k)): int(v) for k, v in zip(unique.tolist(), counts.tolist())},
        "probabilities": {str(int(k)): float(v) for k, v in zip(unique.tolist(), probs.tolist())},
        "summary": {
            "det_min": int(np.min(det)),
            "det_max": int(np.max(det)),
            "det_mean": float(np.mean(det)),
            "det_median": float(np.median(det)),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
