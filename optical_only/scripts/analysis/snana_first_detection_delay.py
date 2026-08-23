#!/usr/bin/env python3
"""Build the optical first-detection delay distribution from an ALBEF HDF5 file."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_BASE = Path(os.environ.get("BASE_DIR", "/fred/oz016/bgao_kn"))
OPTICAL_ONLY_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_BASE_DIR = (
    OPTICAL_ONLY_DIR / "outputs" / "h5_first_detection_delay_bns_nsbh"
)
DEFAULT_INPUT_H5 = _BASE / "data" / "ALBEF_dataset" / "combined_dataset_train.h5"
DEFAULT_CANONICAL_OFFSET_NPZ = (
    _BASE / "data" / "Optical_Only_dataset" / "delta_days_distribution.npz"
)
DEFAULT_CHUNK_SIZE = 500_000

GW_GROUP = "events/gw_data"
OPTICAL_GROUP = "events/optical_data"
REQUIRED_GW_DATASETS = ("event_time_mjd", "source_type")
REQUIRED_OPTICAL_DATASETS = ("parent_gw_idx",)
FIRST_DETECTION_CANDIDATES = ("first_detection_mjd", "zero_time_mjd_base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract first-detection delays directly from an ALBEF HDF5 dataset "
            "and overwrite the canonical optical-only offset NPZ."
        )
    )
    parser.add_argument("--input-h5", type=Path, default=DEFAULT_INPUT_H5)
    parser.add_argument("--output-base-dir", type=Path, default=DEFAULT_OUTPUT_BASE_DIR)
    parser.add_argument(
        "--canonical-offset-npz", type=Path, default=DEFAULT_CANONICAL_OFFSET_NPZ
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--save-detailed-deltas-csv", action="store_true")
    return parser.parse_args()


def _decode_strings(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    return values.astype(str)


def _require_datasets(group: h5py.Group, names: Iterable[str], group_path: str) -> None:
    missing = [name for name in names if name not in group]
    if missing:
        raise KeyError(f"Missing datasets under {group_path}: {missing}")


def _first_detection_dataset(optical_group: h5py.Group) -> str:
    for name in FIRST_DETECTION_CANDIDATES:
        if name in optical_group:
            return name
    raise KeyError(f"None of {FIRST_DETECTION_CANDIDATES} exists under {OPTICAL_GROUP}")


def extract_delay_distributions(
    input_h5: Path, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extract delays as first-detection MJD minus the parent GW-event MJD.

    Optical rows are read in chunks so that the large light-curve HDF5 need not
    be loaded into memory. One delay is returned per optical realization.
    """
    input_h5 = Path(input_h5)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not input_h5.is_file():
        raise FileNotFoundError(f"Input HDF5 does not exist: {input_h5}")

    bns_chunks = []
    nsbh_chunks = []

    with h5py.File(input_h5, "r") as handle:
        if GW_GROUP not in handle or OPTICAL_GROUP not in handle:
            raise KeyError(
                f"Input must contain both {GW_GROUP!r} and {OPTICAL_GROUP!r}: "
                f"{input_h5}"
            )

        gw_group = handle[GW_GROUP]
        optical_group = handle[OPTICAL_GROUP]
        _require_datasets(gw_group, REQUIRED_GW_DATASETS, GW_GROUP)
        _require_datasets(optical_group, REQUIRED_OPTICAL_DATASETS, OPTICAL_GROUP)
        detection_name = _first_detection_dataset(optical_group)

        event_time_mjd = np.asarray(gw_group["event_time_mjd"][:], dtype=np.float64)
        source_type = np.char.lower(_decode_strings(gw_group["source_type"][:]))
        parent_dataset = optical_group["parent_gw_idx"]
        detection_dataset = optical_group[detection_name]

        if event_time_mjd.shape != source_type.shape:
            raise ValueError(
                "GW event_time_mjd and source_type lengths differ: "
                f"{event_time_mjd.shape} vs {source_type.shape}"
            )
        if parent_dataset.shape != detection_dataset.shape:
            raise ValueError(
                "Optical parent_gw_idx and first-detection lengths differ: "
                f"{parent_dataset.shape} vs {detection_dataset.shape}"
            )

        n_gw = int(event_time_mjd.size)
        n_optical = int(parent_dataset.shape[0])
        for start in range(0, n_optical, chunk_size):
            stop = min(start + chunk_size, n_optical)
            parent_idx = np.asarray(parent_dataset[start:stop], dtype=np.int64)
            first_detection_mjd = np.asarray(
                detection_dataset[start:stop], dtype=np.float64
            )

            invalid_parent = (parent_idx < 0) | (parent_idx >= n_gw)
            if np.any(invalid_parent):
                bad = parent_idx[invalid_parent][0]
                raise IndexError(
                    f"parent_gw_idx {int(bad)} outside valid range [0, {n_gw})"
                )

            parent_event_time = event_time_mjd[parent_idx]
            finite = np.isfinite(first_detection_mjd) & np.isfinite(parent_event_time)
            if not np.all(finite):
                raise ValueError(
                    f"Found {int((~finite).sum())} non-finite detection/event times "
                    f"in optical rows [{start}, {stop})"
                )

            delays = (first_detection_mjd - parent_event_time).astype(np.float32)
            parent_source = source_type[parent_idx]
            known_source = np.isin(parent_source, ("bns", "nsbh"))
            if not np.all(known_source):
                unknown = np.unique(parent_source[~known_source]).tolist()
                raise ValueError(f"Unsupported parent source_type values: {unknown}")

            bns_chunks.append(delays[parent_source == "bns"])
            nsbh_chunks.append(delays[parent_source == "nsbh"])

        metadata: dict[str, object] = {
            "source_h5": str(input_h5.resolve()),
            "source_h5_mtime_utc": datetime.fromtimestamp(
                input_h5.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "first_detection_dataset": f"{OPTICAL_GROUP}/{detection_name}",
            "delay_definition": "first_detection_mjd - parent_event_time_mjd",
            "n_gw": n_gw,
            "n_optical": n_optical,
            "snr_detection_threshold": float(
                handle.attrs.get("first_detection_snr_threshold", 5.0)
            ),
            "merge_window_hours": float(
                handle.attrs.get("lightcurve_merge_window_hours", 2.0)
            ),
            "fluxcal_to_psfflux_factor": float(
                handle.attrs.get("fluxcal_to_psfflux_factor", np.nan)
            ),
        }

    bns = (
        np.concatenate(bns_chunks).astype(np.float32, copy=False)
        if bns_chunks
        else np.empty(0, dtype=np.float32)
    )
    nsbh = (
        np.concatenate(nsbh_chunks).astype(np.float32, copy=False)
        if nsbh_chunks
        else np.empty(0, dtype=np.float32)
    )
    if bns.size + nsbh.size != metadata["n_optical"]:
        raise RuntimeError(
            "Extracted population counts do not match the optical row count: "
            f"{bns.size} + {nsbh.size} != {metadata['n_optical']}"
        )
    return bns, nsbh, metadata


def summarize_distribution(delta_days: np.ndarray, source: str) -> dict[str, object]:
    row: dict[str, object] = {"source": source, "count": int(delta_days.size)}
    if delta_days.size == 0:
        for key in (
            "mean_days",
            "std_days",
            "q01_days",
            "q05_days",
            "q25_days",
            "q50_days",
            "q75_days",
            "q95_days",
            "q99_days",
            "min_days",
            "max_days",
        ):
            row[key] = np.nan
        return row

    quantiles = np.quantile(delta_days, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    row.update(
        {
            "mean_days": float(np.mean(delta_days)),
            "std_days": float(np.std(delta_days)),
            "q01_days": float(quantiles[0]),
            "q05_days": float(quantiles[1]),
            "q25_days": float(quantiles[2]),
            "q50_days": float(quantiles[3]),
            "q75_days": float(quantiles[4]),
            "q95_days": float(quantiles[5]),
            "q99_days": float(quantiles[6]),
            "min_days": float(np.min(delta_days)),
            "max_days": float(np.max(delta_days)),
        }
    )
    return row


def _atomic_savez(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}.", suffix=".npz", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        np.savez_compressed(temporary_path, **payload)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_outputs(
    output_base_dir: Path,
    canonical_offset_npz: Path,
    bns: np.ndarray,
    nsbh: np.ndarray,
    metadata: dict[str, object],
    save_detailed_deltas_csv: bool,
) -> dict[str, Path]:
    combined = np.concatenate((bns, nsbh)).astype(np.float32, copy=False)
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_base_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=False)

    payload: dict[str, object] = {
        "delta_days_bns": bns,
        "delta_days_nsbh": nsbh,
        "delta_days_combined": combined,
        "snr_detection_threshold": np.float32(metadata["snr_detection_threshold"]),
        "merge_window_hours": np.float32(metadata["merge_window_hours"]),
        "fluxcal_to_psfflux_factor": np.float32(metadata["fluxcal_to_psfflux_factor"]),
        "source_h5": np.asarray(metadata["source_h5"]),
        "source_h5_mtime_utc": np.asarray(metadata["source_h5_mtime_utc"]),
        "first_detection_dataset": np.asarray(metadata["first_detection_dataset"]),
        "delay_definition": np.asarray(metadata["delay_definition"]),
    }
    run_npz = run_dir / "delta_days_distribution.npz"
    _atomic_savez(run_npz, payload)
    _atomic_savez(canonical_offset_npz, payload)

    combined_summary = pd.DataFrame(
        [
            summarize_distribution(bns, "BNS"),
            summarize_distribution(nsbh, "NSBH"),
            summarize_distribution(combined, "Combined"),
        ]
    )
    summary_csv = run_dir / "summary_stats.csv"
    combined_summary.to_csv(summary_csv, index=False)

    if save_detailed_deltas_csv:
        pd.concat(
            [
                pd.DataFrame({"source": "BNS", "delta_days": bns}),
                pd.DataFrame({"source": "NSBH", "delta_days": nsbh}),
            ],
            ignore_index=True,
        ).to_csv(run_dir / "detected_delta_days_detailed.csv", index=False)

    plot_path = run_dir / "first_detection_delay_histogram.png"
    if combined.size:
        lower, upper = np.quantile(combined, [0.001, 0.999])
        bins = np.linspace(float(lower), float(upper), 150)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(combined, bins=bins, density=True, histtype="step", label="Combined")
        ax.hist(bns, bins=bins, density=True, histtype="step", label="BNS")
        ax.hist(nsbh, bins=bins, density=True, histtype="step", label="NSBH")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("First optical detection minus GW trigger (days)")
        ax.set_ylabel("Probability density")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

    latest_run_path = output_base_dir / "latest_run.txt"
    latest_run_path.write_text(str(run_dir), encoding="utf-8")
    return {
        "run_dir": run_dir,
        "run_npz": run_npz,
        "canonical_npz": canonical_offset_npz,
        "summary_csv": summary_csv,
        "plot": plot_path,
    }


def main() -> None:
    args = parse_args()
    bns, nsbh, metadata = extract_delay_distributions(
        args.input_h5, chunk_size=args.chunk_size
    )
    paths = save_outputs(
        output_base_dir=args.output_base_dir,
        canonical_offset_npz=args.canonical_offset_npz,
        bns=bns,
        nsbh=nsbh,
        metadata=metadata,
        save_detailed_deltas_csv=args.save_detailed_deltas_csv,
    )
    print("BNS optical realizations:", bns.size)
    print("NSBH optical realizations:", nsbh.size)
    print("Combined optical realizations:", bns.size + nsbh.size)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
