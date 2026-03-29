#!/usr/bin/env python3
"""
Plot representative preprocessed optical-only test-set light curves in luptitude space.

The HDF5 files already store:
- values: luptitude
- errors: luptitude_sigma
- times: first-detection aligned, scaled by time_scale_divisor_days

This script selects "typical" positive and negative samples by:
1. ranking all samples with stored meta features (stored `meta_n_det` is treated as retained `Nobs`, not `Ndet`),
2. refining a shortlist with light-curve summary statistics in luptitude space,
3. plotting the closest samples to the class center.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BANDS: Sequence[str] = ("u", "g", "r", "i", "z", "Y")
BAND_COLORS: Dict[str, str] = {
    "u": "#3B82F6",
    "g": "#10B981",
    "r": "#EF4444",
    "i": "#F59E0B",
    "z": "#8B5CF6",
    "Y": "#6B7280",
}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POS_H5 = Path("<BASE_DIR>/data/Optical_Only_dataset/combined_dataset_test.h5")
DEFAULT_NEG_H5 = Path("<BASE_DIR>/data/Optical_Only_dataset/Tutorial_negative_dataset.h5")
DEFAULT_NEG_GROUP = "Tutorial/optical_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "typical_test_lightcurves"
ASINH_MAG_FACTOR = 2.5 / np.log(10.0)


@dataclass
class DatasetConfig:
    class_name: str
    h5_path: Path
    group: str
    unique_mode: Optional[str]


@dataclass
class SelectionResult:
    class_name: str
    sample_index: int
    meta_score: float
    typical_score: float
    source_label: str
    source_id: str
    parent_event_idx: Optional[int]
    n_obs: int
    n_det_snr5: int
    n_bands: int
    t_span_scaled: float
    t_span_days: float
    mean_lupt: float
    std_lupt: float
    amp_lupt: float
    mean_err: float
    plot_path: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot typical preprocessed optical-only test-set light curves in luptitude space."
    )
    parser.add_argument("--pos-h5", type=Path, default=DEFAULT_POS_H5)
    parser.add_argument("--neg-h5", type=Path, default=DEFAULT_NEG_H5)
    parser.add_argument("--neg-group", type=str, default=DEFAULT_NEG_GROUP)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument(
        "--shortlist-size",
        type=int,
        default=4096,
        help="Per-class shortlist size used before luptitude-stat refinement.",
    )
    parser.add_argument(
        "--plot-time-unit",
        choices=("scaled", "days"),
        default="days",
        help="Use stored scaled time or convert back to days with time_scale_divisor_days.",
    )
    parser.add_argument(
        "--negative-selection-mode",
        choices=("per_type_typical", "overall_typical"),
        default="per_type_typical",
        help="How to choose negative examples: one representative per negative type, or overall typical negatives.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def decode_scalar(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def robust_scales(features: np.ndarray) -> np.ndarray:
    med = np.nanmedian(features, axis=0)
    mad = np.nanmedian(np.abs(features - med), axis=0)
    std = np.nanstd(features, axis=0)
    scale = np.where(mad > 1.0e-6, mad, np.where(std > 1.0e-6, std, 1.0))
    return scale.astype(np.float64, copy=False)


def robust_scores(features: np.ndarray) -> np.ndarray:
    center = np.nanmedian(features, axis=0)
    scale = robust_scales(features)
    z = (features - center) / scale
    return np.sqrt(np.sum(np.square(z), axis=1, dtype=np.float64), dtype=np.float64)


def compute_lc_summary_for_indices(grp: h5py.Group, sample_indices: np.ndarray) -> np.ndarray:
    sorted_pos = np.argsort(sample_indices)
    sorted_indices = np.asarray(sample_indices[sorted_pos], dtype=np.int64)

    values = np.asarray(grp["values"][sorted_indices], dtype=np.float32)
    errors = np.asarray(grp["errors"][sorted_indices], dtype=np.float32)
    masks = np.asarray(grp["masks"][sorted_indices], dtype=np.float32) > 0.0

    counts = masks.sum(axis=(1, 2)).astype(np.float32)
    counts_safe = np.where(counts > 0.0, counts, 1.0)

    values_masked = np.where(masks, values, 0.0)
    errors_masked = np.where(masks, errors, 0.0)

    mean_lupt = values_masked.sum(axis=(1, 2), dtype=np.float64) / counts_safe
    sq_mean = np.square(values_masked, dtype=np.float64).sum(axis=(1, 2), dtype=np.float64) / counts_safe
    std_lupt = np.sqrt(np.maximum(0.0, sq_mean - np.square(mean_lupt, dtype=np.float64)))
    amp_lupt = (
        np.max(np.where(masks, values, -np.inf), axis=(1, 2)).astype(np.float64)
        - np.min(np.where(masks, values, np.inf), axis=(1, 2)).astype(np.float64)
    )
    mean_err = errors_masked.sum(axis=(1, 2), dtype=np.float64) / counts_safe

    summary_sorted = np.column_stack([mean_lupt, std_lupt, amp_lupt, mean_err]).astype(np.float64, copy=False)
    summary = np.empty_like(summary_sorted)
    summary[sorted_pos] = summary_sorted
    return summary


def compute_observation_and_detection_counts(
    h5f: h5py.File,
    grp: h5py.Group,
    sample_index: int,
) -> tuple[int, int]:
    values = np.asarray(grp["values"][sample_index], dtype=np.float64)
    errors = np.asarray(grp["errors"][sample_index], dtype=np.float64)
    masks = np.asarray(grp["masks"][sample_index], dtype=np.float32) > 0.0
    n_obs = int(np.count_nonzero(masks))
    if n_obs < 1:
        return 0, 0

    lupt_b_njy = np.asarray(h5f.attrs["lupt_b_njy"], dtype=np.float64).reshape(-1)
    psfflux_zp = float(h5f.attrs["psfflux_zp"])
    snr_threshold = float(h5f.attrs.get("snr_threshold", 5.0))

    band_grid = np.broadcast_to(np.arange(values.shape[1], dtype=np.int64), values.shape)
    valid = masks & np.isfinite(values) & np.isfinite(errors) & (errors > 0.0)
    if not np.any(valid):
        return n_obs, 0

    m_lupt = values[valid]
    sigma_lupt = errors[valid]
    band_idx = band_grid[valid]
    b_valid = lupt_b_njy[band_idx]

    asinh_arg = (psfflux_zp - m_lupt) / ASINH_MAG_FACTOR - np.log(b_valid)
    f_psf = 2.0 * b_valid * np.sinh(asinh_arg)
    sigma_psf = sigma_lupt * np.sqrt((f_psf * f_psf) + (2.0 * b_valid) ** 2) / ASINH_MAG_FACTOR

    snr = np.full(f_psf.shape, -np.inf, dtype=np.float64)
    finite = np.isfinite(f_psf) & np.isfinite(sigma_psf) & (sigma_psf > 0.0)
    snr[finite] = f_psf[finite] / sigma_psf[finite]
    n_det_snr5 = int(np.count_nonzero(snr > snr_threshold))
    return n_obs, n_det_snr5


def select_typical_subset(
    *,
    h5f: h5py.File,
    grp: h5py.Group,
    cfg: DatasetConfig,
    time_scale: float,
    n_obs_all: np.ndarray,
    n_bands_all: np.ndarray,
    t_span_scaled_all: np.ndarray,
    candidate_indices: np.ndarray,
    max_results: int,
    shortlist_size: int,
    source_label_override: Optional[str] = None,
) -> List[SelectionResult]:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_indices.size < 1:
        return []

    meta_features = np.column_stack(
        [
            n_obs_all[candidate_indices],
            n_bands_all[candidate_indices],
            t_span_scaled_all[candidate_indices],
        ]
    ).astype(np.float64, copy=False)
    meta_scores_subset = robust_scores(meta_features)
    shortlist_n = min(candidate_indices.size, max(int(shortlist_size), int(max_results) * 32))
    shortlist_local = np.argsort(meta_scores_subset)[:shortlist_n]
    shortlist = candidate_indices[shortlist_local]

    lc_summary = compute_lc_summary_for_indices(grp, shortlist)
    combined_features = np.column_stack([meta_features[shortlist_local], lc_summary]).astype(np.float64, copy=False)
    combined_scores = robust_scores(combined_features)
    order = np.argsort(combined_scores)

    if cfg.unique_mode == "parent_event_idx":
        if "parent_event_idx" in grp:
            parent_event_idx_all = np.asarray(grp["parent_event_idx"][:], dtype=np.int64)
            unique_values = parent_event_idx_all[shortlist]
        elif "parent_gw_idx" in grp:
            parent_event_idx_all = np.asarray(grp["parent_gw_idx"][:], dtype=np.int64)
            unique_values = parent_event_idx_all[shortlist]
        else:
            unique_values = None
    else:
        unique_values = None

    selected_short_positions: List[int] = []
    seen_unique = set()
    for short_pos in order.tolist():
        if unique_values is not None:
            unique_key = int(unique_values[short_pos])
            if unique_key in seen_unique:
                continue
            seen_unique.add(unique_key)
        selected_short_positions.append(short_pos)
        if len(selected_short_positions) >= max_results:
            break

    if len(selected_short_positions) < max_results:
        selected_set = set(selected_short_positions)
        for short_pos in order.tolist():
            if short_pos in selected_set:
                continue
            selected_short_positions.append(short_pos)
            selected_set.add(short_pos)
            if len(selected_short_positions) >= max_results:
                break

    results: List[SelectionResult] = []
    for short_pos in selected_short_positions:
        sample_idx = int(shortlist[short_pos])
        parent_event_idx: Optional[int] = None
        source_label = ""
        source_id = ""

        if cfg.class_name == "positive":
            if "parent_event_idx" in grp:
                parent_event_idx = int(grp["parent_event_idx"][sample_idx])
            else:
                parent_event_idx = int(grp["parent_gw_idx"][sample_idx])
        else:
            source_label = source_label_override or (
                decode_scalar(grp["types"][sample_idx]) if "types" in grp else "unknown"
            )
        n_obs, n_det_snr5 = compute_observation_and_detection_counts(h5f=h5f, grp=grp, sample_index=sample_idx)

        results.append(
            SelectionResult(
                class_name=cfg.class_name,
                sample_index=sample_idx,
                meta_score=float(meta_scores_subset[shortlist_local[short_pos]]),
                typical_score=float(combined_scores[short_pos]),
                source_label=source_label,
                source_id=source_id,
                parent_event_idx=parent_event_idx,
                n_obs=int(n_obs),
                n_det_snr5=int(n_det_snr5),
                n_bands=int(n_bands_all[sample_idx]),
                t_span_scaled=float(t_span_scaled_all[sample_idx]),
                t_span_days=float(t_span_scaled_all[sample_idx] * time_scale),
                mean_lupt=float(lc_summary[short_pos, 0]),
                std_lupt=float(lc_summary[short_pos, 1]),
                amp_lupt=float(lc_summary[short_pos, 2]),
                mean_err=float(lc_summary[short_pos, 3]),
            )
        )

    results.sort(key=lambda item: item.typical_score)
    return results


def select_typical_samples(
    cfg: DatasetConfig,
    samples_per_class: int,
    shortlist_size: int,
    negative_selection_mode: str,
) -> List[SelectionResult]:
    with h5py.File(cfg.h5_path, "r") as h5f:
        grp = h5f[cfg.group]
        n_total = int(grp["values"].shape[0])
        if n_total < 1:
            raise ValueError(f"No samples found in {cfg.h5_path}:{cfg.group}")

        time_scale = float(h5f.attrs.get("time_scale_divisor_days", 1.0))
        # HDF5 field name kept for backward compatibility, but this quantity is
        # the number of retained observation points after formatting, not SNR>5 detections.
        n_obs_meta = np.asarray(grp["meta_n_det"][:], dtype=np.float64)
        n_bands = np.asarray(grp["meta_n_bands"][:], dtype=np.float64)
        t_span_scaled = np.asarray(grp["meta_t_span"][:], dtype=np.float64)

        if cfg.class_name == "negative" and "types" in grp and negative_selection_mode == "per_type_typical":
            type_labels = np.asarray([decode_scalar(v) for v in grp["types"][:]], dtype=object)
            unique_labels, counts = np.unique(type_labels, return_counts=True)
            label_order = [str(lbl) for lbl in unique_labels[np.argsort(-counts, kind="stable")]]
            results: List[SelectionResult] = []
            for label in label_order:
                type_indices = np.flatnonzero(type_labels == label)
                results.extend(
                    select_typical_subset(
                        h5f=h5f,
                        grp=grp,
                        cfg=cfg,
                        time_scale=time_scale,
                        n_obs_all=n_obs_meta,
                        n_bands_all=n_bands,
                        t_span_scaled_all=t_span_scaled,
                        candidate_indices=type_indices,
                        max_results=1,
                        shortlist_size=shortlist_size,
                        source_label_override=label,
                    )
                )
            return results

        return select_typical_subset(
            h5f=h5f,
            grp=grp,
            cfg=cfg,
            time_scale=time_scale,
            n_obs_all=n_obs_meta,
            n_bands_all=n_bands,
            t_span_scaled_all=t_span_scaled,
            candidate_indices=np.arange(n_total, dtype=np.int64),
            max_results=samples_per_class,
            shortlist_size=shortlist_size,
        )


def extract_sample_plot_data(
    h5_path: Path,
    group: str,
    sample_index: int,
    plot_time_unit: str,
) -> Dict[str, object]:
    with h5py.File(h5_path, "r") as h5f:
        grp = h5f[group]
        values = np.asarray(grp["values"][sample_index], dtype=np.float64)
        errors = np.asarray(grp["errors"][sample_index], dtype=np.float64)
        masks = np.asarray(grp["masks"][sample_index], dtype=np.float32) > 0.0
        times_scaled = np.asarray(grp["times"][sample_index], dtype=np.float64)
        time_scale = float(h5f.attrs.get("time_scale_divisor_days", 1.0))
        if plot_time_unit == "days":
            times = times_scaled * time_scale
            x_label = "Time since first detection [days]"
        else:
            times = times_scaled
            x_label = "Scaled time since first detection [days / time_scale_divisor_days]"

        return {
            "values": values,
            "errors": errors,
            "masks": masks,
            "times": times,
            "x_label": x_label,
            "time_scale_divisor_days": time_scale,
            "photometry_representation": decode_scalar(h5f.attrs.get("photometry_representation", "unknown")),
            "values_semantics": decode_scalar(h5f.attrs.get("values_semantics", "unknown")),
        }


def make_title(result: SelectionResult, rank: int) -> str:
    label_bits = [f"{result.class_name.capitalize()} #{rank}", f"idx={result.sample_index}"]
    if result.class_name == "positive":
        if result.parent_event_idx is not None:
            label_bits.append(f"parent_event={result.parent_event_idx}")
        if result.source_label:
            label_bits.append(result.source_label)
    else:
        if result.source_label:
            label_bits.append(result.source_label)

    line1 = " | ".join(label_bits)
    line2 = (
        f"Nobs={result.n_obs}, Ndet(SNR>5)={result.n_det_snr5}, Nbands={result.n_bands}, span={result.t_span_days:.1f} d, "
        f"score={result.typical_score:.2f}"
    )
    return f"{line1}\n{line2}"


def build_band_legend_handles() -> List[Line2D]:
    handles: List[Line2D] = []
    for band_name in BANDS:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=6.0,
                markerfacecolor=BAND_COLORS[band_name],
                markeredgecolor=BAND_COLORS[band_name],
                label=f"{band_name}-band",
            )
        )
    return handles


def plot_single_sample(ax: plt.Axes, result: SelectionResult, sample_data: Dict[str, object]) -> None:
    values = sample_data["values"]
    errors = sample_data["errors"]
    masks = sample_data["masks"]
    times = sample_data["times"]

    for band_idx, band_name in enumerate(BANDS):
        valid = masks[:, band_idx]
        if not np.any(valid):
            continue
        x = np.asarray(times[valid], dtype=np.float64)
        y = np.asarray(values[valid, band_idx], dtype=np.float64)
        yerr = np.asarray(errors[valid, band_idx], dtype=np.float64)
        sort_idx = np.argsort(x)
        ax.errorbar(
            x[sort_idx],
            y[sort_idx],
            yerr=yerr[sort_idx],
            fmt="o",
            linestyle="none",
            ms=3.2,
            elinewidth=0.95,
            capsize=2.0,
            alpha=0.9,
            color=BAND_COLORS[band_name],
            ecolor=BAND_COLORS[band_name],
        )

    ax.axvline(0.0, color="#111827", lw=0.9, ls="--", alpha=0.7)
    ax.grid(True, alpha=0.22, lw=0.6)
    ax.invert_yaxis()
    ax.set_title(make_title(result, rank=0), fontsize=9)
    ax.set_xlabel(sample_data["x_label"])
    ax.set_ylabel("Luptitude")


def save_individual_plots(
    results_by_class: Dict[str, List[SelectionResult]],
    configs: Dict[str, DatasetConfig],
    output_dir: Path,
    plot_time_unit: str,
    dpi: int,
) -> None:
    single_dir = output_dir / "individual"
    single_dir.mkdir(parents=True, exist_ok=True)
    band_legend_handles = build_band_legend_handles()

    for class_name, results in results_by_class.items():
        for rank, result in enumerate(results, start=1):
            sample_data = extract_sample_plot_data(
                h5_path=configs[class_name].h5_path,
                group=configs[class_name].group,
                sample_index=result.sample_index,
                plot_time_unit=plot_time_unit,
            )
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            plot_single_sample(ax, result, sample_data)
            ax.set_title(make_title(result, rank=rank), fontsize=10)
            ax.legend(handles=band_legend_handles, loc="best", ncol=3, fontsize=8, frameon=False, title="Bands")
            fig.tight_layout()

            filename = f"{class_name}_rank_{rank:02d}_idx_{result.sample_index}.png"
            out_path = single_dir / filename
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            result.plot_path = out_path


def save_combined_plot(
    results_by_class: Dict[str, List[SelectionResult]],
    configs: Dict[str, DatasetConfig],
    output_dir: Path,
    plot_time_unit: str,
    dpi: int,
) -> Path:
    samples_per_class = max(len(v) for v in results_by_class.values())
    fig, axes = plt.subplots(
        2,
        samples_per_class,
        figsize=(4.8 * samples_per_class, 8.0),
        squeeze=False,
        sharey=False,
    )

    class_order = ("positive", "negative")
    band_legend_handles = build_band_legend_handles()

    for row, class_name in enumerate(class_order):
        results = results_by_class.get(class_name, [])
        for col in range(samples_per_class):
            ax = axes[row, col]
            if col >= len(results):
                ax.axis("off")
                continue
            result = results[col]
            sample_data = extract_sample_plot_data(
                h5_path=configs[class_name].h5_path,
                group=configs[class_name].group,
                sample_index=result.sample_index,
                plot_time_unit=plot_time_unit,
            )
            plot_single_sample(ax, result, sample_data)
            ax.set_title(make_title(result, rank=col + 1), fontsize=9)

    title = "Representative optical-only test-set light curves (preprocessed luptitude)"
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.legend(band_legend_handles, [h.get_label() for h in band_legend_handles], loc="upper center", ncol=len(BANDS), frameon=False, title="Bands")
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    out_path = output_dir / "typical_luptitude_lightcurves.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_manifest(manifest_path: Path, results_by_class: Dict[str, List[SelectionResult]]) -> None:
    fieldnames = [
        "class_name",
        "rank",
        "sample_index",
        "source_label",
        "source_id",
        "parent_event_idx",
        "n_obs",
        "n_det_snr5",
        "n_bands",
        "t_span_scaled",
        "t_span_days",
        "mean_lupt",
        "std_lupt",
        "amp_lupt",
        "mean_err",
        "meta_score",
        "typical_score",
        "plot_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for class_name in ("positive", "negative"):
            for rank, result in enumerate(results_by_class.get(class_name, []), start=1):
                writer.writerow(
                    {
                        "class_name": result.class_name,
                        "rank": rank,
                        "sample_index": result.sample_index,
                        "source_label": result.source_label,
                        "source_id": result.source_id,
                        "parent_event_idx": "" if result.parent_event_idx is None else result.parent_event_idx,
                        "n_obs": result.n_obs,
                        "n_det_snr5": result.n_det_snr5,
                        "n_bands": result.n_bands,
                        "t_span_scaled": f"{result.t_span_scaled:.6f}",
                        "t_span_days": f"{result.t_span_days:.6f}",
                        "mean_lupt": f"{result.mean_lupt:.6f}",
                        "std_lupt": f"{result.std_lupt:.6f}",
                        "amp_lupt": f"{result.amp_lupt:.6f}",
                        "mean_err": f"{result.mean_err:.6f}",
                        "meta_score": f"{result.meta_score:.6f}",
                        "typical_score": f"{result.typical_score:.6f}",
                        "plot_path": "" if result.plot_path is None else str(result.plot_path),
                    }
                )


def validate_inputs(args: argparse.Namespace) -> None:
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be >= 1")
    if args.shortlist_size < 8:
        raise ValueError("--shortlist-size must be >= 8")
    if not args.pos_h5.exists():
        raise FileNotFoundError(f"Positive HDF5 not found: {args.pos_h5}")
    if not args.neg_h5.exists():
        raise FileNotFoundError(f"Negative HDF5 not found: {args.neg_h5}")


def main() -> None:
    args = parse_args()
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "positive": DatasetConfig(
            class_name="positive",
            h5_path=args.pos_h5,
            group="events/optical_data",
            unique_mode="parent_event_idx",
        ),
        "negative": DatasetConfig(
            class_name="negative",
            h5_path=args.neg_h5,
            group=args.neg_group,
            unique_mode=None,
        ),
    }

    results_by_class = {
        class_name: select_typical_samples(
            cfg=cfg,
            samples_per_class=args.samples_per_class,
            shortlist_size=args.shortlist_size,
            negative_selection_mode=args.negative_selection_mode,
        )
        for class_name, cfg in configs.items()
    }

    save_individual_plots(
        results_by_class=results_by_class,
        configs=configs,
        output_dir=args.output_dir,
        plot_time_unit=args.plot_time_unit,
        dpi=args.dpi,
    )
    combined_plot_path = save_combined_plot(
        results_by_class=results_by_class,
        configs=configs,
        output_dir=args.output_dir,
        plot_time_unit=args.plot_time_unit,
        dpi=args.dpi,
    )
    manifest_path = args.output_dir / "typical_luptitude_lightcurves_manifest.csv"
    write_manifest(manifest_path=manifest_path, results_by_class=results_by_class)

    print(f"Saved combined plot: {combined_plot_path}")
    print(f"Saved manifest: {manifest_path}")
    for class_name in ("positive", "negative"):
        for rank, result in enumerate(results_by_class[class_name], start=1):
            print(
                f"{class_name:>8s} rank {rank:02d} | idx={result.sample_index:<7d} "
                f"| label={result.source_label:<8s} | n_obs={result.n_obs:<3d} "
                f"| n_det_snr5={result.n_det_snr5:<3d} "
                f"| span_days={result.t_span_days:5.1f} | score={result.typical_score:6.3f}"
            )


if __name__ == "__main__":
    main()
