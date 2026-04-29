from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import h5py
import numpy as np
import torch


TABLE_METRIC_LABELS = ["R@1", "R@5", "R@10", "MRR"]
TABLE_METRIC_KEYS = ["recall_at_1", "recall_at_5", "recall_at_10", "mrr"]
PLOT_DPI = 300
PLOT_METHOD_LABELS = {
    "skymap-only": "Skymap-only",
    "optical-only": "Optical-only",
    "w/o \u5bf9\u6bd4\u5b66\u4e60": "w/o Contrastive Learning",
    "w/o \u4ea4\u53c9\u6ce8\u610f\u529b": "w/o Cross-Attention",
    "w/o \u878d\u5408\u5206\u652f": "w/o Fusion Branch",
    "\u5168\u6a21\u6001": "Full Multimodal",
    "\u5168\u6a21\u6001 + \u56f0\u96be\u6837\u672c\u6316\u6398": "Full Multimodal + Hard Mining",
}


def _plot_method_label(method: str) -> str:
    return PLOT_METHOD_LABELS.get(str(method), str(method))


def _remove_stale_pdf(path: Path) -> None:
    if path.exists():
        path.unlink()


def _as_numpy_1d(value: Any, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    return np.asarray(arr, dtype=dtype).reshape(-1)


def resolve_optional_path(base_dir: Path, value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def resolve_checkpoint_path(checkpoint_path: str, model_type: str) -> str:
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        return str(path.resolve())
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Unsupported checkpoint path: {checkpoint_path}")

    preferred_names = ["optical_only_best.pth", "best.pth"] if model_type == "optical" else ["albef_best.pth", "best.pth"]
    for filename in preferred_names:
        matches = sorted(path.rglob(filename))
        if matches:
            return str(matches[0].resolve())

    epoch_candidates: List[Tuple[int, Path]] = []
    epoch_patterns = [
        re.compile(r"albef_epoch_(\d+)\.pth$"),
        re.compile(r"optical_only_epoch_(\d+)\.pth$"),
    ]
    for candidate in path.rglob("*.pth"):
        for pattern in epoch_patterns:
            match = pattern.fullmatch(candidate.name)
            if match:
                epoch_candidates.append((int(match.group(1)), candidate))
                break
    if epoch_candidates:
        return str(max(epoch_candidates, key=lambda item: item[0])[1].resolve())

    fallback_names = ["optical_only_last.pth", "albef_last.pth", "last.pth"]
    for filename in fallback_names:
        matches = sorted(path.rglob(filename))
        if matches:
            return str(matches[0].resolve())

    any_pth = sorted(path.rglob("*.pth"))
    if any_pth:
        return str(any_pth[0].resolve())

    raise FileNotFoundError(f"No checkpoint file found under directory: {checkpoint_path}")


def build_comparison_model_specs(cfg: Mapping[str, Any], cfg_dir: Path) -> List[Dict[str, Any]]:
    model_specs: List[Dict[str, Any]] = []
    valid_scoring = {"optical", "logits", "contrastive", "auto", "skymap"}
    for raw_spec in cfg.get("models", []):
        spec = dict(raw_spec)
        spec["name"] = str(spec["name"])
        spec["type"] = str(spec["type"])
        spec["checkpoint"] = resolve_optional_path(cfg_dir, spec.get("checkpoint"))
        spec["config"] = resolve_optional_path(cfg_dir, spec.get("config"))
        if spec["type"] in {"optical", "multimodal"} and spec["checkpoint"] is None:
            raise ValueError(f"models[{spec['name']}] is missing checkpoint")
        if spec["type"] == "multimodal" and spec["config"] is None:
            raise ValueError(f"Multimodal model '{spec['name']}' requires a config path")

        if spec["type"] == "skymap":
            spec["resolved_checkpoint"] = None
        else:
            spec["resolved_checkpoint"] = resolve_checkpoint_path(spec["checkpoint"], spec["type"])
        spec["resolved_config"] = spec.get("config")

        if "scoring" not in spec:
            if spec["type"] == "optical":
                spec["scoring"] = "optical"
            elif spec["type"] == "skymap":
                spec["scoring"] = "skymap"
            else:
                spec["scoring"] = "auto"
        if spec["scoring"] not in valid_scoring:
            raise ValueError(
                f"models[{spec['name']}] has invalid scoring '{spec['scoring']}', "
                f"must be one of {valid_scoring}"
            )
        model_specs.append(spec)

    if not model_specs:
        raise ValueError("Comparison config must define a non-empty models array")
    return model_specs


def _as_torch_coords(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        coords = value.detach().cpu().to(torch.float32)
    else:
        coords = torch.as_tensor(value, dtype=torch.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coordinate array with shape [N, 2], got {tuple(coords.shape)}")
    return coords


def _coords_to_radians(coords: torch.Tensor) -> torch.Tensor:
    if coords.numel() == 0:
        return coords
    max_abs = float(coords.detach().abs().max().item())
    if max_abs > (2 * np.pi + 1e-3):
        return coords * (np.pi / 180.0)
    return coords


def _coords_to_unit_xyz(coords: torch.Tensor) -> torch.Tensor:
    coords = _coords_to_radians(coords.to(torch.float32))
    if coords.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=coords.device)
    ra = coords[:, 0]
    dec = coords[:, 1]
    return torch.stack(
        [
            torch.cos(dec) * torch.cos(ra),
            torch.cos(dec) * torch.sin(ra),
            torch.sin(dec),
        ],
        dim=-1,
    )


def compute_credible_levels_single_gw(gw_skymap: torch.Tensor, opt_coords: torch.Tensor) -> torch.Tensor:
    opt_xyz = _coords_to_unit_xyz(opt_coords)

    pix_xyz = gw_skymap[:3, :].to(torch.float32)
    dot = torch.matmul(opt_xyz, pix_xyz)
    nearest_idx = dot.argmax(dim=-1)

    dP = gw_skymap[4, :].to(torch.float32)
    dP_at_opt = dP[nearest_idx]
    cred_level = (dP.unsqueeze(0) >= dP_at_opt.unsqueeze(-1)).float().mean(dim=-1)
    return cred_level


def _nearest_pixel_indices_from_xyz(opt_xyz: torch.Tensor, pixel_xyz: torch.Tensor, chunk_size: int = 4096) -> np.ndarray:
    pixel_xyz = pixel_xyz.to(torch.float32)

    nearest_chunks: List[torch.Tensor] = []
    for start in range(0, opt_xyz.shape[0], int(chunk_size)):
        chunk = opt_xyz[start : start + int(chunk_size)]
        nearest_chunks.append(torch.matmul(chunk, pixel_xyz).argmax(dim=-1).cpu())
    if not nearest_chunks:
        return np.empty((0,), dtype=np.int64)
    return torch.cat(nearest_chunks).numpy().astype(np.int64, copy=False)


def _nearest_pixel_indices(opt_coords: torch.Tensor, pixel_xyz: torch.Tensor, chunk_size: int = 4096) -> np.ndarray:
    return _nearest_pixel_indices_from_xyz(_coords_to_unit_xyz(opt_coords), pixel_xyz, chunk_size=int(chunk_size))


def build_time_sky_candidate_sequence(
    *,
    anchor_time_mjd: float,
    candidate_zero_time_mjd: Any,
    candidate_credible_levels: Any,
    time_window_days: float,
    credible_level_max: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    candidate_times = _as_numpy_1d(candidate_zero_time_mjd, dtype=np.float64)
    candidate_credible = _as_numpy_1d(candidate_credible_levels, dtype=np.float64)
    if candidate_times.shape != candidate_credible.shape:
        raise ValueError("candidate_zero_time_mjd and candidate_credible_levels must have the same shape.")

    if not np.isfinite(anchor_time_mjd):
        return {
            "candidate_indices": np.empty((0,), dtype=np.int64),
            "credible_levels": np.empty((0,), dtype=np.float32),
            "abs_dt_days": np.empty((0,), dtype=np.float32),
        }

    abs_dt_days = np.abs(candidate_times - float(anchor_time_mjd))
    keep = (
        np.isfinite(candidate_times)
        & np.isfinite(candidate_credible)
        & (abs_dt_days <= float(time_window_days))
        & (candidate_credible <= float(credible_level_max))
    )

    valid_idx = np.nonzero(keep)[0].astype(np.int64, copy=False)
    if valid_idx.size == 0:
        return {
            "candidate_indices": np.empty((0,), dtype=np.int64),
            "credible_levels": np.empty((0,), dtype=np.float32),
            "abs_dt_days": np.empty((0,), dtype=np.float32),
        }

    rng = np.random.default_rng(int(seed))
    tie_break = rng.random(valid_idx.size)
    order = np.lexsort(
        (
            tie_break,
            candidate_credible[valid_idx],
            abs_dt_days[valid_idx],
        )
    )
    ordered_idx = valid_idx[order]
    return {
        "candidate_indices": ordered_idx.astype(np.int64, copy=False),
        "credible_levels": np.asarray(candidate_credible[ordered_idx], dtype=np.float32),
        "abs_dt_days": np.asarray(abs_dt_days[ordered_idx], dtype=np.float32),
    }


def build_time_sky_candidate_sequences(
    *,
    test_data_path: str,
    unique_gw_ids: Sequence[int],
    neg_optical_data: Mapping[str, Any],
    n_trials: int,
    seed: int,
    time_window_days: float,
    credible_level_max: float,
    zero_time_field: str = "zero_time_mjd_cls_base",
) -> Tuple[Dict[Tuple[int, int], Dict[str, np.ndarray]], Dict[int, torch.Tensor], Dict[int, float]]:
    if zero_time_field not in neg_optical_data:
        raise KeyError(f"Negative optical data is missing '{zero_time_field}' required for time-aware gallery filtering.")
    if "coordinates" not in neg_optical_data:
        raise KeyError("Negative optical data is missing 'coordinates' required for skymap filtering.")

    unique_gw = [int(gw_id) for gw_id in unique_gw_ids]
    neg_times = _as_numpy_1d(neg_optical_data[zero_time_field], dtype=np.float64)
    neg_coords = _as_torch_coords(neg_optical_data["coordinates"])
    neg_opt_xyz = _coords_to_unit_xyz(neg_coords)

    if neg_times.shape[0] != neg_coords.shape[0]:
        raise ValueError("Negative optical times and coordinates must have the same first dimension.")
    if not unique_gw:
        return {}, {}, {}

    with h5py.File(test_data_path, "r") as f:
        if "events/gw_data/event_time_mjd" not in f:
            raise KeyError(f"Missing required field 'events/gw_data/event_time_mjd' in {test_data_path}")
        if "events/gw_data/skymaps" not in f:
            raise KeyError(f"Missing required field 'events/gw_data/skymaps' in {test_data_path}")

        gw_event_times = np.asarray(f["events/gw_data/event_time_mjd"][unique_gw], dtype=np.float64)
        gw_skymaps_np = np.asarray(f["events/gw_data/skymaps"][unique_gw], dtype=np.float32)

    candidate_sequences: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    gw_skymaps: Dict[int, torch.Tensor] = {}
    gw_time_lookup: Dict[int, float] = {}

    for local_idx, gw_id in enumerate(unique_gw):
        gw_skymap = torch.from_numpy(gw_skymaps_np[local_idx]).to(torch.float32)
        gw_skymaps[int(gw_id)] = gw_skymap
        anchor_time = float(gw_event_times[local_idx])
        gw_time_lookup[int(gw_id)] = anchor_time

        dP = np.asarray(gw_skymaps_np[local_idx, 4, :], dtype=np.float64)
        sorted_dP = np.sort(dP, kind="mergesort")
        candidate_credible = np.full(neg_times.shape, np.nan, dtype=np.float64)
        if np.isfinite(anchor_time):
            time_mask = np.isfinite(neg_times) & (np.abs(neg_times - anchor_time) <= float(time_window_days))
            if np.any(time_mask):
                event_pixel_idx = _nearest_pixel_indices_from_xyz(
                    neg_opt_xyz[time_mask],
                    gw_skymap[:3, :],
                )
                candidate_dP = dP[event_pixel_idx]
                candidate_credible[time_mask] = (
                    dP.size - np.searchsorted(sorted_dP, candidate_dP, side="left")
                ) / float(dP.size)

        for trial in range(int(n_trials)):
            seq_seed = int(seed) + 7919 * int(trial) + 104729 * int(gw_id)
            candidate_sequences[(int(trial), int(gw_id))] = build_time_sky_candidate_sequence(
                anchor_time_mjd=anchor_time,
                candidate_zero_time_mjd=neg_times,
                candidate_credible_levels=candidate_credible,
                time_window_days=float(time_window_days),
                credible_level_max=float(credible_level_max),
                seed=seq_seed,
            )

    return candidate_sequences, gw_skymaps, gw_time_lookup


def build_prefixed_gallery_specs(
    *,
    gw_positive_indices: Mapping[int, Any],
    candidate_sequences: Mapping[Tuple[int, int], Mapping[str, Any]],
    gallery_sizes: Sequence[int],
    n_trials: int,
    seed: int,
    include_undersized: bool,
) -> Tuple[Dict[Tuple[int, int, int], Dict[str, Any]], List[int]]:
    unique_gw = sorted(int(gw_id) for gw_id in gw_positive_indices.keys())
    galleries: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    gallery_sizes = [int(size) for size in gallery_sizes]

    for trial in range(int(n_trials)):
        trial_rng = np.random.default_rng(int(seed) + 15485863 * int(trial))
        for gw_id in unique_gw:
            gw_pos_idx = _as_numpy_1d(gw_positive_indices.get(int(gw_id), []), dtype=np.int64)
            if gw_pos_idx.size == 0:
                raise ValueError(f"GW event {gw_id} has no positive optical samples available for gallery construction.")
            positive_index = int(gw_pos_idx[trial_rng.integers(gw_pos_idx.size)])

            seq = candidate_sequences.get((int(trial), int(gw_id)), {})
            neg_idx_full = _as_numpy_1d(seq.get("candidate_indices", []), dtype=np.int64)
            neg_cred_full = _as_numpy_1d(seq.get("credible_levels", []), dtype=np.float32)
            neg_dt_full = _as_numpy_1d(seq.get("abs_dt_days", []), dtype=np.float32)

            for gallery_size in gallery_sizes:
                requested = int(gallery_size)
                target_neg = max(requested - 1, 0)
                available_neg = int(neg_idx_full.shape[0])
                if target_neg > available_neg and not bool(include_undersized):
                    continue
                take_neg = min(target_neg, available_neg)
                actual_gallery_size = 1 + int(take_neg)
                galleries[(requested, int(trial), int(gw_id))] = {
                    "positive_index": positive_index,
                    "negative_indices": neg_idx_full[:take_neg].astype(np.int64, copy=False),
                    "negative_credible_levels": neg_cred_full[:take_neg].astype(np.float32, copy=False),
                    "negative_abs_dt_days": neg_dt_full[:take_neg].astype(np.float32, copy=False),
                    "requested_gallery_size": requested,
                    "actual_gallery_size": actual_gallery_size,
                    "coverage_met": bool(actual_gallery_size >= requested),
                    "is_undersized": bool(actual_gallery_size < requested),
                }

    return galleries, unique_gw


def score_all_galleries_skymap(
    *,
    positive_bank: Mapping[str, Any],
    galleries: Mapping[Tuple[int, int, int], Mapping[str, Any]],
    gw_skymaps: Mapping[int, torch.Tensor],
) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    if "opt_coords" not in positive_bank:
        raise KeyError("positive_bank must contain 'opt_coords' for skymap-only scoring.")
    positive_coords = _as_torch_coords(positive_bank["opt_coords"])

    pos_credible_cache: Dict[Tuple[int, int], float] = {}
    outcomes: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    for key, gallery_spec in galleries.items():
        requested_gallery_size, trial, gw_id = key
        pos_index = int(gallery_spec["positive_index"])
        cache_key = (int(gw_id), pos_index)
        if cache_key not in pos_credible_cache:
            pos_coords = positive_coords[pos_index].unsqueeze(0)
            pos_cred = compute_credible_levels_single_gw(gw_skymaps[int(gw_id)], pos_coords)
            pos_credible_cache[cache_key] = float(pos_cred.squeeze(0).item())

        pos_score = 1.0 - float(pos_credible_cache[cache_key])
        neg_scores = 1.0 - _as_numpy_1d(gallery_spec.get("negative_credible_levels", []), dtype=np.float64)
        scores = np.concatenate([np.asarray([pos_score], dtype=np.float64), neg_scores], axis=0)
        ranked = np.argsort(-scores, kind="mergesort")

        outcomes[(int(requested_gallery_size), int(trial), int(gw_id))] = {
            "rank": int(np.where(ranked == 0)[0][0]),
            "requested_gallery_size": int(gallery_spec["requested_gallery_size"]),
            "actual_gallery_size": int(gallery_spec["actual_gallery_size"]),
            "coverage_met": bool(gallery_spec["coverage_met"]),
            "is_undersized": bool(gallery_spec["is_undersized"]),
        }

    return outcomes


def aggregate_gallery_outcomes(
    *,
    outcomes: Mapping[Tuple[int, int, int], Mapping[str, Any]],
    gallery_sizes: Sequence[int],
    n_trials: int,
    unique_gw: Sequence[int],
    gw_source_map: Mapping[int, str] | None = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, Any]]]:
    metrics: Dict[str, float] = {}
    coverage_stats: Dict[str, Dict[str, Any]] = {}
    by_source: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for gallery_size in [int(size) for size in gallery_sizes]:
        recalls = {1: [], 5: [], 10: []}
        mrrs: List[float] = []
        coverage_flags: List[float] = []
        actual_sizes: List[int] = []

        for trial in range(int(n_trials)):
            for gw_id in [int(gw) for gw in unique_gw]:
                key = (int(gallery_size), int(trial), int(gw_id))
                if key not in outcomes:
                    continue
                outcome = outcomes[key]
                rank = int(outcome["rank"])
                actual_gallery_size = int(outcome["actual_gallery_size"])
                coverage_met = bool(outcome.get("coverage_met", actual_gallery_size >= int(gallery_size)))

                for k in recalls:
                    recalls[k].append(1.0 if rank < k else 0.0)
                mrrs.append(1.0 / float(rank + 1))
                coverage_flags.append(1.0 if coverage_met else 0.0)
                actual_sizes.append(actual_gallery_size)

                if gw_source_map is not None:
                    source_label = gw_source_map.get(int(gw_id), "unknown")
                    if source_label not in by_source:
                        by_source[source_label] = {}
                    if int(gallery_size) not in by_source[source_label]:
                        by_source[source_label][int(gallery_size)] = {
                            "recalls": {1: [], 5: [], 10: []},
                            "mrrs": [],
                        }
                    source_bucket = by_source[source_label][int(gallery_size)]
                    for k in source_bucket["recalls"]:
                        source_bucket["recalls"][k].append(1.0 if rank < k else 0.0)
                    source_bucket["mrrs"].append(1.0 / float(rank + 1))

        for k, values in recalls.items():
            metrics[f"gallery_{gallery_size}_recall_at_{k}"] = float(np.mean(values)) if values else 0.0
        metrics[f"gallery_{gallery_size}_mrr"] = float(np.mean(mrrs)) if mrrs else 0.0

        coverage_key = f"gallery_{gallery_size}"
        if actual_sizes:
            coverage_stats[coverage_key] = {
                "coverage": float(np.mean(coverage_flags)),
                "n_queries_total": int(len(actual_sizes)),
                "n_queries_covered": int(sum(1 for flag in coverage_flags if flag > 0.0)),
                "effective_gallery_size_mean": float(np.mean(actual_sizes)),
                "effective_gallery_size_min": int(np.min(actual_sizes)),
                "effective_gallery_size_max": int(np.max(actual_sizes)),
            }
        else:
            coverage_stats[coverage_key] = {
                "coverage": 0.0,
                "n_queries_total": 0,
                "n_queries_covered": 0,
                "effective_gallery_size_mean": 0.0,
                "effective_gallery_size_min": 0,
                "effective_gallery_size_max": 0,
            }

    source_metrics: Dict[str, Dict[str, float]] = {}
    for source_label, size_acc in by_source.items():
        source_metrics[source_label] = {}
        for gallery_size, acc in size_acc.items():
            for k, vals in acc["recalls"].items():
                source_metrics[source_label][f"gallery_{gallery_size}_recall_at_{k}"] = float(np.mean(vals)) if vals else 0.0
            source_metrics[source_label][f"gallery_{gallery_size}_mrr"] = float(np.mean(acc["mrrs"])) if acc["mrrs"] else 0.0

    return metrics, source_metrics, coverage_stats


def build_curve_rows(
    *,
    method_name: str,
    gallery_sizes: Sequence[int],
    retrieval_metrics: Mapping[str, Any],
    coverage_stats: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for gallery_size in [int(size) for size in gallery_sizes]:
        coverage_key = f"gallery_{gallery_size}"
        coverage_info = coverage_stats.get(coverage_key, {})
        for metric_label, metric_key in zip(TABLE_METRIC_LABELS, TABLE_METRIC_KEYS):
            rows.append(
                {
                    "method": str(method_name),
                    "gallery_size_target": int(gallery_size),
                    "gallery_size_actual": float(coverage_info.get("effective_gallery_size_mean", float(gallery_size))),
                    "coverage": float(coverage_info.get("coverage", 0.0)),
                    "metric_name": metric_label,
                    "metric_value": float(retrieval_metrics.get(f"gallery_{gallery_size}_{metric_key}", 0.0)),
                }
            )
    return rows


def plot_retrieval_curves(curve_rows: Sequence[Mapping[str, Any]], output_dir: Path | str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = list(curve_rows)
    if not rows:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = sorted({str(row["method"]) for row in rows})
    metrics = ["R@1", "R@5", "R@10"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.8), sharex=False, sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics):
        metric_rows = [row for row in rows if str(row.get("metric_name")) == metric_name]
        for method in methods:
            method_rows = sorted(
                [row for row in metric_rows if str(row.get("method")) == method],
                key=lambda row: int(row["gallery_size_target"]),
            )
            if not method_rows:
                continue
            ax.plot(
                [int(row["gallery_size_target"]) for row in method_rows],
                [float(row["metric_value"]) for row in method_rows],
                marker="o",
                linewidth=2,
                label=_plot_method_label(method),
            )
        ax.set_xscale("log")
        ax.set_xlabel("Gallery Size")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs Gallery Size")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, min(4, len(labels))), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_dir / "retrieval_curves.png", dpi=PLOT_DPI, bbox_inches="tight")
    _remove_stale_pdf(output_dir / "retrieval_curves.pdf")
    plt.close(fig)


def plot_retrieval_coverage(curve_rows: Sequence[Mapping[str, Any]], output_dir: Path | str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = list(curve_rows)
    if not rows:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_method: Dict[str, Dict[int, Dict[str, float]]] = {}
    for row in rows:
        method = str(row["method"])
        gallery_size = int(row["gallery_size_target"])
        by_method.setdefault(method, {})
        by_method[method][gallery_size] = {
            "coverage": float(row.get("coverage", 0.0)),
            "gallery_size_actual": float(row.get("gallery_size_actual", gallery_size)),
        }

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in sorted(by_method):
        ordered = sorted(by_method[method].items(), key=lambda item: int(item[0]))
        ax.plot(
            [gallery_size for gallery_size, _ in ordered],
            [payload["coverage"] for _, payload in ordered],
            marker="o",
            linewidth=2,
            label=_plot_method_label(method),
        )

    ax.set_xscale("log")
    ax.set_xlabel("Gallery Size")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Coverage Under Hard Candidate Constraints")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "retrieval_coverage.png", dpi=PLOT_DPI, bbox_inches="tight")
    _remove_stale_pdf(output_dir / "retrieval_coverage.pdf")
    plt.close(fig)
