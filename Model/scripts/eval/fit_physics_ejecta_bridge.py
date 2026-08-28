#!/usr/bin/env python3
"""Fit the frozen empirical Physics Ejecta Bridge artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval.physics_ejecta_bridge import (
    CADENCE_NAMES,
    GW_FEATURE_NAMES,
    LIGHTCURVE_FEATURE_NAMES,
    PSI_NAMES,
    build_psi,
    decode_strings,
    derive_gw_feature_matrix,
    deterministic_event_split,
    extract_lightcurve_features,
    masked_robust_location_scale,
    robust_location_scale,
    stable_hash_int,
    weighted_gaussian,
)


def _resolve(config_dir: Path, value: Any) -> str | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value)).expanduser()
    return str(
        (config_dir / path).resolve() if not path.is_absolute() else path.resolve()
    )


def normalise_config(raw: Mapping[str, Any], path: Path) -> dict[str, Any]:
    cfg = dict(raw)
    directory = path.parent
    cfg["train_data_path"] = _resolve(directory, cfg.get("train_data_path"))
    cfg["test_data_path"] = _resolve(directory, cfg.get("test_data_path"))
    cfg["artifact_dir"] = _resolve(directory, cfg.get("artifact_dir"))
    cfg["seed"] = int(cfg.get("seed", 42))
    cfg["fit_fraction"] = float(cfg.get("fit_fraction", 0.8))
    cfg["max_curves_per_event"] = int(cfg.get("max_curves_per_event", 4))
    cfg["calibration_device"] = str(cfg.get("calibration_device", "cpu"))
    cfg["calibration_batch_size"] = int(cfg.get("calibration_batch_size", 128))
    cfg["hdf5_read_block_rows"] = int(cfg.get("hdf5_read_block_rows", 8192))
    cfg["num_threads"] = int(
        cfg.get(
            "num_threads", os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)
        )
    )
    cfg["comparison_window"] = [
        float(value) for value in cfg.get("comparison_window", [-0.1, 0.2])
    ]
    cfg["k_candidates"] = [
        int(value) for value in cfg.get("k_candidates", [16, 32, 64, 128])
    ]
    cfg["alpha_candidates"] = [
        float(value) for value in cfg.get("alpha_candidates", [0.5, 1.0, 2.0])
    ]
    cfg["covariance_floor_candidates"] = [
        float(value)
        for value in cfg.get("covariance_floor_candidates", [0.01, 0.03, 0.05, 0.1])
    ]
    if not cfg["train_data_path"] or not cfg["artifact_dir"]:
        raise ValueError("train_data_path and artifact_dir are required")
    if not Path(cfg["train_data_path"]).is_file():
        raise FileNotFoundError(cfg["train_data_path"])
    if cfg["test_data_path"] and not Path(cfg["test_data_path"]).is_file():
        raise FileNotFoundError(cfg["test_data_path"])
    if (
        len(cfg["comparison_window"]) != 2
        or cfg["comparison_window"][0] >= cfg["comparison_window"][1]
    ):
        raise ValueError("comparison_window must contain two increasing values")
    if not 0.0 < cfg["fit_fraction"] < 1.0:
        raise ValueError("fit_fraction must be in (0, 1)")
    if cfg["max_curves_per_event"] < 1:
        raise ValueError("max_curves_per_event must be positive")
    if (
        min(
            cfg["calibration_batch_size"],
            cfg["hdf5_read_block_rows"],
            cfg["num_threads"],
        )
        < 1
    ):
        raise ValueError("Performance batch/block/thread settings must be positive")
    if cfg["calibration_device"] not in {"cpu", "cuda"}:
        raise ValueError("calibration_device must be cpu or cuda")
    if min(cfg["k_candidates"]) < 2 or any(
        value <= 0 for value in cfg["alpha_candidates"]
    ):
        raise ValueError("Invalid k/alpha candidates")
    if any(value <= 0 for value in cfg["covariance_floor_candidates"]):
        raise ValueError("Covariance floors must be positive")
    return cfg


def _dataset_identity(path: str, handle: h5py.File) -> dict[str, Any]:
    target = Path(path)
    simulations = np.asarray(handle["events/gw_data/simulation_id"][:], dtype=np.int64)
    digest = hashlib.sha256(simulations.tobytes()).hexdigest()
    return {
        "path": str(target),
        "size": int(target.stat().st_size),
        "simulation_id_sha256": digest,
        "n_gw": len(simulations),
        "n_optical": int(handle["events/optical_data/values"].shape[0]),
    }


@contextmanager
def _timed_phase(label: str):
    started = time.perf_counter()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {label}...", flush=True)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {label}: {elapsed:.1f}s",
            flush=True,
        )


def _stable_hash_array(seed: int, indices: np.ndarray) -> np.ndarray:
    """Vectorized deterministic SplitMix64 hash for optical row selection."""
    values = np.asarray(indices, dtype=np.uint64) + np.uint64(int(seed))
    values = values + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def _select_curves_fast(
    parent_gw_idx: np.ndarray,
    allowed_events: np.ndarray,
    *,
    seed: int,
    max_curves_per_event: int,
) -> np.ndarray:
    """Select stable hash minima without scanning nine million rows in Python."""
    parent = np.asarray(parent_gw_idx, dtype=np.int64)
    events = np.sort(np.asarray(allowed_events, dtype=np.int64))
    monotonic = bool(parent.size < 2 or np.all(parent[:-1] <= parent[1:]))
    order = None if monotonic else np.argsort(parent, kind="stable")
    ordered_parent = parent if monotonic else parent[order]
    selected: list[np.ndarray] = []
    for event in events:
        left = int(np.searchsorted(ordered_parent, event, side="left"))
        right = int(np.searchsorted(ordered_parent, event, side="right"))
        if left == right:
            continue
        rows = (
            np.arange(left, right, dtype=np.int64)
            if order is None
            else order[left:right]
        )
        hashes = _stable_hash_array(seed, rows)
        keep = min(int(max_curves_per_event), len(rows))
        local = (
            np.argpartition(hashes, keep - 1)[:keep]
            if keep < len(rows)
            else np.arange(keep)
        )
        chosen = rows[local]
        chosen_hashes = hashes[local]
        selected.append(chosen[np.lexsort((chosen, chosen_hashes))])
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def _read_selected_rows(
    dataset: h5py.Dataset,
    selected: np.ndarray,
    *,
    block_rows: int,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Read selected rows through contiguous blocks instead of a point selection."""
    rows = np.asarray(selected, dtype=np.int64)
    output = np.empty((len(rows), *dataset.shape[1:]), dtype=dtype)
    if not len(rows):
        return output
    block_ids = rows // int(block_rows)
    unique_blocks, starts = np.unique(block_ids, return_index=True)
    stops = np.r_[starts[1:], len(rows)]
    for block_id, left, right in zip(unique_blocks, starts, stops):
        begin = int(block_id) * int(block_rows)
        end = min(begin + int(block_rows), int(dataset.shape[0]))
        block = np.asarray(dataset[begin:end], dtype=dtype)
        output[left:right] = block[rows[left:right] - begin]
    return output


def _selected_curves(
    handle: h5py.File,
    event_indices: np.ndarray,
    *,
    parent_all: np.ndarray,
    seed: int,
    max_curves: int,
    window: tuple[float, float],
    block_rows: int,
) -> dict[str, np.ndarray]:
    optical = handle["events/optical_data"]
    with _timed_phase("Selecting at most four curves per positive train event"):
        selected = _select_curves_fast(
            parent_all,
            event_indices,
            seed=seed,
            max_curves_per_event=max_curves,
        )
    print(f"  Selected {len(selected):,} optical curves", flush=True)
    loaded: dict[str, np.ndarray] = {}
    for name in ("times", "values", "masks", "errors"):
        with _timed_phase(f"Reading optical {name} in contiguous HDF5 blocks"):
            loaded[name] = _read_selected_rows(
                optical[name], selected, block_rows=block_rows
            )
    times = loaded["times"]
    masks = loaded["masks"]
    in_window = (times >= float(window[0])) & (times <= float(window[1]))
    masks = masks * in_window[..., None]
    with _timed_phase("Extracting Physics Bridge light-curve features"):
        features, feature_mask, cadence = extract_lightcurve_features(
            times,
            loaded["values"],
            masks,
            loaded["errors"],
            psfflux_zp=float(handle.attrs["psfflux_zp"]),
            lupt_b_njy=np.asarray(handle.attrs["lupt_b_njy"], dtype=np.float64),
        )
    return {
        "indices": selected,
        "parent": parent_all[selected],
        "features": features,
        "feature_mask": feature_mask,
        "cadence": cadence,
    }


def _curve_subset(
    curves: Mapping[str, np.ndarray], rows: np.ndarray
) -> dict[str, np.ndarray]:
    indices = np.asarray(rows, dtype=np.int64)
    return {name: np.asarray(values)[indices] for name, values in curves.items()}


def _curve_rows_for_events(
    curves: Mapping[str, np.ndarray],
    event_indices: np.ndarray,
    *,
    one_per_event: bool,
    seed: int,
) -> np.ndarray:
    allowed = set(np.asarray(event_indices, dtype=np.int64).tolist())
    candidates = [
        index
        for index, parent in enumerate(np.asarray(curves["parent"], dtype=np.int64))
        if int(parent) in allowed
    ]
    if not one_per_event:
        return np.asarray(candidates, dtype=np.int64)
    selected: dict[int, tuple[int, int]] = {}
    optical = np.asarray(curves["indices"], dtype=np.int64)
    parent = np.asarray(curves["parent"], dtype=np.int64)
    for row in candidates:
        key = (stable_hash_int(seed, int(optical[row])), int(row))
        event = int(parent[row])
        if event not in selected or key < selected[event]:
            selected[event] = key
    return np.asarray(
        [value[1] for _, value in sorted(selected.items())], dtype=np.int64
    )


def _normal_nll(truth: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> float:
    delta = np.asarray(truth, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    chol = np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))
    solved = np.linalg.solve(chol, delta)
    return 0.5 * (
        float(solved @ solved)
        + 2.0 * float(np.log(np.diag(chol)).sum())
        + len(delta) * np.log(2.0 * np.pi)
    )


def _metrics_from_posteriors(
    truths: np.ndarray,
    samples: list[np.ndarray],
    distances: list[np.ndarray],
    *,
    k: int,
    alpha: float,
    covariance_floor: float,
    dimensions: tuple[int, ...],
) -> dict[str, float]:
    nll: list[float] = []
    covered50: list[np.ndarray] = []
    covered90: list[np.ndarray] = []
    for truth, sample_values, sample_distances in zip(truths, samples, distances):
        bandwidth = float(alpha) * max(float(sample_distances[int(k) - 1]), 1e-8)
        weights = np.exp(-0.5 * (sample_distances[: int(k)] / bandwidth) ** 2)
        weights /= weights.sum()
        mean, covariance = weighted_gaussian(
            sample_values[: int(k)],
            weights,
            covariance_floor=float(covariance_floor),
        )
        idx = np.asarray(dimensions, dtype=np.int64)
        sub_truth = truth[idx]
        sub_mean = mean[idx]
        sub_cov = covariance[np.ix_(idx, idx)]
        nll.append(_normal_nll(sub_truth, sub_mean, sub_cov))
        sigma = np.sqrt(np.diag(sub_cov))
        covered50.append(np.abs(sub_truth - sub_mean) <= 0.67448975 * sigma)
        covered90.append(np.abs(sub_truth - sub_mean) <= 1.64485363 * sigma)
    return {
        "nll": float(np.mean(nll)),
        "coverage50": float(np.mean(np.asarray(covered50))),
        "coverage90": float(np.mean(np.asarray(covered90))),
    }


def _prepare_neighbor_samples(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    reference_parent: np.ndarray,
    reference_psi: np.ndarray,
    *,
    max_k: int,
    query_mask: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
    query_cadence: np.ndarray | None = None,
    reference_cadence: np.ndarray | None = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Batched exact distance search with unique parent-event posteriors."""
    torch_device = torch.device(device)
    references = torch.as_tensor(
        np.asarray(reference_features, dtype=np.float64), device=torch_device
    )
    parents = np.asarray(reference_parent, dtype=np.int64)
    sample_rows: list[np.ndarray] = []
    distance_rows: list[np.ndarray] = []
    masked = query_mask is not None
    if masked:
        reference_masks_np = np.asarray(reference_mask, dtype=bool)
        reference_masks_t = torch.as_tensor(
            reference_masks_np.astype(np.float64), device=torch_device
        )
        reference_masked = references * reference_masks_t
        reference_squared_masked = references.square() * reference_masks_t
    if query_cadence is not None:
        reference_cadence_t = torch.as_tensor(
            np.asarray(reference_cadence, dtype=np.float64), device=torch_device
        )
    queries = np.asarray(query_features, dtype=np.float64)
    required = 8 * int(max_k)
    for start in range(0, len(queries), int(batch_size)):
        stop = min(start + int(batch_size), len(queries))
        query_t = torch.as_tensor(queries[start:stop], device=torch_device)
        if not masked:
            distances_t = torch.cdist(query_t, references)
        else:
            query_masks_t = torch.as_tensor(
                np.asarray(query_mask[start:stop], dtype=np.float64),
                device=torch_device,
            )
            query_masked = query_t * query_masks_t
            common = query_masks_t @ reference_masks_t.T
            squared = (
                (query_t.square() * query_masks_t) @ reference_masks_t.T
                + query_masks_t @ reference_squared_masked.T
                - 2.0 * (query_masked @ reference_masked.T)
            )
            distances_t = torch.sqrt(
                torch.clamp(squared, min=0.0) / torch.clamp(common, min=1.0)
            )
            distances_t = torch.where(
                common > 0,
                distances_t,
                torch.full_like(distances_t, float("inf")),
            )
        if query_cadence is not None:
            cadence_t = torch.as_tensor(
                np.asarray(query_cadence[start:stop], dtype=np.float64),
                device=torch_device,
            )
            difference = torch.abs(
                cadence_t[:, None, :] - reference_cadence_t[None, :, :]
            )
            strict = (difference.amax(dim=2) <= 1.0) & (difference.sum(dim=2) <= 3.0)
            relaxed = (difference.amax(dim=2) <= 1.5) & (difference.sum(dim=2) <= 4.5)
            strict_rows = strict.sum(dim=1) >= required
            relaxed_rows = (~strict_rows) & (relaxed.sum(dim=1) >= required)
            eligible = torch.ones_like(strict)
            eligible[strict_rows] = strict[strict_rows]
            eligible[relaxed_rows] = relaxed[relaxed_rows]
            distances_t = torch.where(
                eligible,
                distances_t,
                torch.full_like(distances_t, float("inf")),
            )
        search_k = int(max_k) if not masked else max(required * 2, 2048)
        search_k = min(search_k, distances_t.shape[1])
        top_distances, top_indices = torch.topk(
            distances_t, k=search_k, largest=False, sorted=True
        )
        top_distances_np = np.asarray(top_distances.cpu(), dtype=np.float64)
        top_indices_np = np.asarray(top_indices.cpu(), dtype=np.int64)
        for local in range(stop - start):
            chosen: list[int] = []
            chosen_distances: list[float] = []
            seen: set[int] = set()
            for reference_index, distance in zip(
                top_indices_np[local], top_distances_np[local]
            ):
                if not np.isfinite(distance):
                    break
                parent = int(parents[reference_index])
                if parent in seen:
                    continue
                chosen.append(int(reference_index))
                chosen_distances.append(float(distance))
                seen.add(parent)
                if len(chosen) == int(max_k):
                    break
            if len(chosen) < int(max_k):
                raise ValueError(
                    f"Only {len(chosen)} unique calibration neighbors; need {max_k}"
                )
            indices = np.asarray(chosen, dtype=np.int64)
            sample_rows.append(reference_psi[indices])
            distance_rows.append(np.asarray(chosen_distances, dtype=np.float64))
    return sample_rows, distance_rows


def _grid_metrics(
    truths: np.ndarray,
    samples: list[np.ndarray],
    distances: list[np.ndarray],
    cfg: Mapping[str, Any],
    *,
    dimensions: tuple[int, ...],
) -> list[dict[str, Any]]:
    candidates = [
        (float(floor), int(k), float(alpha))
        for floor in cfg["covariance_floor_candidates"]
        for k in cfg["k_candidates"]
        for alpha in cfg["alpha_candidates"]
    ]

    def evaluate(candidate: tuple[float, int, float]) -> dict[str, Any]:
        floor, k, alpha = candidate
        return {
            "k": k,
            "alpha": alpha,
            "covariance_floor": floor,
            **_metrics_from_posteriors(
                truths,
                samples,
                distances,
                k=k,
                alpha=alpha,
                covariance_floor=floor,
                dimensions=dimensions,
            ),
        }

    workers = min(int(cfg["num_threads"]), len(candidates))
    if workers == 1:
        return [evaluate(candidate) for candidate in candidates]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(evaluate, candidates))


def _choose_common_hyperparameters(
    gw_rows: list[dict[str, Any]],
    lc_rows: list[dict[str, Any]],
    floors: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    candidates: list[tuple[bool, float, dict[str, Any], dict[str, Any]]] = []
    for floor in floors:
        gw_floor = [row for row in gw_rows if row["covariance_floor"] == floor]
        lc_floor = [row for row in lc_rows if row["covariance_floor"] == floor]
        gw_ok = [row for row in gw_floor if 0.85 <= row["coverage90"] <= 0.95]
        lc_ok = [row for row in lc_floor if 0.85 <= row["coverage90"] <= 0.95]
        gw = min(
            gw_ok or gw_floor, key=lambda row: (row["nll"], row["k"], row["alpha"])
        )
        lc = min(
            lc_ok or lc_floor, key=lambda row: (row["nll"], row["k"], row["alpha"])
        )
        acceptable = bool(gw_ok and lc_ok)
        candidates.append((acceptable, float(gw["nll"] + lc["nll"]), gw, lc))
    acceptable = [item for item in candidates if item[0]]
    chosen = min(
        acceptable or candidates,
        key=lambda item: (item[1], item[2]["covariance_floor"]),
    )
    return dict(chosen[2]), dict(chosen[3]), not bool(acceptable)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def fit_bridge(cfg: Mapping[str, Any], config_path: Path) -> None:
    torch.set_num_threads(int(cfg["num_threads"]))
    print(
        "Bridge fit runtime: device={} threads={} batch={} HDF5_block_rows={}".format(
            cfg["calibration_device"],
            cfg["num_threads"],
            cfg["calibration_batch_size"],
            cfg["hdf5_read_block_rows"],
        ),
        flush=True,
    )
    output = Path(cfg["artifact_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Bridge artifact directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    calibration_rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    source_manifest: dict[str, Any] = {}
    with h5py.File(cfg["train_data_path"], "r") as handle:
        gw = handle["events/gw_data"]
        scalars = np.asarray(gw["scalars"][:], dtype=np.float64)
        sources = decode_strings(gw["source_type"][:])
        simulations = np.asarray(gw["simulation_id"][:], dtype=np.int64)
        has_kn = np.asarray(gw["has_kn"][:], dtype=np.int8) == 1
        fit_mask = deterministic_event_split(
            sources,
            simulations,
            seed=int(cfg["seed"]),
            fit_fraction=float(cfg["fit_fraction"]),
        )
        positive_indices = np.flatnonzero(has_kn)
        all_psi = np.full((len(scalars), len(PSI_NAMES)), np.nan, dtype=np.float64)
        all_psi[positive_indices] = build_psi(
            np.asarray(gw["mej_dynamic"][positive_indices], dtype=np.float64),
            np.asarray(gw["mej_wind"][positive_indices], dtype=np.float64),
            scalars[positive_indices],
        )
        identity = _dataset_identity(cfg["train_data_path"], handle)
        split_sha256 = hashlib.sha256(
            np.flatnonzero(has_kn & fit_mask).astype(np.int64).tobytes()
        ).hexdigest()
        with _timed_phase("Reading train optical parent index once"):
            parent_all = np.asarray(
                handle["events/optical_data/parent_gw_idx"][:], dtype=np.int64
            )
        all_curves = _selected_curves(
            handle,
            positive_indices,
            parent_all=parent_all,
            seed=int(cfg["seed"]),
            max_curves=int(cfg["max_curves_per_event"]),
            window=tuple(cfg["comparison_window"]),
            block_rows=int(cfg["hdf5_read_block_rows"]),
        )
        for source in ("bns", "nsbh"):
            print(f"Preparing {source.upper()} Bridge references...", flush=True)
            source_all = np.flatnonzero(has_kn & (sources == source))
            source_fit = np.flatnonzero(has_kn & (sources == source) & fit_mask)
            source_cal = np.flatnonzero(has_kn & (sources == source) & ~fit_mask)
            psi_center, psi_scale = robust_location_scale(all_psi[source_fit])
            fit_psi = (all_psi[source_fit] - psi_center) / psi_scale
            cal_psi = (all_psi[source_cal] - psi_center) / psi_scale
            fit_gw_raw = derive_gw_feature_matrix(scalars[source_fit], source)
            cal_gw_raw = derive_gw_feature_matrix(scalars[source_cal], source)
            gw_center, gw_scale = robust_location_scale(fit_gw_raw)
            fit_gw = (fit_gw_raw - gw_center) / gw_scale
            cal_gw = (cal_gw_raw - gw_center) / gw_scale
            with _timed_phase(
                f"{source.upper()} batched GW calibration-neighbor search"
            ):
                gw_samples, gw_distances = _prepare_neighbor_samples(
                    cal_gw,
                    fit_gw,
                    source_fit,
                    fit_psi,
                    max_k=max(cfg["k_candidates"]),
                    device=str(cfg["calibration_device"]),
                    batch_size=int(cfg["calibration_batch_size"]),
                )
            full_curves = _curve_subset(
                all_curves,
                _curve_rows_for_events(
                    all_curves,
                    source_all,
                    one_per_event=False,
                    seed=int(cfg["seed"]),
                ),
            )
            fit_curves = _curve_subset(
                full_curves,
                _curve_rows_for_events(
                    full_curves,
                    source_fit,
                    one_per_event=False,
                    seed=int(cfg["seed"]),
                ),
            )
            cal_curves = _curve_subset(
                full_curves,
                _curve_rows_for_events(
                    full_curves,
                    source_cal,
                    one_per_event=True,
                    seed=int(cfg["seed"]) + 1,
                ),
            )
            lc_center, lc_scale = masked_robust_location_scale(
                fit_curves["features"], fit_curves["feature_mask"]
            )
            fit_lc = np.clip((fit_curves["features"] - lc_center) / lc_scale, -8.0, 8.0)
            cal_lc = np.clip((cal_curves["features"] - lc_center) / lc_scale, -8.0, 8.0)
            fit_curve_psi = (all_psi[fit_curves["parent"]] - psi_center) / psi_scale
            cal_curve_psi = (all_psi[cal_curves["parent"]] - psi_center) / psi_scale
            calibration_cadence_center, calibration_cadence_scale = (
                robust_location_scale(fit_curves["cadence"])
            )
            with _timed_phase(
                f"{source.upper()} batched light-curve calibration-neighbor search"
            ):
                lc_samples, lc_distances = _prepare_neighbor_samples(
                    cal_lc,
                    fit_lc,
                    fit_curves["parent"],
                    fit_curve_psi,
                    max_k=max(cfg["k_candidates"]),
                    query_mask=cal_curves["feature_mask"],
                    reference_mask=fit_curves["feature_mask"],
                    query_cadence=(cal_curves["cadence"] - calibration_cadence_center)
                    / calibration_cadence_scale,
                    reference_cadence=(
                        fit_curves["cadence"] - calibration_cadence_center
                    )
                    / calibration_cadence_scale,
                    device=str(cfg["calibration_device"]),
                    batch_size=int(cfg["calibration_batch_size"]),
                )
            with _timed_phase(f"{source.upper()} parallel hyperparameter grid"):
                gw_rows = _grid_metrics(
                    cal_psi,
                    gw_samples,
                    gw_distances,
                    cfg,
                    dimensions=(0, 1),
                )
                lc_rows = _grid_metrics(
                    cal_curve_psi,
                    lc_samples,
                    lc_distances,
                    cfg,
                    dimensions=(0, 1, 2, 3),
                )
            gw_hp, lc_hp, warning = _choose_common_hyperparameters(
                gw_rows, lc_rows, cfg["covariance_floor_candidates"]
            )
            print(
                "  Locked {} GW k={} alpha={} and LC k={} alpha={} "
                "with floor={}".format(
                    source.upper(),
                    gw_hp["k"],
                    gw_hp["alpha"],
                    lc_hp["k"],
                    lc_hp["alpha"],
                    gw_hp["covariance_floor"],
                ),
                flush=True,
            )
            calibration_rows.extend(
                {"source": source, "side": "gw", **row} for row in gw_rows
            )
            calibration_rows.extend(
                {"source": source, "side": "lc", **row} for row in lc_rows
            )
            full_psi = all_psi[source_all]
            full_psi_center, full_psi_scale = robust_location_scale(full_psi)
            full_gw_raw = derive_gw_feature_matrix(scalars[source_all], source)
            full_gw_center, full_gw_scale = robust_location_scale(full_gw_raw)
            full_lc_center, full_lc_scale = masked_robust_location_scale(
                full_curves["features"], full_curves["feature_mask"]
            )
            cadence_center, cadence_scale = robust_location_scale(
                full_curves["cadence"]
            )
            prefix = f"{source}__"
            arrays[prefix + "psi_center"] = full_psi_center.astype(np.float32)
            arrays[prefix + "psi_scale"] = full_psi_scale.astype(np.float32)
            arrays[prefix + "gw_center"] = full_gw_center.astype(np.float32)
            arrays[prefix + "gw_scale"] = full_gw_scale.astype(np.float32)
            arrays[prefix + "gw_features"] = (
                (full_gw_raw - full_gw_center) / full_gw_scale
            ).astype(np.float32)
            arrays[prefix + "gw_parent_ids"] = source_all.astype(np.int64)
            arrays[prefix + "gw_psi"] = (
                (full_psi - full_psi_center) / full_psi_scale
            ).astype(np.float32)
            arrays[prefix + "lc_center"] = full_lc_center.astype(np.float32)
            arrays[prefix + "lc_scale"] = full_lc_scale.astype(np.float32)
            arrays[prefix + "cadence_center"] = cadence_center.astype(np.float32)
            arrays[prefix + "cadence_scale"] = cadence_scale.astype(np.float32)
            arrays[prefix + "lc_features"] = np.clip(
                (full_curves["features"] - full_lc_center) / full_lc_scale,
                -8.0,
                8.0,
            ).astype(np.float32)
            arrays[prefix + "lc_feature_mask"] = full_curves["feature_mask"].astype(
                np.uint8
            )
            arrays[prefix + "lc_cadence"] = (
                (full_curves["cadence"] - cadence_center) / cadence_scale
            ).astype(np.float32)
            arrays[prefix + "lc_parent_ids"] = full_curves["parent"].astype(np.int64)
            arrays[prefix + "lc_optical_indices"] = full_curves["indices"].astype(
                np.int64
            )
            arrays[prefix + "lc_psi"] = (
                (all_psi[full_curves["parent"]] - full_psi_center) / full_psi_scale
            ).astype(np.float32)
            source_manifest[source] = {
                "n_events": len(source_all),
                "n_fit_events": int(
                    np.count_nonzero(has_kn & (sources == source) & fit_mask)
                ),
                "n_calibration_events_used": len(source_cal),
                "n_reference_curves": len(full_curves["indices"]),
                "gw_feature_names": list(GW_FEATURE_NAMES[source]),
                "gw_hyperparameters": gw_hp,
                "lc_hyperparameters": lc_hp,
                "calibration_warning": bool(warning),
            }
    overlap_diagnostics: dict[str, Any] = {}
    if cfg["test_data_path"]:
        with (
            h5py.File(cfg["train_data_path"], "r") as train,
            h5py.File(cfg["test_data_path"], "r") as test,
        ):
            train_uid = set(
                decode_strings(train["events/gw_data/event_uid"][:]).tolist()
            )
            test_uid = set(decode_strings(test["events/gw_data/event_uid"][:]).tolist())
            uid_overlap = train_uid.intersection(test_uid)
            if uid_overlap:
                raise ValueError(
                    f"Train/test canonical event_uid overlap detected: {sorted(uid_overlap)[:5]}"
                )
            test_sources = decode_strings(test["events/gw_data/source_type"][:])
            test_simulations = np.asarray(
                test["events/gw_data/simulation_id"][:], dtype=np.int64
            )
            simulation_overlap = {
                (str(source), int(simulation))
                for source, simulation in zip(sources.tolist(), simulations.tolist())
            }.intersection(
                (str(source), int(simulation))
                for source, simulation in zip(
                    test_sources.tolist(), test_simulations.tolist()
                )
            )
            overlap_diagnostics = {
                "canonical_event_uid_overlap_count": 0,
                "source_simulation_id_overlap_count": len(simulation_overlap),
                "source_simulation_id_scope": "split_local_not_identity",
            }
    npz_temp = output / "bridge_reference.npz.tmp"
    with npz_temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    reference_path = output / "bridge_reference.npz"
    os.replace(npz_temp, reference_path)
    reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    import pandas as pd

    pd.DataFrame(calibration_rows).to_csv(
        output / "bridge_calibration_metrics.csv", index=False
    )
    manifest = {
        "status": "complete",
        "artifact_version": "physics_ejecta_bridge_v1",
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "fit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bridge_module_sha256": hashlib.sha256(
            (SCRIPT_DIR / "physics_ejecta_bridge.py").read_bytes()
        ).hexdigest(),
        "split_sha256": split_sha256,
        "reference_npz_sha256": reference_sha256,
        "train_dataset": identity,
        "test_data_path_for_overlap_check_only": cfg["test_data_path"],
        "test_truth_fields_used_for_scoring": [],
        "train_test_identity_diagnostics": overlap_diagnostics,
        "seed": int(cfg["seed"]),
        "fit_fraction": float(cfg["fit_fraction"]),
        "max_curves_per_event": int(cfg["max_curves_per_event"]),
        "fit_runtime": {
            "calibration_device": str(cfg["calibration_device"]),
            "calibration_batch_size": int(cfg["calibration_batch_size"]),
            "hdf5_read_block_rows": int(cfg["hdf5_read_block_rows"]),
            "num_threads": int(cfg["num_threads"]),
            "combined_source_optical_read": True,
            "curve_selection_hash": "splitmix64_seed_plus_optical_index_v1",
            "neighbor_distance_precision": "float64",
        },
        "comparison_window": list(cfg["comparison_window"]),
        "psi_names": list(PSI_NAMES),
        "lightcurve_feature_names": list(LIGHTCURVE_FEATURE_NAMES),
        "cadence_names": list(CADENCE_NAMES),
        "sources": source_manifest,
    }
    _write_json(output / "bridge_fit_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "artifact_dir": str(output),
                "sources": source_manifest,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = normalise_config(json.load(handle), config_path)
    if args.validate_only:
        print(json.dumps({"status": "valid", "config": str(config_path)}, indent=2))
        return
    fit_bridge(cfg, config_path)


if __name__ == "__main__":
    main()
