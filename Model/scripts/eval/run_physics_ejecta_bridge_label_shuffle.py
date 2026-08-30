#!/usr/bin/env python3
"""Event-level label-shuffle controls for the Physics Ejecta Bridge."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from scripts.eval import eval_gw_kn_directional_bridge as directional
from scripts.eval import eval_gw_kn_pairing_sensitivity as v1
from scripts.eval import eval_retrieval_comparison as base
from scripts.eval.eval_run_io import (
    AtomicGzipCsvWriter,
    mark_run_success,
    prepare_output_directory,
    write_csv_atomic,
    write_json_atomic,
)
from scripts.eval.physics_ejecta_bridge import (
    PhysicsEjectaBridge,
    bhattacharyya_score,
    extract_lightcurve_features,
    weighted_gaussian,
)

CONTROL_SPECS = {
    "shuffle_ejecta_on_gw_side": {
        "side": "gw",
        "dimensions": (0, 1),
        "targets": (
            "chirp_mass_detector",
            "mass_ratio",
            "chi_eff",
            "primary_spin_z",
        ),
    },
    "shuffle_costheta_on_lc_side": {
        "side": "lc",
        "dimensions": (2,),
        "targets": ("abs_costheta",),
    },
    "shuffle_distance_on_lc_side": {
        "side": "lc",
        "dimensions": (3,),
        "targets": ("log10_distance_gpc",),
    },
}
PAIR_FIELDS = (
    "control",
    "shuffle_id",
    "target_parameter",
    "source_type",
    "pair_id",
    "n_curve_pairs",
    "directional_win_rate",
)
SUMMARY_FIELDS = (
    "control",
    "target_parameter",
    "source",
    "n_pairs",
    "n_shuffles",
    "estimate",
    "null_value",
    "ci95_low",
    "ci95_high",
    "two_sided_sign_flip_p",
    "shuffle_estimate_sd",
    "shuffle_estimate_min",
    "shuffle_estimate_max",
    "ci_covers_0_5",
)


def _stable_seed(seed: int, *parts: Any) -> int:
    payload = "|".join([str(int(seed)), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def deranged_donor_indices(size: int, seed: int) -> np.ndarray:
    """Return a deterministic permutation with no fixed parent event."""
    if int(size) < 2:
        raise ValueError("A label shuffle requires at least two parent events")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(size))
    donor = np.empty(int(size), dtype=np.int64)
    donor[order] = np.roll(order, 1)
    if np.any(donor == np.arange(int(size))):
        raise AssertionError("Derangement unexpectedly contains a fixed point")
    return donor


def permute_parent_label_dimensions(
    values: np.ndarray,
    parent_ids: np.ndarray,
    dimensions: Sequence[int],
    *,
    seed: int,
) -> np.ndarray:
    """Permute selected label dimensions consistently at parent-event grain."""
    labels = np.asarray(values, dtype=np.float64)
    parents = np.asarray(parent_ids, dtype=np.int64)
    unique, first, inverse = np.unique(parents, return_index=True, return_inverse=True)
    donor = deranged_donor_indices(len(unique), int(seed))
    result = labels.copy()
    dims = np.asarray(tuple(dimensions), dtype=np.int64)
    donor_values = labels[first[donor]][:, dims]
    result[:, dims] = donor_values[inverse]
    return result


def _gw_posterior_from_labels(
    bridge: PhysicsEjectaBridge,
    source: str,
    scalars: np.ndarray,
    neighbors: Any,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.asarray(labels[neighbors.indices], dtype=np.float64).copy()
    psi_center = bridge._key(source, "psi_center")
    psi_scale = bridge._key(source, "psi_scale")
    query = np.asarray(scalars, dtype=np.float64)
    samples[:, 2] = (abs(float(query[4])) - psi_center[2]) / psi_scale[2]
    samples[:, 3] = (math.log10(float(query[5])) - psi_center[3]) / psi_scale[3]
    sigma_log_distance = (
        float(query[6]) / float(query[5]) / math.log(10.0) / psi_scale[3]
    )
    extra = np.zeros(4, dtype=np.float64)
    extra[3] = sigma_log_distance**2
    hp = bridge.hyperparameters(source, "gw")
    return weighted_gaussian(
        samples,
        neighbors.weights,
        covariance_floor=float(hp["covariance_floor"]),
        extra_diagonal=extra,
    )


def _lc_posterior_from_labels(
    bridge: PhysicsEjectaBridge,
    source: str,
    neighbors: Any,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    hp = bridge.hyperparameters(source, "lc")
    return weighted_gaussian(
        np.asarray(labels[neighbors.indices], dtype=np.float64),
        neighbors.weights,
        covariance_floor=float(hp["covariance_floor"]),
    )


def _win(value: float) -> float:
    return 1.0 if value > 0.0 else (0.0 if value < 0.0 else 0.5)


def _score_pair(
    gw_a: tuple[np.ndarray, np.ndarray],
    gw_b: tuple[np.ndarray, np.ndarray],
    lc_a: tuple[np.ndarray, np.ndarray],
    lc_b: tuple[np.ndarray, np.ndarray],
) -> float:
    score_aa = bhattacharyya_score(*gw_a, *lc_a)
    score_ab = bhattacharyya_score(*gw_a, *lc_b)
    score_ba = bhattacharyya_score(*gw_b, *lc_a)
    score_bb = bhattacharyya_score(*gw_b, *lc_b)
    return 0.5 * (_win(score_aa - score_ab) + _win(score_bb - score_ba))


def _hierarchical_summary(
    frame: pd.DataFrame,
    *,
    source: str,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: int,
) -> dict[str, Any]:
    matrices: list[np.ndarray] = []
    if source == "source_macro":
        source_groups = [group for _, group in frame.groupby("source_type", sort=True)]
        if len(source_groups) != 2:
            raise ValueError("source_macro shuffle QA requires both BNS and NSBH")
    else:
        source_groups = [frame[frame["source_type"] == source]]
    for group in source_groups:
        matrix = group.pivot(
            index="shuffle_id", columns="pair_id", values="directional_win_rate"
        )
        if matrix.isna().any().any():
            raise ValueError("Incomplete shuffle-by-pair matrix")
        matrices.append(matrix.to_numpy(dtype=np.float64))
    n_shuffles = matrices[0].shape[0]
    if any(matrix.shape[0] != n_shuffles for matrix in matrices):
        raise ValueError("Sources have inconsistent shuffle counts")
    pair_means = [matrix.mean(axis=0) for matrix in matrices]
    estimate = float(np.mean([values.mean() for values in pair_means]))
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(int(bootstrap_samples), dtype=np.float64)
    for start in range(0, int(bootstrap_samples), 100):
        stop = min(start + 100, int(bootstrap_samples))
        batch = stop - start
        shuffle_draws = rng.integers(0, n_shuffles, size=(batch, n_shuffles))
        source_draws: list[np.ndarray] = []
        for matrix in matrices:
            pair_draws = rng.integers(0, matrix.shape[1], size=(batch, matrix.shape[1]))
            sampled = matrix[shuffle_draws[:, :, None], pair_draws[:, None, :]].mean(
                axis=(1, 2)
            )
            source_draws.append(sampled)
        bootstrap[start:stop] = np.mean(source_draws, axis=0)
    centered_groups = [values - 0.5 for values in pair_means]
    permutation = np.empty(int(permutation_samples), dtype=np.float64)
    for start in range(0, int(permutation_samples), 250):
        stop = min(start + 250, int(permutation_samples))
        draws = []
        for centered in centered_groups:
            signs = rng.choice(
                np.asarray([-1.0, 1.0]), size=(stop - start, centered.size)
            )
            draws.append((signs * centered).mean(axis=1))
        permutation[start:stop] = np.mean(draws, axis=0)
    centered_observed = estimate - 0.5
    extreme = np.count_nonzero(np.abs(permutation) >= abs(centered_observed))
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    per_shuffle = []
    for shuffle_id in range(n_shuffles):
        per_shuffle.append(
            float(np.mean([matrix[shuffle_id].mean() for matrix in matrices]))
        )
    return {
        "n_pairs": int(sum(matrix.shape[1] for matrix in matrices)),
        "n_shuffles": int(n_shuffles),
        "estimate": estimate,
        "null_value": 0.5,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "two_sided_sign_flip_p": float((1 + extreme) / (int(permutation_samples) + 1)),
        "shuffle_estimate_sd": float(np.std(per_shuffle, ddof=1)),
        "shuffle_estimate_min": float(np.min(per_shuffle)),
        "shuffle_estimate_max": float(np.max(per_shuffle)),
        "ci_covers_0_5": bool(low <= 0.5 <= high),
    }


def run(
    config_path: Path,
    *,
    n_shuffles: int,
    bootstrap_samples: int,
    permutation_samples: int,
    device_name: str,
) -> None:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    cfg, _, bridge_spec = directional.normalise_v2_config(raw, config_path)
    formal_dir = Path(cfg["output_dir"])
    success_path = formal_dir / "_SUCCESS.json"
    if not success_path.is_file():
        raise FileNotFoundError(f"Formal v2 result is incomplete: {formal_dir}")
    event_frame = pd.read_csv(formal_dir / "event_pairs.csv")
    curve_frame = pd.read_csv(formal_dir / "curve_pairs.csv")
    primary = event_frame[np.isclose(event_frame["caliper_iqr"], 0.5)].copy()
    pair_ids = set(primary["pair_id"].astype(str))
    primary_curves = curve_frame[
        curve_frame["pair_id"].astype(str).isin(pair_ids)
    ].copy()
    digest = json.loads((formal_dir / "run_manifest.json").read_text())[
        "pair_manifest_sha256"
    ]
    if digest != cfg["expected_pair_manifest_sha256"]:
        raise AssertionError(f"Formal pair digest changed: {digest}")
    output_dir = formal_dir / "label_shuffle_qa"
    manifest = prepare_output_directory(
        output_dir,
        manifest={
            "experiment_id": f"{raw['experiment_id']}_label_shuffle_qa",
            "config_path": str(config_path),
            "formal_result_dir": str(formal_dir),
            "pair_manifest_sha256": digest,
            "bridge_artifact_path": str(bridge_spec["artifact_path"]),
            "n_label_shuffles": int(n_shuffles),
            "bootstrap_samples": int(bootstrap_samples),
            "permutation_samples": int(permutation_samples),
            "seed": int(cfg["seed"]),
            "test_truth_fields_used": [],
        },
    )
    metadata = v1._read_metadata(cfg["test_data_path"], cfg["comparison_window"])
    selected_optical = np.unique(
        primary_curves[["optical_index_a", "optical_index_b"]].to_numpy(dtype=np.int64)
    )
    bank, _ = base.load_selected_positive_bank(
        cfg["test_data_path"],
        selected_optical,
        runtime_input_window_start=float(cfg["comparison_window"][0]),
        runtime_input_window_end=float(cfg["comparison_window"][1]),
    )
    with h5py.File(cfg["test_data_path"], "r") as handle:
        psfflux_zp = float(handle.attrs["psfflux_zp"])
        lupt_b_njy = np.asarray(handle.attrs["lupt_b_njy"], dtype=np.float64)
    features, feature_mask, cadence = extract_lightcurve_features(
        np.asarray(bank["times"]),
        np.asarray(bank["values"]),
        np.asarray(bank["masks"]),
        np.asarray(bank["errors"]),
        psfflux_zp=psfflux_zp,
        lupt_b_njy=lupt_b_njy,
    )
    bridge = PhysicsEjectaBridge(bridge_spec["artifact_path"])
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA label-shuffle QA requested but unavailable")
    optical_ids = np.asarray(bank["source_optical_indices"], dtype=np.int64)
    optical_parents = np.asarray(bank["gw_indices"], dtype=np.int64)
    lc_base: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    lc_neighbors: dict[tuple[str, int], Any] = {}
    for source in ("bns", "nsbh"):
        rows = np.asarray(
            [
                index
                for index, parent in enumerate(optical_parents.tolist())
                if str(metadata["gw_source_types"][parent]) == source
            ],
            dtype=np.int64,
        )
        means, covariances, neighbors = bridge.lc_posteriors_batch(
            features[rows],
            feature_mask[rows],
            cadence[rows],
            source,
            device=str(device),
            batch_size=int(cfg.get("bridge_lc_batch_size", 64)),
        )
        for local, row in enumerate(rows.tolist()):
            key = (source, int(optical_ids[row]))
            lc_base[key] = (means[local], covariances[local])
            lc_neighbors[key] = neighbors[local]
    all_gw = sorted(
        {
            int(value)
            for value in primary[["gw_a", "gw_b"]].to_numpy(dtype=np.int64).ravel()
        }
    )
    gw_base: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    gw_neighbors: dict[int, Any] = {}
    for gw_id in all_gw:
        source = str(metadata["gw_source_types"][gw_id])
        mean, covariance, neighbors = bridge.gw_posterior(
            metadata["gw_scalars"][gw_id], source
        )
        gw_base[gw_id] = (mean, covariance)
        gw_neighbors[gw_id] = neighbors
    curves_by_pair = {
        str(pair_id): group.to_dict("records")
        for pair_id, group in primary_curves.groupby("pair_id", sort=False)
    }
    pair_rows: list[dict[str, Any]] = []
    seed = int(cfg["seed"])
    for shuffle_id in tqdm(range(int(n_shuffles)), desc="Bridge label shuffles"):
        for control, spec in CONTROL_SPECS.items():
            control_pairs = primary[primary["target_parameter"].isin(spec["targets"])]
            gw_control = gw_base
            lc_control = lc_base
            if spec["side"] == "gw":
                gw_control = dict(gw_base)
                for source in ("bns", "nsbh"):
                    labels = bridge._key(source, "gw_psi")
                    parents = bridge._key(source, "gw_parent_ids")
                    shuffled = permute_parent_label_dimensions(
                        labels,
                        parents,
                        spec["dimensions"],
                        seed=_stable_seed(seed, control, source, shuffle_id),
                    )
                    for gw_id in {
                        int(value)
                        for value in control_pairs.loc[
                            control_pairs["source_type"] == source, ["gw_a", "gw_b"]
                        ]
                        .to_numpy(dtype=np.int64)
                        .ravel()
                    }:
                        gw_control[gw_id] = _gw_posterior_from_labels(
                            bridge,
                            source,
                            metadata["gw_scalars"][gw_id],
                            gw_neighbors[gw_id],
                            shuffled,
                        )
            else:
                lc_control = dict(lc_base)
                for source in ("bns", "nsbh"):
                    labels = bridge._key(source, "lc_psi")
                    parents = bridge._key(source, "lc_parent_ids")
                    shuffled = permute_parent_label_dimensions(
                        labels,
                        parents,
                        spec["dimensions"],
                        seed=_stable_seed(seed, control, source, shuffle_id),
                    )
                    source_optical = {
                        int(value)
                        for value in control_pairs.loc[
                            control_pairs["source_type"] == source, "pair_id"
                        ]
                        .astype(str)
                        .map(curves_by_pair)
                        .explode()
                        .dropna()
                        .map(
                            lambda row: (row["optical_index_a"], row["optical_index_b"])
                        )
                        .explode()
                    }
                    for optical_id in source_optical:
                        key = (source, optical_id)
                        lc_control[key] = _lc_posterior_from_labels(
                            bridge, source, lc_neighbors[key], shuffled
                        )
            for pair in control_pairs.to_dict("records"):
                source = str(pair["source_type"])
                curve_wins = []
                for curve in curves_by_pair[str(pair["pair_id"])]:
                    curve_wins.append(
                        _score_pair(
                            gw_control[int(pair["gw_a"])],
                            gw_control[int(pair["gw_b"])],
                            lc_control[(source, int(curve["optical_index_a"]))],
                            lc_control[(source, int(curve["optical_index_b"]))],
                        )
                    )
                pair_rows.append(
                    {
                        "control": control,
                        "shuffle_id": int(shuffle_id),
                        "target_parameter": str(pair["target_parameter"]),
                        "source_type": source,
                        "pair_id": str(pair["pair_id"]),
                        "n_curve_pairs": len(curve_wins),
                        "directional_win_rate": float(np.mean(curve_wins)),
                    }
                )
    pair_frame = pd.DataFrame(pair_rows)
    summary_rows: list[dict[str, Any]] = []
    for (control, target), group in pair_frame.groupby(
        ["control", "target_parameter"], sort=True
    ):
        source = directional._primary_source(str(target))
        summary_rows.append(
            {
                "control": str(control),
                "target_parameter": str(target),
                "source": source,
                **_hierarchical_summary(
                    group,
                    source=source,
                    bootstrap_samples=int(bootstrap_samples),
                    permutation_samples=int(permutation_samples),
                    seed=_stable_seed(seed, "summary", control, target),
                ),
            }
        )
    if not all(row["ci_covers_0_5"] for row in summary_rows):
        failed = [
            row["target_parameter"] for row in summary_rows if not row["ci_covers_0_5"]
        ]
        raise AssertionError(f"Label-shuffle CI excludes 0.5 for {failed}")
    writer = AtomicGzipCsvWriter(
        output_dir / "pair_label_shuffle_wins.csv.gz", PAIR_FIELDS
    )
    try:
        writer.writerows(pair_rows)
        writer.commit()
    finally:
        writer.close()
    write_csv_atomic(
        output_dir / "label_shuffle_summary.csv", summary_rows, SUMMARY_FIELDS
    )
    result = {
        "status": "complete",
        "pair_manifest_sha256": digest,
        "controls": CONTROL_SPECS,
        "summary": summary_rows,
        "protocol": {
            "shuffle_grain": "parent_event_within_source",
            "ejecta_control_side": "gw_reference_labels",
            "costheta_control_side": "lc_reference_labels",
            "distance_control_side": "lc_reference_labels",
            "uncertainty": "two_way_bootstrap_over_shuffle_replicates_and_event_pairs",
            "test_truth_fields_used": [],
        },
    }
    write_json_atomic(output_dir / "label_shuffle_summary.json", result)
    mark_run_success(
        output_dir,
        manifest,
        [
            "pair_label_shuffle_wins.csv.gz",
            "label_shuffle_summary.csv",
            "label_shuffle_summary.json",
        ],
    )
    bridge.arrays.close()
    del bank
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-shuffles", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--permutation-samples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.n_shuffles, args.bootstrap_samples, args.permutation_samples) < 1:
        parser.error("shuffle/bootstrap/permutation counts must be positive")
    run(
        Path(args.config).expanduser().resolve(),
        n_shuffles=args.n_shuffles,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
