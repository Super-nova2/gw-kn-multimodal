"""Mixed KN/non-KN gallery construction and metric helpers."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np
from retrieval_gallery import uniform_random_tie_metric_contributions

MIXED_SCOPES = ("all", "kn_only", "nonkn_only")


def allocate_mixed_negative_counts(
    gallery_size: int, kn_fraction: float
) -> tuple[int, int]:
    """Allocate negative slots with deterministic nearest-integer rounding."""
    gallery_size = int(gallery_size)
    fraction = float(kn_fraction)
    if gallery_size < 3:
        raise ValueError("mixed gallery_size must be >= 3")
    if not 0.0 < fraction < 1.0:
        raise ValueError("kn_fraction must be in (0, 1)")
    n_negative = gallery_size - 1
    n_kn = int(np.floor(n_negative * fraction + 0.5))
    n_kn = min(max(n_kn, 1), n_negative - 1)
    return n_kn, n_negative - n_kn


def select_source_balanced_queries(
    gw_ids: Sequence[int],
    source_types: Sequence[str],
    *,
    queries_per_source: int,
    seed: int,
    source_labels: Sequence[str] = ("bns", "nsbh"),
) -> list[int]:
    """Select a deterministic, source-balanced set of detected GW parents."""
    source = np.asarray([str(value).strip().lower() for value in source_types])
    requested = int(queries_per_source)
    if requested < 1:
        raise ValueError("queries_per_source must be >= 1")
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    available = np.asarray(sorted({int(value) for value in gw_ids}), dtype=np.int64)
    for label in source_labels:
        eligible = available[source[available] == str(label).lower()]
        if eligible.size < requested:
            raise ValueError(
                f"source={label!r} has {eligible.size} detected parents; "
                f"need {requested}."
            )
        chosen = rng.choice(eligible, size=requested, replace=False)
        selected.extend(int(value) for value in chosen.tolist())
    return sorted(selected)


def _positive_index(
    gw_positive_indices: Mapping[int, Sequence[int]],
    gw_id: int,
    *,
    trial: int,
    seed: int,
) -> int:
    values = np.asarray(gw_positive_indices[int(gw_id)], dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"GW event {gw_id} has no positive optical samples")
    rng = np.random.default_rng(
        int(seed) + 15485863 * int(trial) + 32452843 * int(gw_id)
    )
    return int(values[rng.integers(values.size)])


def _random_kn_sequence(
    parent: np.ndarray,
    query_gw_id: int,
    *,
    n_needed: int,
    trial: int,
    seed: int,
) -> np.ndarray:
    by_parent = {
        int(parent_id): np.flatnonzero(parent == int(parent_id))
        for parent_id in np.unique(parent).tolist()
    }
    eligible = np.asarray(
        [value for value in sorted(by_parent) if value != int(query_gw_id)],
        dtype=np.int64,
    )
    if eligible.size < int(n_needed):
        raise ValueError(
            f"GW event {query_gw_id} has {eligible.size} eligible KN parents; "
            f"need {n_needed}."
        )
    rng = np.random.default_rng(
        int(seed) + 49979687 * int(trial) + 67867967 * int(query_gw_id)
    )
    selected_parent = rng.choice(eligible, size=int(n_needed), replace=False)
    return np.asarray(
        [
            int(by_parent[int(parent_id)][rng.integers(by_parent[int(parent_id)].size)])
            for parent_id in selected_parent.tolist()
        ],
        dtype=np.int64,
    )


def _matched_kn_sequence(
    parent: np.ndarray,
    source_types: Sequence[str],
    nuisance_features: np.ndarray,
    positive_index: int,
    query_gw_id: int,
    *,
    n_needed: int,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray([str(value).strip().lower() for value in source_types])
    features = np.asarray(nuisance_features, dtype=np.float64)
    standardized = np.empty_like(features)
    for label in sorted(np.unique(source).tolist()):
        rows = np.flatnonzero(source[parent] == label)
        values = features[rows]
        median = np.median(values, axis=0)
        scale = np.median(np.abs(values - median), axis=0) * 1.4826
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized[rows] = (values - median) / scale
    candidates = np.flatnonzero(
        (source[parent] == source[int(query_gw_id)]) & (parent != int(query_gw_id))
    )
    costs = np.abs(standardized[candidates] - standardized[int(positive_index)]).sum(
        axis=1
    )
    order = np.lexsort((candidates, costs))
    rows: list[int] = []
    row_costs: list[float] = []
    seen: set[int] = set()
    for local_idx in order.tolist():
        row = int(candidates[local_idx])
        candidate_parent = int(parent[row])
        if candidate_parent in seen:
            continue
        seen.add(candidate_parent)
        rows.append(row)
        row_costs.append(float(costs[local_idx]))
        if len(rows) >= int(n_needed):
            break
    if len(rows) < int(n_needed):
        raise ValueError(
            f"GW event {query_gw_id} has {len(rows)} nuisance-matched parents; "
            f"need {n_needed}."
        )
    return np.asarray(rows, dtype=np.int64), np.asarray(row_costs, dtype=np.float32)


def build_mixed_galleries(
    *,
    gw_positive_indices: Mapping[int, Sequence[int]],
    query_gw_ids: Sequence[int],
    optical_parent_gw_idx: Sequence[int],
    nonkn_candidate_sequences: Mapping[tuple[int, int], Mapping[str, Any]],
    gallery_sizes: Sequence[int],
    n_trials: int,
    seed: int,
    kn_fraction: float,
    kn_candidate_mode: str = "kn_random",
    gw_source_types: Sequence[str] | None = None,
    nuisance_features: np.ndarray | None = None,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Build deterministic mixed galleries with separately nested KN/non-KN prefixes."""
    mode = str(kn_candidate_mode).strip().lower()
    if mode not in {"kn_random", "kn_nuisance_matched"}:
        raise ValueError(
            "kn_candidate_mode must be 'kn_random' or 'kn_nuisance_matched'"
        )
    if mode == "kn_nuisance_matched" and (
        gw_source_types is None or nuisance_features is None
    ):
        raise ValueError(
            "kn_nuisance_matched requires gw_source_types and nuisance_features"
        )
    sizes = sorted({int(value) for value in gallery_sizes})
    allocations = {
        size: allocate_mixed_negative_counts(size, kn_fraction) for size in sizes
    }
    max_kn = max(value[0] for value in allocations.values())
    max_nonkn = max(value[1] for value in allocations.values())
    parent = np.asarray(optical_parent_gw_idx, dtype=np.int64)
    galleries: dict[tuple[int, int, int], dict[str, Any]] = {}
    for trial in range(int(n_trials)):
        for gw_id_raw in sorted(int(value) for value in query_gw_ids):
            gw_id = int(gw_id_raw)
            positive = _positive_index(
                gw_positive_indices, gw_id, trial=trial, seed=seed
            )
            if mode == "kn_random":
                kn_sequence = _random_kn_sequence(
                    parent,
                    gw_id,
                    n_needed=max_kn,
                    trial=trial,
                    seed=seed,
                )
                match_costs = np.full(max_kn, np.nan, dtype=np.float32)
            else:
                kn_sequence, match_costs = _matched_kn_sequence(
                    parent,
                    gw_source_types,
                    nuisance_features,
                    positive,
                    gw_id,
                    n_needed=max_kn,
                )
            seq = nonkn_candidate_sequences.get((trial, gw_id))
            if seq is None:
                raise KeyError(f"Missing non-KN sequence for trial={trial}, gw={gw_id}")
            nonkn_sequence = np.asarray(
                seq.get("candidate_indices", []), dtype=np.int64
            ).reshape(-1)
            if nonkn_sequence.size < max_nonkn:
                raise ValueError(
                    f"trial={trial}, gw={gw_id} has {nonkn_sequence.size} non-KN "
                    f"candidates; need {max_nonkn}."
                )
            nonkn_coords = np.asarray(
                seq.get("synthetic_coordinates", []), dtype=np.float32
            ).reshape(-1, 2)
            nonkn_dt = np.asarray(seq.get("abs_dt_days", []), dtype=np.float32).reshape(
                -1
            )
            for size in sizes:
                n_kn, n_nonkn = allocations[size]
                galleries[(size, trial, gw_id)] = {
                    "positive_index": positive,
                    "kn_negative_indices": kn_sequence[:n_kn].copy(),
                    "kn_negative_parent_gw_ids": parent[kn_sequence[:n_kn]].copy(),
                    "kn_negative_match_costs": match_costs[:n_kn].copy(),
                    "nonkn_negative_indices": nonkn_sequence[:n_nonkn].copy(),
                    "nonkn_synthetic_coordinates": nonkn_coords[:n_nonkn].copy(),
                    "nonkn_training_aligned_dt_days": nonkn_dt[:n_nonkn].copy(),
                    "requested_gallery_size": size,
                    "actual_gallery_size": 1 + n_kn + n_nonkn,
                    "n_kn_negative": n_kn,
                    "n_nonkn_negative": n_nonkn,
                    "coverage_met": True,
                    "is_undersized": False,
                }
    validate_mixed_galleries(galleries, parent)
    return galleries


def validate_mixed_galleries(
    galleries: Mapping[tuple[int, int, int], Mapping[str, Any]],
    optical_parent_gw_idx: Sequence[int],
) -> None:
    parent = np.asarray(optical_parent_gw_idx, dtype=np.int64)
    grouped: dict[tuple[int, int], list[tuple[int, np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )
    for (size, trial, gw_id), spec in galleries.items():
        kn = np.asarray(spec["kn_negative_indices"], dtype=np.int64)
        nonkn = np.asarray(spec["nonkn_negative_indices"], dtype=np.int64)
        nonkn_coords = np.asarray(
            spec["nonkn_synthetic_coordinates"], dtype=np.float32
        ).reshape(-1, 2)
        nonkn_dt = np.asarray(
            spec["nonkn_training_aligned_dt_days"], dtype=np.float32
        ).reshape(-1)
        if nonkn_coords.shape[0] != nonkn.size or nonkn_dt.size != nonkn.size:
            raise ValueError(
                f"non-KN metadata length mismatch in trial={trial}, gw={gw_id}"
            )
        if not np.all(np.isfinite(nonkn_coords)) or not np.all(np.isfinite(nonkn_dt)):
            raise ValueError(f"Non-finite non-KN metadata in trial={trial}, gw={gw_id}")
        kn_parent = parent[kn]
        if np.any(kn_parent == int(gw_id)):
            raise ValueError(f"Same-parent leakage in trial={trial}, gw={gw_id}")
        if np.unique(kn_parent).size != kn_parent.size:
            raise ValueError(f"Duplicate KN parent in trial={trial}, gw={gw_id}")
        if np.unique(nonkn).size != nonkn.size:
            raise ValueError(f"Duplicate non-KN row in trial={trial}, gw={gw_id}")
        grouped[(int(trial), int(gw_id))].append((int(size), kn, nonkn))
    for key, values in grouped.items():
        values.sort(key=lambda item: item[0])
        for (
            (_, small_kn, small_nonkn),
            (_, large_kn, large_nonkn),
        ) in pairwise(values):
            if not np.array_equal(small_kn, large_kn[: small_kn.size]):
                raise ValueError(f"KN prefix nesting failed for trial/gw={key}")
            if not np.array_equal(small_nonkn, large_nonkn[: small_nonkn.size]):
                raise ValueError(f"non-KN prefix nesting failed for trial/gw={key}")


def mixed_gallery_identity_digest(
    galleries: Mapping[tuple[int, int, int], Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for key in sorted(galleries):
        spec = galleries[key]
        digest.update(np.asarray(key, dtype="<i8").tobytes())
        digest.update(np.asarray([spec["positive_index"]], dtype="<i8").tobytes())
        for field in (
            "kn_negative_indices",
            "kn_negative_parent_gw_ids",
            "nonkn_negative_indices",
        ):
            digest.update(np.asarray(spec[field], dtype="<i8").tobytes())
        digest.update(
            np.asarray(spec["nonkn_synthetic_coordinates"], dtype="<f4").tobytes()
        )
        digest.update(
            np.asarray(spec["nonkn_training_aligned_dt_days"], dtype="<f4").tobytes()
        )
    return digest.hexdigest()


def rank_metric_contributions(
    positive_score: float,
    negative_scores: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Return expected zero-based rank and retrieval metrics under uniform ties."""
    negative = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    positive = float(positive_score)
    tolerance = 1e-7 * max(1.0, abs(positive))
    better = int(np.count_nonzero(negative > positive + tolerance))
    tied = int(np.count_nonzero(np.abs(negative - positive) <= tolerance))
    contributions = uniform_random_tie_metric_contributions(better, tied)
    expected_rank = float(better + tied / 2.0)
    return expected_rank, contributions


def mixed_score_outcome(
    positive_score: float,
    kn_scores: Sequence[float],
    nonkn_scores: Sequence[float],
) -> dict[str, Any]:
    kn = np.asarray(kn_scores, dtype=np.float64).reshape(-1)
    nonkn = np.asarray(nonkn_scores, dtype=np.float64).reshape(-1)
    scopes = {
        "all": np.concatenate([kn, nonkn]),
        "kn_only": kn,
        "nonkn_only": nonkn,
    }
    result: dict[str, Any] = {}
    for scope, scores in scopes.items():
        rank, metrics = rank_metric_contributions(positive_score, scores)
        result[scope] = {"rank": rank, **metrics}
    best_kn = float(np.max(kn)) if kn.size else float("-inf")
    best_nonkn = float(np.max(nonkn)) if nonkn.size else float("-inf")
    best = max(float(positive_score), best_kn, best_nonkn)
    if np.isclose(best, float(positive_score)):
        top1_type = "positive"
    elif best_kn >= best_nonkn:
        top1_type = "kn"
    else:
        top1_type = "nonkn"
    result.update(
        {
            "positive_score": float(positive_score),
            "best_kn_score": best_kn,
            "best_nonkn_score": best_nonkn,
            "kn_margin": float(positive_score - best_kn),
            "nonkn_margin": float(positive_score - best_nonkn),
            "n_kn_above_positive": int(np.count_nonzero(kn > positive_score)),
            "n_nonkn_above_positive": int(np.count_nonzero(nonkn > positive_score)),
            "top1_type": top1_type,
        }
    )
    return result
