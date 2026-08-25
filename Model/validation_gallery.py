from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import h5py
import numpy as np
import torch

from mixed_retrieval import (
    MIXED_SCOPES,
    allocate_mixed_negative_counts,
    build_mixed_galleries,
    mixed_gallery_identity_digest,
    mixed_score_outcome,
)
from data_loader import load_gw_source_type_map
from retrieval_gallery import (
    aggregate_gallery_outcomes,
    build_prefixed_gallery_specs,
    build_synthetic_time_sky_candidate_sequences,
    compute_source_macro_and_gap,
)
from scripts.eval.eval_retrieval_comparison import (
    _build_gallery_query_cache,
    _is_dual_fusion_model,
    _score_candidate_bank_with_logits,
    enrich_gallery_outcomes,
    extract_negative_gallery_embeddings,
    extract_optical_candidate_embeddings,
    load_selected_positive_bank,
    remap_gallery_positive_indices,
    score_all_galleries_contrastive,
    score_all_galleries_multimodal,
)
from scripts.eval.evaluate import load_negative_optical_samples

SUPPORTED_VALIDATION_GALLERY_MODE = "synthetic_time_sky_hard"
MIXED_VALIDATION_GALLERY_MODE = "mixed_kn_nonkn"
SUPPORTED_MIXED_VALIDATION_CONDITIONS = {
    "positive_shared",
    "training_aligned",
}


def resolve_mixed_validation_settings(args) -> Dict[str, Any]:
    """Resolve validation-only mixed-gallery settings without changing training."""
    condition = str(
        getattr(args, "validation_gallery_condition", "positive_shared")
    ).strip().lower()
    if condition not in SUPPORTED_MIXED_VALIDATION_CONDITIONS:
        raise ValueError(
            "validation_gallery_condition must be 'positive_shared' or "
            "'training_aligned'."
        )
    training_fraction = float(getattr(args, "gallery_kn_distractor_fraction", 0.25))
    raw_fraction = getattr(args, "validation_gallery_kn_distractor_fraction", None)
    kn_fraction = training_fraction if raw_fraction is None else float(raw_fraction)
    if not 0.0 < kn_fraction < 1.0:
        raise ValueError(
            "validation_gallery_kn_distractor_fraction must be in (0, 1)."
        )
    training_empirical = float(
        getattr(args, "gallery_nonkn_empirical_fraction", 0.5)
    )
    raw_empirical = getattr(
        args, "validation_gallery_nonkn_empirical_fraction", None
    )
    empirical_fraction = (
        training_empirical if raw_empirical is None else float(raw_empirical)
    )
    if not 0.0 <= empirical_fraction <= 1.0:
        raise ValueError(
            "validation_gallery_nonkn_empirical_fraction must be in [0, 1]."
        )
    return {
        "condition": condition,
        "kn_fraction": kn_fraction,
        "nonkn_empirical_fraction": empirical_fraction,
    }


def mixed_validation_candidate_coordinates(
    condition: str,
    positive_coordinate: np.ndarray,
    nonkn_synthetic_coordinates: np.ndarray,
    *,
    n_kn: int,
    n_nonkn: int,
) -> Dict[str, np.ndarray | None]:
    """Build candidate-coordinate inputs for a mixed validation condition."""
    positive = np.asarray(positive_coordinate, dtype=np.float32).reshape(1, 2)
    kn_coordinates = np.repeat(positive, int(n_kn), axis=0)
    if condition == "positive_shared":
        return {
            "positive": positive,
            "kn": kn_coordinates,
            "nonkn": np.repeat(positive, int(n_nonkn), axis=0),
        }
    if condition == "training_aligned":
        return {
            "positive": None,
            "kn": kn_coordinates,
            "nonkn": np.asarray(nonkn_synthetic_coordinates, dtype=np.float32),
        }
    raise ValueError(f"Unsupported mixed validation condition: {condition}")


def parse_validation_gallery_sizes(value: Any) -> Sequence[int]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        values = list(value)
    sizes = sorted({int(item) for item in values})
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("validation_gallery_sizes must contain integers >= 2.")
    return sizes


def compute_hard_gallery_selection_summary(
    source_macro: Mapping[str, float],
    gallery_sizes: Sequence[int],
    *,
    mrr_weight: float,
    recall_at_1_weight: float,
    gallery_size_weights: Mapping[Any, float] | None = None,
) -> Dict[str, float]:
    sizes = [int(size) for size in gallery_sizes]
    raw_weights = gallery_size_weights or {size: 1.0 for size in sizes}
    size_weights = np.asarray(
        [
            float(raw_weights.get(size, raw_weights.get(str(size), 0.0)))
            for size in sizes
        ],
        dtype=np.float64,
    )
    if np.any(size_weights < 0.0) or float(size_weights.sum()) <= 0.0:
        raise ValueError(
            "validation gallery size weights must be non-negative and sum to > 0."
        )
    size_weights /= size_weights.sum()
    macro_mrr = float(
        np.dot(
            size_weights, [float(source_macro[f"gallery_{size}_mrr"]) for size in sizes]
        )
    )
    macro_r1 = float(
        np.dot(
            size_weights,
            [float(source_macro[f"gallery_{size}_recall_at_1"]) for size in sizes],
        )
    )
    mrr_weight = float(mrr_weight)
    recall_at_1_weight = float(recall_at_1_weight)
    weight_total = mrr_weight + recall_at_1_weight
    if mrr_weight < 0.0 or recall_at_1_weight < 0.0 or weight_total <= 0.0:
        raise ValueError(
            "validation gallery metric weights must be non-negative and sum to > 0."
        )
    score = (mrr_weight * macro_mrr + recall_at_1_weight * macro_r1) / weight_total
    return {
        "macro_mrr": macro_mrr,
        "macro_recall_at_1": macro_r1,
        "selection_score": float(score),
    }


def partition_validation_gw_ids(
    by_source: Mapping[str, Sequence[int]], *, fraction: float, seed: int
) -> Dict[str, Dict[str, list[int]]]:
    """Create deterministic, source-stratified and disjoint tune/confirmation pools."""
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("validation_gallery_tune_fraction must be in (0, 1).")
    rng = np.random.default_rng(int(seed))
    partitions = {"tune": {}, "confirmation": {}}
    for source, values in by_source.items():
        shuffled = np.asarray(sorted({int(value) for value in values}), dtype=np.int64)
        rng.shuffle(shuffled)
        cut = int(np.floor(shuffled.size * float(fraction)))
        partitions["tune"][source] = shuffled[:cut].tolist()
        partitions["confirmation"][source] = shuffled[cut:].tolist()
    return partitions


def _jsonable_gallery_manifest(
    *,
    galleries: Mapping[Any, Mapping[str, Any]],
    negative_source_indices: np.ndarray,
    gw_source_map: Mapping[int, str],
    gallery_sizes: Sequence[int],
    n_trials: int,
    seed: int,
    time_window_days: float,
    credible_level_max: float,
) -> Dict[str, Any]:
    instances = []
    for (gallery_size, trial, gw_id), spec in sorted(galleries.items()):
        compact_neg = np.asarray(spec["negative_indices"], dtype=np.int64)
        instances.append(
            {
                "gallery_size": int(gallery_size),
                "trial": int(trial),
                "gw_id": int(gw_id),
                "source_type": str(gw_source_map[int(gw_id)]),
                "positive_optical_index": int(
                    spec.get("source_positive_index", spec["positive_index"])
                ),
                "negative_optical_indices": negative_source_indices[compact_neg]
                .astype(np.int64, copy=False)
                .tolist(),
            }
        )
    return {
        "mode": SUPPORTED_VALIDATION_GALLERY_MODE,
        "gallery_sizes": [int(size) for size in gallery_sizes],
        "n_trials": int(n_trials),
        "seed": int(seed),
        "time_window_days": float(time_window_days),
        "credible_level_max": float(credible_level_max),
        "queries": [
            {"gw_id": int(gw_id), "source_type": str(gw_source_map[int(gw_id)])}
            for gw_id in sorted(gw_source_map)
        ],
        "instances": instances,
    }


def _assign_mixed_nonkn_time_deltas(
    sequences,
    empirical_dt,
    *,
    fraction,
    window_days,
    seed,
):
    empirical = np.asarray(empirical_dt, dtype=np.float64)
    empirical = empirical[np.isfinite(empirical) & (empirical >= 0.0)]
    if empirical.size == 0 and float(fraction) > 0.0:
        raise ValueError("No finite empirical KN delays for mixed validation.")
    for (trial, gw_id), sequence in sequences.items():
        n_values = int(np.asarray(sequence["candidate_indices"]).size)
        rng = np.random.default_rng(
            int(seed) + 433494437 * int(trial) + 2971215073 * int(gw_id)
        )
        n_empirical = int(np.floor(n_values * float(fraction) + 0.5))
        values = np.concatenate(
            [
                rng.choice(empirical, size=n_empirical, replace=True),
                rng.uniform(0.0, float(window_days), size=n_values - n_empirical),
            ]
        ).astype(np.float32)
        sequence["abs_dt_days"] = values[rng.permutation(n_values)]


def _aggregate_mixed_validation_outcomes(
    outcomes,
    *,
    gallery_sizes,
    n_trials,
    unique_gw,
    gw_source_map,
    mrr_weight,
    recall_at_1_weight,
    gallery_size_weights,
):
    metrics = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")
    scopes = {}
    for scope in MIXED_SCOPES:
        by_source = {}
        for source in ("bns", "nsbh"):
            source_metrics = {}
            source_gw = [
                int(gw_id) for gw_id in unique_gw if gw_source_map[int(gw_id)] == source
            ]
            for gallery_size in gallery_sizes:
                rows = [
                    outcomes[(int(gallery_size), trial, gw_id)][scope]
                    for trial in range(int(n_trials))
                    for gw_id in source_gw
                ]
                for metric in metrics:
                    source_metrics[f"gallery_{gallery_size}_{metric}"] = float(
                        np.mean([float(row[metric]) for row in rows])
                    )
            by_source[source] = source_metrics
        source_macro, source_gap = compute_source_macro_and_gap(by_source)
        overall = {}
        for gallery_size in gallery_sizes:
            rows = [
                outcomes[(int(gallery_size), trial, int(gw_id))][scope]
                for trial in range(int(n_trials))
                for gw_id in unique_gw
            ]
            for metric in metrics:
                overall[f"gallery_{gallery_size}_{metric}"] = float(
                    np.mean([float(row[metric]) for row in rows])
                )
        scope_summary = compute_hard_gallery_selection_summary(
            source_macro,
            gallery_sizes,
            mrr_weight=mrr_weight,
            recall_at_1_weight=recall_at_1_weight,
            gallery_size_weights=gallery_size_weights,
        )
        scopes[scope] = {
            "overall": overall,
            "by_source": by_source,
            "source_macro": source_macro,
            "source_gap": source_gap,
            **scope_summary,
        }
    selected = scopes["all"]
    summary = compute_hard_gallery_selection_summary(
        selected["source_macro"],
        gallery_sizes,
        mrr_weight=mrr_weight,
        recall_at_1_weight=recall_at_1_weight,
        gallery_size_weights=gallery_size_weights,
    )
    return {
        **selected,
        "coverage": {f"gallery_{size}": 1.0 for size in gallery_sizes},
        "mode": MIXED_VALIDATION_GALLERY_MODE,
        "scopes": scopes,
        **summary,
    }


@dataclass
class ValidationGalleryContext:
    data_path: str
    positive_bank: Dict[str, torch.Tensor]
    negative_bank_raw: Dict[str, Any]
    galleries: Dict[Any, Dict[str, Any]]
    unique_gw: Sequence[int]
    gw_source_map: Dict[int, str]
    gallery_sizes: Sequence[int]
    n_trials: int
    mrr_weight: float
    recall_at_1_weight: float
    gallery_size_weights: Mapping[Any, float] | None = None
    mode: str = SUPPORTED_VALIDATION_GALLERY_MODE
    kn_parent_gw_ids: np.ndarray | None = None
    condition: str = "positive_shared"
    kn_distractor_fraction: float = 0.25
    nonkn_empirical_fraction: float = 0.5

    @classmethod
    def build(cls, args, val_gw_map: Mapping[int, Sequence[int]]):
        mode = str(args.validation_gallery_mode).strip().lower()
        if mode == MIXED_VALIDATION_GALLERY_MODE:
            return cls._build_mixed(args, val_gw_map)
        if mode != SUPPORTED_VALIDATION_GALLERY_MODE:
            raise ValueError(
                "validation_gallery_mode must be "
                f"{SUPPORTED_VALIDATION_GALLERY_MODE!r} or "
                f"{MIXED_VALIDATION_GALLERY_MODE!r}."
            )
        if args.neg_data_path is None:
            raise ValueError("validation_gallery_enable requires neg_data_path.")

        gallery_sizes = parse_validation_gallery_sizes(args.validation_gallery_sizes)
        queries_per_source = int(args.validation_gallery_queries_per_source)
        if queries_per_source < 1:
            raise ValueError("validation_gallery_queries_per_source must be >= 1.")
        n_trials = int(args.validation_gallery_trials)
        if n_trials < 1:
            raise ValueError("validation_gallery_trials must be >= 1.")

        source_type_map = load_gw_source_type_map(args.data_path)
        min_lc = max(1, int(getattr(args, "min_lc_per_gw", 1)))
        by_source: Dict[str, list[int]] = {"bns": [], "nsbh": []}
        for gw_id, optical_indices in val_gw_map.items():
            source = source_type_map[int(gw_id)]
            if source in by_source and len(optical_indices) >= min_lc:
                by_source[source].append(int(gw_id))

        partition = str(getattr(args, "validation_gallery_partition", "all")).lower()
        if partition not in {"all", "tune", "confirmation"}:
            raise ValueError(
                "validation_gallery_partition must be all, tune, or confirmation."
            )
        if partition != "all":
            pools = partition_validation_gw_ids(
                by_source,
                fraction=float(getattr(args, "validation_gallery_tune_fraction", 0.75)),
                seed=int(getattr(args, "validation_gallery_partition_seed", 42)),
            )
            by_source = pools[partition]

        rng = np.random.default_rng(int(args.validation_gallery_seed))
        selected_gw = []
        for source in ("bns", "nsbh"):
            candidates = np.asarray(sorted(by_source[source]), dtype=np.int64)
            if candidates.size < queries_per_source:
                raise ValueError(
                    f"Validation source '{source}' has {candidates.size} eligible GW events; "
                    f"validation_gallery_queries_per_source={queries_per_source}."
                )
            selected = rng.choice(
                candidates,
                size=queries_per_source,
                replace=False,
            )
            selected_gw.extend(int(gw_id) for gw_id in selected.tolist())
        selected_gw = sorted(selected_gw)

        selected_positive_map = {
            int(gw_id): np.asarray(val_gw_map[int(gw_id)], dtype=np.int64)
            for gw_id in selected_gw
        }
        negative_pool_size = 4 * max(gallery_sizes)
        negative_bank_raw = load_negative_optical_samples(
            args.neg_data_path,
            args.neg_group,
            n_samples=negative_pool_size,
            seed=int(args.validation_gallery_seed),
            runtime_input_window_start=float(args.ref_start),
            runtime_input_window_end=float(args.ref_end),
        )
        if negative_bank_raw is None:
            raise ValueError(
                "Failed to load negative optical samples for validation gallery."
            )

        candidate_sequences, _gw_skymaps, _gw_times = (
            build_synthetic_time_sky_candidate_sequences(
                test_data_path=args.data_path,
                unique_gw_ids=selected_gw,
                neg_optical_data=negative_bank_raw,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                seed=int(args.validation_gallery_seed),
                time_window_days=float(args.validation_gallery_time_window_days),
                credible_level_max=float(args.validation_gallery_credible_level_max),
            )
        )
        galleries, unique_gw = build_prefixed_gallery_specs(
            gw_positive_indices=selected_positive_map,
            candidate_sequences=candidate_sequences,
            gallery_sizes=gallery_sizes,
            n_trials=n_trials,
            seed=int(args.validation_gallery_seed),
            include_undersized=bool(args.validation_gallery_include_undersized),
        )
        if not galleries:
            raise ValueError(
                "Validation hard-gallery construction produced no instances."
            )

        selected_positive_indices = np.unique(
            np.asarray(
                [int(spec["positive_index"]) for spec in galleries.values()],
                dtype=np.int64,
            )
        )
        positive_bank, positive_remap = load_selected_positive_bank(
            args.data_path,
            selected_positive_indices,
            runtime_input_window_start=float(args.ref_start),
            runtime_input_window_end=float(args.ref_end),
        )
        galleries = remap_gallery_positive_indices(galleries, positive_remap)
        gw_source_map = {int(gw_id): source_type_map[int(gw_id)] for gw_id in unique_gw}

        manifest = _jsonable_gallery_manifest(
            galleries=galleries,
            negative_source_indices=np.asarray(
                negative_bank_raw["source_indices"], dtype=np.int64
            ),
            gw_source_map=gw_source_map,
            gallery_sizes=gallery_sizes,
            n_trials=n_trials,
            seed=int(args.validation_gallery_seed),
            time_window_days=float(args.validation_gallery_time_window_days),
            credible_level_max=float(args.validation_gallery_credible_level_max),
        )
        manifest_dir = os.path.join(args.ckpt_path, "ALBEF")
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(
            manifest_dir, f"validation_gallery_manifest_{partition}.json"
        )
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(
            "Validation hard gallery prepared: "
            f"queries={len(unique_gw)}, instances={len(galleries)}, "
            f"sizes={list(gallery_sizes)}, negative_pool={negative_pool_size}, "
            f"manifest={manifest_path}"
        )

        return cls(
            data_path=args.data_path,
            positive_bank=positive_bank,
            negative_bank_raw=negative_bank_raw,
            galleries=galleries,
            unique_gw=unique_gw,
            gw_source_map=gw_source_map,
            gallery_sizes=gallery_sizes,
            n_trials=n_trials,
            mrr_weight=float(args.validation_gallery_mrr_weight),
            recall_at_1_weight=float(args.validation_gallery_recall_at_1_weight),
            gallery_size_weights=getattr(args, "validation_gallery_size_weights", None),
        )

    @classmethod
    def _build_mixed(cls, args, val_gw_map: Mapping[int, Sequence[int]]):
        if args.neg_data_path is None:
            raise ValueError("mixed validation gallery requires neg_data_path.")
        gallery_sizes = parse_validation_gallery_sizes(args.validation_gallery_sizes)
        settings = resolve_mixed_validation_settings(args)
        for size in gallery_sizes:
            allocate_mixed_negative_counts(size, settings["kn_fraction"])
        queries_per_source = int(args.validation_gallery_queries_per_source)
        n_trials = int(args.validation_gallery_trials)
        if queries_per_source < 1 or n_trials < 1:
            raise ValueError("mixed validation requires positive query/trial counts.")

        source_type_map = load_gw_source_type_map(args.data_path)
        min_lc = max(1, int(getattr(args, "min_lc_per_gw", 1)))
        by_source = {"bns": [], "nsbh": []}
        for gw_id, optical_indices in val_gw_map.items():
            source = source_type_map[int(gw_id)]
            if source in by_source and len(optical_indices) >= min_lc:
                by_source[source].append(int(gw_id))
        partition = str(getattr(args, "validation_gallery_partition", "all")).lower()
        if partition != "all":
            pools = partition_validation_gw_ids(
                by_source,
                fraction=float(getattr(args, "validation_gallery_tune_fraction", 0.75)),
                seed=int(getattr(args, "validation_gallery_partition_seed", 42)),
            )
            by_source = pools[partition]

        seed = int(args.validation_gallery_seed)
        rng = np.random.default_rng(seed)
        selected_gw = []
        for source in ("bns", "nsbh"):
            candidates = np.asarray(sorted(by_source[source]), dtype=np.int64)
            if candidates.size < queries_per_source:
                raise ValueError(
                    f"Validation source {source!r} has {candidates.size} events; "
                    f"need {queries_per_source}."
                )
            selected_gw.extend(
                int(value)
                for value in rng.choice(
                    candidates, size=queries_per_source, replace=False
                ).tolist()
            )
        selected_gw = sorted(selected_gw)
        selected_set = set(selected_gw)

        # Keep every possible target row for selected queries, but only one
        # representative light curve per other validation parent. This provides
        # unique-parent KN distractors without encoding the full validation split.
        source_rows = []
        selected_positive_source_rows = {}
        for gw_id in sorted(val_gw_map):
            rows = np.asarray(val_gw_map[int(gw_id)], dtype=np.int64).reshape(-1)
            if rows.size == 0:
                continue
            if int(gw_id) in selected_set:
                n_positive_rows = min(max(1, n_trials), int(rows.size))
                chosen = rng.choice(rows, size=n_positive_rows, replace=False).astype(
                    np.int64
                )
                selected_positive_source_rows[int(gw_id)] = chosen
                source_rows.extend(int(value) for value in chosen.tolist())
            else:
                source_rows.append(int(rows[rng.integers(rows.size)]))
        kn_bank, remap = load_selected_positive_bank(
            args.data_path,
            np.asarray(source_rows, dtype=np.int64),
            runtime_input_window_start=float(args.ref_start),
            runtime_input_window_end=float(args.ref_end),
        )
        positive_map = {
            int(gw_id): np.asarray(
                [
                    remap[int(value)]
                    for value in selected_positive_source_rows[int(gw_id)]
                ],
                dtype=np.int64,
            )
            for gw_id in selected_gw
        }
        kn_parent = kn_bank["gw_indices"].numpy().astype(np.int64, copy=False)

        negative_pool_size = 4 * max(gallery_sizes)
        negative_bank_raw = load_negative_optical_samples(
            args.neg_data_path,
            args.neg_group,
            n_samples=negative_pool_size,
            seed=seed,
            runtime_input_window_start=float(args.ref_start),
            runtime_input_window_end=float(args.ref_end),
        )
        if negative_bank_raw is None:
            raise ValueError("Failed to load non-KN validation candidates.")
        sequences, _gw_skymaps, _gw_times = (
            build_synthetic_time_sky_candidate_sequences(
                test_data_path=args.data_path,
                unique_gw_ids=selected_gw,
                neg_optical_data=negative_bank_raw,
                gallery_sizes=gallery_sizes,
                n_trials=n_trials,
                seed=seed,
                time_window_days=float(args.gallery_distractor_time_window_days),
                credible_level_max=float(args.validation_gallery_credible_level_max),
            )
        )
        with h5py.File(args.data_path, "r") as handle:
            event_time = np.asarray(
                handle["events/gw_data/event_time_mjd"][:], dtype=np.float64
            )
        first_detection = kn_bank["first_detection_mjd"].numpy().astype(np.float64)
        empirical_dt = first_detection - event_time[kn_parent]
        _assign_mixed_nonkn_time_deltas(
            sequences,
            empirical_dt,
            fraction=settings["nonkn_empirical_fraction"],
            window_days=float(args.gallery_distractor_time_window_days),
            seed=seed,
        )
        galleries = build_mixed_galleries(
            gw_positive_indices=positive_map,
            query_gw_ids=selected_gw,
            optical_parent_gw_idx=kn_parent,
            nonkn_candidate_sequences=sequences,
            gallery_sizes=gallery_sizes,
            n_trials=n_trials,
            seed=seed,
            kn_fraction=settings["kn_fraction"],
            kn_candidate_mode="kn_random",
        )
        digest = mixed_gallery_identity_digest(galleries)
        max_size = max(gallery_sizes)
        manifest_instances = []
        kn_source_indices = kn_bank["source_optical_indices"].numpy()
        nonkn_source_indices = np.asarray(
            negative_bank_raw["source_indices"], dtype=np.int64
        )
        for (size, trial, gw_id), spec in sorted(galleries.items()):
            if int(size) != int(max_size):
                continue
            manifest_instances.append(
                {
                    "gallery_size": int(size),
                    "trial": int(trial),
                    "gw_id": int(gw_id),
                    "source_type": source_type_map[int(gw_id)],
                    "positive_source_index": int(
                        kn_bank["source_optical_indices"][
                            int(spec["positive_index"])
                        ].item()
                    ),
                    "kn_negative_source_indices": kn_source_indices[
                        np.asarray(spec["kn_negative_indices"], dtype=np.int64)
                    ].tolist(),
                    "nonkn_negative_source_indices": nonkn_source_indices[
                        np.asarray(spec["nonkn_negative_indices"], dtype=np.int64)
                    ].tolist(),
                    "n_kn_negative": int(spec["n_kn_negative"]),
                    "n_nonkn_negative": int(spec["n_nonkn_negative"]),
                }
            )
        manifest = {
            "mode": MIXED_VALIDATION_GALLERY_MODE,
            "digest": digest,
            "gallery_sizes": list(gallery_sizes),
            "condition": settings["condition"],
            "kn_distractor_fraction": settings["kn_fraction"],
            "nonkn_empirical_fraction": settings["nonkn_empirical_fraction"],
            "instances": manifest_instances,
        }
        manifest_dir = os.path.join(args.ckpt_path, "ALBEF")
        os.makedirs(manifest_dir, exist_ok=True)
        manifest_path = os.path.join(
            manifest_dir, f"validation_gallery_manifest_{partition}.json"
        )
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)
        print(
            "Validation mixed gallery prepared: "
            f"queries={len(selected_gw)}, sizes={gallery_sizes}, "
            f"condition={settings['condition']}, "
            f"KN fraction={settings['kn_fraction']:.3f}, "
            f"digest={digest}, manifest={manifest_path}"
        )
        return cls(
            data_path=args.data_path,
            positive_bank=kn_bank,
            negative_bank_raw=negative_bank_raw,
            galleries=galleries,
            unique_gw=selected_gw,
            gw_source_map={
                int(gw_id): source_type_map[int(gw_id)] for gw_id in selected_gw
            },
            gallery_sizes=gallery_sizes,
            n_trials=n_trials,
            mrr_weight=float(args.validation_gallery_mrr_weight),
            recall_at_1_weight=float(args.validation_gallery_recall_at_1_weight),
            gallery_size_weights=getattr(args, "validation_gallery_size_weights", None),
            mode=MIXED_VALIDATION_GALLERY_MODE,
            kn_parent_gw_ids=kn_parent,
            condition=settings["condition"],
            kn_distractor_fraction=settings["kn_fraction"],
            nonkn_empirical_fraction=settings["nonkn_empirical_fraction"],
        )

    def _evaluate_mixed(self, model, args, device, amp_dtype, gw_event_time_mjd_table):
        if self.kn_parent_gw_ids is None:
            raise ValueError("mixed validation context is missing KN parent ids.")
        was_training = bool(model.training)
        model.eval()
        amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
        try:
            kn_embeddings = extract_optical_candidate_embeddings(
                model,
                self.positive_bank,
                device,
                n_ref=int(args.n_ref),
                ref_start=float(args.ref_start),
                ref_end=float(args.ref_end),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
                desc="  Validation mixed KN embeddings",
            )
            nonkn_embeddings = extract_negative_gallery_embeddings(
                model,
                self.negative_bank_raw,
                device,
                n_ref=int(args.n_ref),
                ref_start=float(args.ref_start),
                ref_end=float(args.ref_end),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            query_cache = _build_gallery_query_cache(
                model,
                self.unique_gw,
                self.data_path,
                device,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            dual = _is_dual_fusion_model(model)
            if isinstance(gw_event_time_mjd_table, torch.Tensor):
                event_time = gw_event_time_mjd_table.detach().float().cpu().numpy()
            else:
                event_time = np.asarray(gw_event_time_mjd_table, dtype=np.float64)
            first_detection = (
                self.positive_bank["first_detection_mjd"].numpy().astype(np.float64)
            )
            parent = np.asarray(self.kn_parent_gw_ids, dtype=np.int64)
            max_size = max(self.gallery_sizes)
            outcomes = {}
            for trial in range(self.n_trials):
                for gw_id in self.unique_gw:
                    spec = self.galleries[(max_size, trial, int(gw_id))]
                    pos_idx = int(spec["positive_index"])
                    kn_idx = np.asarray(spec["kn_negative_indices"], dtype=np.int64)
                    nonkn_idx = np.asarray(
                        spec["nonkn_negative_indices"], dtype=np.int64
                    )
                    positive_coord = (
                        self.positive_bank["coordinates"][pos_idx]
                        .numpy()
                        .astype(np.float32)
                    )
                    pos_dt = np.asarray(
                        [first_detection[pos_idx] - event_time[int(gw_id)]],
                        dtype=np.float32,
                    )
                    kn_dt = (
                        first_detection[kn_idx] - event_time[parent[kn_idx]]
                    ).astype(np.float32)
                    nonkn_dt = np.asarray(
                        spec["nonkn_training_aligned_dt_days"], dtype=np.float32
                    )
                    coordinates = mixed_validation_candidate_coordinates(
                        self.condition,
                        positive_coord,
                        spec["nonkn_synthetic_coordinates"],
                        n_kn=kn_idx.size,
                        n_nonkn=nonkn_idx.size,
                    )
                    pos_score = _score_candidate_bank_with_logits(
                        model,
                        query_cache[int(gw_id)],
                        np.asarray([pos_idx], dtype=np.int64),
                        kn_embeddings,
                        device,
                        dual,
                        candidate_coords=coordinates["positive"],
                        candidate_abs_dt_days=pos_dt,
                        amp_dtype=amp_dtype,
                        amp_enabled=amp_enabled,
                    )
                    kn_scores = _score_candidate_bank_with_logits(
                        model,
                        query_cache[int(gw_id)],
                        kn_idx,
                        kn_embeddings,
                        device,
                        dual,
                        candidate_coords=coordinates["kn"],
                        candidate_abs_dt_days=kn_dt,
                        amp_dtype=amp_dtype,
                        amp_enabled=amp_enabled,
                    )
                    nonkn_scores = _score_candidate_bank_with_logits(
                        model,
                        query_cache[int(gw_id)],
                        nonkn_idx,
                        nonkn_embeddings,
                        device,
                        dual,
                        candidate_coords=coordinates["nonkn"],
                        candidate_abs_dt_days=nonkn_dt,
                        amp_dtype=amp_dtype,
                        amp_enabled=amp_enabled,
                    )
                    for size in self.gallery_sizes:
                        n_kn, n_nonkn = allocate_mixed_negative_counts(
                            size, self.kn_distractor_fraction
                        )
                        outcomes[(int(size), trial, int(gw_id))] = mixed_score_outcome(
                            float(pos_score[0]),
                            kn_scores[:n_kn],
                            nonkn_scores[:n_nonkn],
                        )
            summary = _aggregate_mixed_validation_outcomes(
                outcomes,
                gallery_sizes=self.gallery_sizes,
                n_trials=self.n_trials,
                unique_gw=self.unique_gw,
                gw_source_map=self.gw_source_map,
                mrr_weight=self.mrr_weight,
                recall_at_1_weight=self.recall_at_1_weight,
                gallery_size_weights=self.gallery_size_weights,
            )
            summary.update(
                condition=self.condition,
                kn_distractor_fraction=self.kn_distractor_fraction,
                nonkn_empirical_fraction=self.nonkn_empirical_fraction,
            )
            return summary
        finally:
            if was_training:
                model.train()

    def evaluate(self, model, args, device, amp_dtype, gw_event_time_mjd_table):
        if self.mode == MIXED_VALIDATION_GALLERY_MODE:
            return self._evaluate_mixed(
                model, args, device, amp_dtype, gw_event_time_mjd_table
            )
        was_training = bool(model.training)
        model.eval()
        amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
        try:
            positive_embeddings = extract_optical_candidate_embeddings(
                model,
                self.positive_bank,
                device,
                n_ref=int(args.n_ref),
                ref_start=float(args.ref_start),
                ref_end=float(args.ref_end),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
                desc="  Validation positive embeddings",
            )
            negative_embeddings = extract_negative_gallery_embeddings(
                model,
                self.negative_bank_raw,
                device,
                n_ref=int(args.n_ref),
                ref_start=float(args.ref_start),
                ref_end=float(args.ref_end),
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            use_fusion = (
                float(getattr(args, "cls_weight", 0.0)) > 0.0
                or float(getattr(args, "gallery_loss_weight", 0.0)) > 0.0
            )
            score_fn = (
                score_all_galleries_multimodal
                if use_fusion
                else score_all_galleries_contrastive
            )
            ranks = score_fn(
                model,
                positive_embeddings,
                negative_embeddings,
                self.galleries,
                self.unique_gw,
                test_data_path=self.data_path,
                device=device,
                model_args=vars(args),
                gw_event_time_mjd_table=gw_event_time_mjd_table,
                amp_dtype=amp_dtype,
                amp_enabled=amp_enabled,
            )
            outcomes = enrich_gallery_outcomes(ranks, self.galleries)
            overall, by_source, coverage = aggregate_gallery_outcomes(
                outcomes=outcomes,
                gallery_sizes=self.gallery_sizes,
                n_trials=self.n_trials,
                unique_gw=self.unique_gw,
                gw_source_map=self.gw_source_map,
            )
            source_macro, source_gap = compute_source_macro_and_gap(by_source)
            summary = compute_hard_gallery_selection_summary(
                source_macro,
                self.gallery_sizes,
                mrr_weight=self.mrr_weight,
                recall_at_1_weight=self.recall_at_1_weight,
                gallery_size_weights=self.gallery_size_weights,
            )
            return {
                "overall": overall,
                "by_source": by_source,
                "source_macro": source_macro,
                "source_gap": source_gap,
                "coverage": coverage,
                **summary,
            }
        finally:
            if was_training:
                model.train()
