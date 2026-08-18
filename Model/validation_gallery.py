from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from data_loader import load_gw_source_type_map
from retrieval_gallery import (
    aggregate_gallery_outcomes,
    build_prefixed_gallery_specs,
    build_synthetic_time_sky_candidate_sequences,
    compute_source_macro_and_gap,
)
from scripts.eval.eval_retrieval_comparison import (
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

    @classmethod
    def build(cls, args, val_gw_map: Mapping[int, Sequence[int]]):
        if (
            str(args.validation_gallery_mode).strip().lower()
            != SUPPORTED_VALIDATION_GALLERY_MODE
        ):
            raise ValueError(
                f"validation_gallery_mode must be '{SUPPORTED_VALIDATION_GALLERY_MODE}'."
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

    def evaluate(self, model, args, device, amp_dtype, gw_event_time_mjd_table):
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
