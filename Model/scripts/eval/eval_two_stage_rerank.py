#!/usr/bin/env python3
"""Two-stage retrieval reranking evaluation.

Stage 1 scores a full gallery with a first multimodal model, keeps the top-K
candidates, then Stage 2 reranks only those candidates with a second model.
This script is intentionally separate from eval_retrieval_comparison.py so the
standard comparison workflow stays unchanged.
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts.eval import eval_retrieval_comparison as cmp  # noqa: E402
from retrieval_gallery import (  # noqa: E402
    build_prefixed_gallery_specs,
    build_synthetic_time_sky_candidate_sequences,
    build_time_sky_candidate_sequences,
)
from scripts.eval.evaluate import (  # noqa: E402
    _build_gallery_query_cache,
    _is_dual_fusion_model,
    _resolve_eval_amp,
    load_gw_event_time_mjd_table,
    load_gw_source_types,
    load_negative_gw_indices,
    load_negative_optical_samples,
)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _model_spec(raw: Mapping[str, Any], cfg_dir: Path, *, label: str) -> Dict[str, Any]:
    name = str(raw.get("name", label))
    checkpoint = cmp._resolve_path(cfg_dir, raw.get("checkpoint"))
    if not checkpoint:
        raise ValueError(f"{label} is missing checkpoint")
    config = cmp._resolve_path(cfg_dir, raw.get("config"))
    resolved_checkpoint = cmp._resolve_checkpoint_path(checkpoint, "multimodal")
    return {
        "name": name,
        "type": "multimodal",
        "checkpoint": checkpoint,
        "config": config,
        "resolved_checkpoint": resolved_checkpoint,
        "resolved_config": config,
        "scoring": str(raw.get("scoring", "logits")),
    }


def _metrics_from_ranks(ranks: Sequence[Optional[int]], ks: Sequence[int] = (1, 10)) -> Dict[str, float]:
    n = len(ranks)
    out: Dict[str, float] = {"n": int(n)}
    if n == 0:
        for k in ks:
            out[f"recall_at_{int(k)}"] = 0.0
        out["mrr"] = 0.0
        return out
    finite = [int(r) for r in ranks if r is not None and int(r) >= 0]
    for k in ks:
        out[f"recall_at_{int(k)}"] = float(sum(1 for r in finite if r < int(k)) / n)
    out["mrr"] = float(sum(1.0 / (r + 1.0) for r in finite) / n)
    return out


def _group_metrics(rows: Sequence[Mapping[str, Any]], rank_key: str) -> Dict[str, Any]:
    overall = _metrics_from_ranks([row.get(rank_key) for row in rows])
    by_source: Dict[str, Dict[str, float]] = {}
    sources = sorted({str(row.get("source", "unknown")) for row in rows})
    for source in sources:
        source_rows = [row for row in rows if str(row.get("source", "unknown")) == source]
        by_source[source] = _metrics_from_ranks([row.get(rank_key) for row in source_rows])
    return {"overall": overall, "by_source": by_source}


def _positive_abs_dt_days(
    positive_bank: Mapping[str, torch.Tensor],
    pos_index: int,
    gw_event_time_mjd: Optional[float],
) -> float:
    if (
        "first_detection_mjd" in positive_bank
        and gw_event_time_mjd is not None
        and np.isfinite(float(gw_event_time_mjd))
    ):
        return abs(float(positive_bank["first_detection_mjd"][int(pos_index)].item()) - float(gw_event_time_mjd))
    return 0.0


@torch.no_grad()
def _score_gallery_logits(
    model,
    query_cache: Mapping[int, Mapping[str, torch.Tensor]],
    positive_bank: Mapping[str, torch.Tensor],
    negative_bank: Mapping[str, torch.Tensor],
    gallery_spec: Mapping[str, Any],
    gw_id: int,
    *,
    device: torch.device,
    dual: bool,
    gw_event_time_mjd_table=None,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> np.ndarray:
    pos_index = int(gallery_spec["positive_index"])
    neg_indices = np.asarray(gallery_spec["negative_indices"], dtype=np.int64)
    gw_event_time_mjd = cmp._lookup_event_time_mjd(gw_event_time_mjd_table, int(gw_id))
    pos_dt = _positive_abs_dt_days(positive_bank, pos_index, gw_event_time_mjd)
    neg_abs_dt_days = cmp.extract_gallery_negative_abs_dt_days(
        gallery_spec,
        gw_event_time_mjd=gw_event_time_mjd,
        n_negative=int(neg_indices.shape[0]),
    )
    pos_scores = cmp._score_candidate_bank_with_logits(
        model,
        query_cache[int(gw_id)],
        np.asarray([pos_index], dtype=np.int64),
        positive_bank,
        device,
        dual,
        candidate_abs_dt_days=np.asarray([pos_dt], dtype=np.float32),
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    neg_scores = cmp._score_candidate_bank_with_logits(
        model,
        query_cache[int(gw_id)],
        neg_indices,
        negative_bank,
        device,
        dual,
        candidate_coords=gallery_spec.get("negative_synthetic_coordinates"),
        candidate_abs_dt_days=neg_abs_dt_days,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    return np.concatenate([pos_scores, neg_scores], axis=0)


def _rank_positive(scores: np.ndarray) -> int:
    ranked = np.argsort(-np.asarray(scores, dtype=np.float64))
    return int(np.where(ranked == 0)[0][0])


@torch.no_grad()
def _score_gallery_positions_logits(
    model,
    query_cache: Mapping[int, Mapping[str, torch.Tensor]],
    positive_bank: Mapping[str, torch.Tensor],
    negative_bank: Mapping[str, torch.Tensor],
    gallery_spec: Mapping[str, Any],
    gw_id: int,
    positions: Sequence[int],
    *,
    device: torch.device,
    dual: bool,
    gw_event_time_mjd_table=None,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> np.ndarray:
    """Score selected gallery positions, where position 0 is the positive."""
    positions_np = np.asarray(positions, dtype=np.int64)
    scores = np.empty((positions_np.shape[0],), dtype=np.float32)
    if positions_np.size == 0:
        return scores

    gw_event_time_mjd = cmp._lookup_event_time_mjd(gw_event_time_mjd_table, int(gw_id))
    pos_mask = positions_np == 0
    if np.any(pos_mask):
        pos_index = int(gallery_spec["positive_index"])
        pos_dt = _positive_abs_dt_days(positive_bank, pos_index, gw_event_time_mjd)
        scores[pos_mask] = cmp._score_candidate_bank_with_logits(
            model,
            query_cache[int(gw_id)],
            np.asarray([pos_index], dtype=np.int64),
            positive_bank,
            device,
            dual,
            candidate_abs_dt_days=np.asarray([pos_dt], dtype=np.float32),
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )[0]

    neg_mask = ~pos_mask
    if np.any(neg_mask):
        neg_positions = positions_np[neg_mask] - 1
        all_neg_indices = np.asarray(gallery_spec["negative_indices"], dtype=np.int64)
        neg_indices = all_neg_indices[neg_positions]
        neg_coords = gallery_spec.get("negative_synthetic_coordinates")
        if neg_coords is not None:
            neg_coords = np.asarray(neg_coords, dtype=np.float32)[neg_positions]
        all_neg_abs_dt_days = cmp.extract_gallery_negative_abs_dt_days(
            gallery_spec,
            gw_event_time_mjd=gw_event_time_mjd,
            n_negative=int(all_neg_indices.shape[0]),
        )
        neg_abs_dt_days = None
        if all_neg_abs_dt_days is not None:
            neg_abs_dt_days = np.asarray(all_neg_abs_dt_days, dtype=np.float32)[neg_positions]
        scores[neg_mask] = cmp._score_candidate_bank_with_logits(
            model,
            query_cache[int(gw_id)],
            neg_indices,
            negative_bank,
            device,
            dual,
            candidate_coords=neg_coords,
            candidate_abs_dt_days=neg_abs_dt_days,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )

    return scores


def _build_shared_galleries(raw_cfg: Dict[str, Any], cfg: Dict[str, Any], seed: int):
    gallery_sizes = list(cfg["gallery_sizes"])
    n_trials = int(cfg["gallery_trials"])
    comparison_window = (float(cfg["comparison_window_start"]), float(cfg["comparison_window_end"]))
    neg_target_samples = cmp._resolve_target_samples(cfg["n_neg_samples"])
    neg_optical_data = load_negative_optical_samples(
        cfg["neg_data_path"],
        cfg["neg_group"],
        n_samples=neg_target_samples,
        seed=seed,
        require_zero_time_mjd_base=False,
        require_zero_time_mjd_cls_base=False,
        nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
        runtime_input_window_start=comparison_window[0],
        runtime_input_window_end=comparison_window[1],
        negative_sample_strategy=cfg.get("negative_sample_strategy", "block_random"),
        negative_sample_block_rows=cfg.get("negative_sample_block_rows", None),
        negative_sample_shuffle=cfg.get("negative_sample_shuffle", True),
    )
    gw_positive_indices, all_test_gw_ids = cmp.load_test_positive_index_map(cfg["test_data_path"])
    if int(cfg["max_gw_events"]) > 0 and len(all_test_gw_ids) > int(cfg["max_gw_events"]):
        rng = np.random.default_rng(seed)
        selected_gw = sorted(
            int(v)
            for v in rng.choice(
                np.asarray(all_test_gw_ids, dtype=np.int64),
                size=int(cfg["max_gw_events"]),
                replace=False,
            ).tolist()
        )
        gw_positive_indices = {int(gw_id): gw_positive_indices[int(gw_id)] for gw_id in selected_gw}
        all_test_gw_ids = selected_gw

    gallery_candidate_mode = str(cfg["gallery_candidate_mode"]).strip().lower()
    if gallery_candidate_mode == "time_sky_hard":
        candidate_sequences, _gallery_gw_skymaps, _gallery_gw_times = build_time_sky_candidate_sequences(
            test_data_path=cfg["test_data_path"],
            unique_gw_ids=all_test_gw_ids,
            neg_optical_data=neg_optical_data,
            n_trials=n_trials,
            seed=seed,
            time_window_days=cfg["gallery_candidate_time_window_days"],
            credible_level_max=cfg["gallery_candidate_credible_level_max"],
            zero_time_field=cfg["nonkn_cls_base_field"],
        )
    elif gallery_candidate_mode == "synthetic_time_sky_hard":
        candidate_sequences, _gallery_gw_skymaps, _gallery_gw_times = build_synthetic_time_sky_candidate_sequences(
            test_data_path=cfg["test_data_path"],
            unique_gw_ids=all_test_gw_ids,
            neg_optical_data=neg_optical_data,
            gallery_sizes=gallery_sizes,
            n_trials=n_trials,
            seed=seed,
            time_window_days=cfg["gallery_candidate_time_window_days"],
            credible_level_max=cfg["gallery_candidate_credible_level_max"],
        )
    else:
        raise ValueError("two-stage rerank currently supports time_sky_hard and synthetic_time_sky_hard only")

    galleries, unique_gw = build_prefixed_gallery_specs(
        gw_positive_indices=gw_positive_indices,
        candidate_sequences=candidate_sequences,
        gallery_sizes=gallery_sizes,
        n_trials=n_trials,
        seed=seed,
        include_undersized=bool(cfg["gallery_include_undersized"]),
    )
    selected_positive_indices = np.unique(
        np.asarray([int(spec["positive_index"]) for spec in galleries.values()], dtype=np.int64)
    )
    positive_query_bank, positive_index_remap = cmp.load_selected_positive_bank(
        cfg["test_data_path"],
        selected_positive_indices,
        runtime_input_window_start=comparison_window[0],
        runtime_input_window_end=comparison_window[1],
    )
    galleries = cmp.remap_gallery_positive_indices(galleries, positive_index_remap)
    return neg_optical_data, positive_query_bank, galleries, unique_gw, all_test_gw_ids


def _load_eval_model(
    spec: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    device: torch.device,
    comparison_window: Tuple[float, float],
):
    model, runtime_args, saved_args = cmp.load_multimodal_bundle(
        str(spec["resolved_checkpoint"]),
        str(spec.get("resolved_config") or spec.get("config") or ""),
        device,
        test_data_path=cfg["test_data_path"],
        neg_data_path=cfg["neg_data_path"],
        comparison_window=comparison_window,
        nonkn_cls_base_field=cfg["nonkn_cls_base_field"],
    )
    return model, runtime_args, saved_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage no-hard -> hard-mining rerank evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to two-stage rerank JSON config")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    raw_cfg = _load_json(config_path)
    cfg_dir = config_path.parent

    shared_raw = dict(raw_cfg)
    shared_raw["models"] = []
    if "gallery_size" in shared_raw:
        shared_raw["gallery_sizes"] = str(int(shared_raw["gallery_size"]))
    cfg = cmp.normalize_shared_config(shared_raw, config_path)
    cfg["gallery_sizes"] = [int(raw_cfg.get("gallery_size", cfg["gallery_sizes"][0]))]
    rerank_topk = int(raw_cfg.get("rerank_topk", 64))
    if rerank_topk < 1:
        raise ValueError("rerank_topk must be >= 1")

    stage1_spec = _model_spec(raw_cfg["stage1_model"], cfg_dir, label="stage1_model")
    stage2_spec = _model_spec(raw_cfg["stage2_model"], cfg_dir, label="stage2_model")

    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
    amp_dtype, amp_enabled = _resolve_eval_amp(cfg["amp_dtype"], device)
    comparison_window = (float(cfg["comparison_window_start"]), float(cfg["comparison_window_end"]))
    seed = int(cfg["seed"])

    print("=" * 60)
    print("Two-Stage Retrieval Rerank")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Gallery size: {cfg['gallery_sizes'][0]}")
    print(f"Rerank top-K: {rerank_topk}")
    print(f"Stage 1: {stage1_spec['name']} -> {stage1_spec['resolved_checkpoint']}")
    print(f"Stage 2: {stage2_spec['name']} -> {stage2_spec['resolved_checkpoint']}")
    print(f"Output dir: {cfg['output_dir']}")

    cmp._seed_all(seed)
    gw_source_types = load_gw_source_types(cfg["test_data_path"])
    gw_source_map = cmp.build_gw_source_map(gw_source_types, [])
    gw_event_time_mjd_table = load_gw_event_time_mjd_table(cfg["test_data_path"], device, required=False)
    load_negative_gw_indices(cfg["test_data_path"])

    print("\nPhase 1: Build shared G=1000 galleries")
    neg_optical_data, positive_query_bank_raw, galleries, unique_gw, all_test_gw_ids = _build_shared_galleries(
        raw_cfg, cfg, seed
    )
    gw_source_map = cmp.build_gw_source_map(gw_source_types, all_test_gw_ids)
    if not galleries:
        raise ValueError("No galleries were generated.")
    print(f"  Built {len(galleries)} galleries over {len(unique_gw)} GW events")

    print("\nPhase 2: Stage 1 full-gallery scoring with no-hard-mining model")
    stage1_model, stage1_args, stage1_saved = _load_eval_model(
        stage1_spec,
        cfg,
        device=device,
        comparison_window=comparison_window,
    )
    stage1_positive_bank = cmp.extract_optical_candidate_embeddings(
        stage1_model,
        positive_query_bank_raw,
        device,
        n_ref=int(stage1_args.get("n_ref", 64)),
        ref_start=float(stage1_args["ref_start"]),
        ref_end=float(stage1_args["ref_end"]),
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
        desc="  Stage 1 positive embeddings",
    )
    stage1_negative_bank = cmp.extract_negative_gallery_embeddings(
        stage1_model,
        neg_optical_data,
        device,
        n_ref=int(stage1_args.get("n_ref", 64)),
        ref_start=float(stage1_args["ref_start"]),
        ref_end=float(stage1_args["ref_end"]),
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    stage1_dual = bool(stage1_positive_bank.get("dual_fusion", _is_dual_fusion_model(stage1_model)))
    print("  Stage 1 GW query cache")
    stage1_query_cache = _build_gallery_query_cache(
        stage1_model,
        unique_gw,
        cfg["test_data_path"],
        device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )

    rows: List[Dict[str, Any]] = []
    kept_by_key: Dict[Tuple[int, int, int], np.ndarray] = {}
    for key, gallery_spec in tqdm(galleries.items(), desc="  Stage 1 full-gallery scoring"):
        gallery_size, trial, gw_id = key
        stage1_scores = _score_gallery_logits(
            stage1_model,
            stage1_query_cache,
            stage1_positive_bank,
            stage1_negative_bank,
            gallery_spec,
            int(gw_id),
            device=device,
            dual=stage1_dual,
            gw_event_time_mjd_table=gw_event_time_mjd_table,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        stage1_rank = _rank_positive(stage1_scores)
        stage1_order = np.argsort(-stage1_scores)
        keep_n = min(int(rerank_topk), int(stage1_order.shape[0]))
        kept = stage1_order[:keep_n].astype(np.int64, copy=False)
        kept_by_key[key] = kept
        rows.append(
            {
                "gallery_size": int(gallery_size),
                "trial": int(trial),
                "gw_id": int(gw_id),
                "source": str(gw_source_map.get(int(gw_id), "unknown")),
                "actual_gallery_size": int(gallery_spec.get("actual_gallery_size", len(stage1_scores))),
                "stage1_rank": int(stage1_rank),
                "two_stage_rank": None,
                "stage1_topk_kept_positive": bool(np.any(kept == 0)),
            }
        )

    del stage1_model, stage1_positive_bank, stage1_negative_bank, stage1_query_cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\nPhase 3: Stage 2 rerank only Stage 1 top-K candidates")
    stage2_model, stage2_args, stage2_saved = _load_eval_model(
        stage2_spec,
        cfg,
        device=device,
        comparison_window=comparison_window,
    )
    stage2_positive_bank = cmp.extract_optical_candidate_embeddings(
        stage2_model,
        positive_query_bank_raw,
        device,
        n_ref=int(stage2_args.get("n_ref", 64)),
        ref_start=float(stage2_args["ref_start"]),
        ref_end=float(stage2_args["ref_end"]),
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
        desc="  Stage 2 positive embeddings",
    )
    stage2_negative_bank = cmp.extract_negative_gallery_embeddings(
        stage2_model,
        neg_optical_data,
        device,
        n_ref=int(stage2_args.get("n_ref", 64)),
        ref_start=float(stage2_args["ref_start"]),
        ref_end=float(stage2_args["ref_end"]),
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )
    stage2_dual = bool(stage2_positive_bank.get("dual_fusion", _is_dual_fusion_model(stage2_model)))
    print("  Stage 2 GW query cache")
    stage2_query_cache = _build_gallery_query_cache(
        stage2_model,
        unique_gw,
        cfg["test_data_path"],
        device,
        amp_dtype=amp_dtype,
        amp_enabled=amp_enabled,
    )

    row_by_key = { (int(row["gallery_size"]), int(row["trial"]), int(row["gw_id"])): row for row in rows }
    for key, gallery_spec in tqdm(galleries.items(), desc="  Stage 2 top-K reranking"):
        _gallery_size, _trial, gw_id = key
        row = row_by_key[key]
        kept = kept_by_key[key]
        if not row["stage1_topk_kept_positive"]:
            continue
        stage2_kept_scores = _score_gallery_positions_logits(
            stage2_model,
            stage2_query_cache,
            stage2_positive_bank,
            stage2_negative_bank,
            gallery_spec,
            int(gw_id),
            kept,
            device=device,
            dual=stage2_dual,
            gw_event_time_mjd_table=gw_event_time_mjd_table,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        rerank_order_local = np.argsort(-stage2_kept_scores)
        reranked_positions = kept[rerank_order_local]
        row["two_stage_rank"] = int(np.where(reranked_positions == 0)[0][0])

    stage1_metrics = _group_metrics(rows, "stage1_rank")
    two_stage_metrics = _group_metrics(rows, "two_stage_rank")
    two_stage_metrics["overall"]["stage1_topk_recall"] = float(
        sum(1 for row in rows if row["stage1_topk_kept_positive"]) / len(rows)
    ) if rows else 0.0
    for source, source_metrics in two_stage_metrics["by_source"].items():
        source_rows = [row for row in rows if str(row.get("source", "unknown")) == source]
        source_metrics["stage1_topk_recall"] = float(
            sum(1 for row in source_rows if row["stage1_topk_kept_positive"]) / len(source_rows)
        ) if source_rows else 0.0

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            "input_config": str(config_path),
            "test_data_path": cfg["test_data_path"],
            "neg_data_path": cfg["neg_data_path"],
            "neg_group": cfg["neg_group"],
            "gallery_size": int(cfg["gallery_sizes"][0]),
            "gallery_trials": int(cfg["gallery_trials"]),
            "gallery_candidate_mode": cfg["gallery_candidate_mode"],
            "gallery_candidate_time_window_days": cfg["gallery_candidate_time_window_days"],
            "gallery_candidate_credible_level_max": cfg["gallery_candidate_credible_level_max"],
            "rerank_topk": int(rerank_topk),
            "seed": int(seed),
            "amp_dtype": cfg["amp_dtype"],
        },
        "stage1_model": {
            **stage1_spec,
            "saved_args_gallery_hard_neg_enable": bool(stage1_saved.get("gallery_hard_neg_enable", False)),
        },
        "stage2_model": {
            **stage2_spec,
            "saved_args_gallery_hard_neg_enable": bool(stage2_saved.get("gallery_hard_neg_enable", False)),
        },
        "metrics": {
            "stage1_full_gallery": stage1_metrics,
            "two_stage_topk_rerank": two_stage_metrics,
        },
        "rows": rows,
    }
    output_path = output_dir / "two_stage_rerank_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(output), f, indent=2)

    print("\nResults")
    for label, metrics in (
        ("Stage 1 full gallery", stage1_metrics["overall"]),
        ("Two-stage top-K rerank", two_stage_metrics["overall"]),
    ):
        print(
            f"  {label}: "
            f"R@1={metrics['recall_at_1']:.4f} "
            f"R@10={metrics['recall_at_10']:.4f} "
            f"MRR={metrics['mrr']:.4f}"
        )
    print(f"  Stage1 top{rerank_topk} recall={two_stage_metrics['overall']['stage1_topk_recall']:.4f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
