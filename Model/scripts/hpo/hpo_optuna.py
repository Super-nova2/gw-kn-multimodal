#!/usr/bin/env python3
"""
Config-driven Optuna hyperparameter optimization for MAGIKS GW-Optical training.

Usage:
    python Model/scripts/hpo/hpo_optuna.py --config Model/args/hpo/hpo_v5_fusion.json
    python Model/scripts/hpo/hpo_optuna.py --config Model/args/hpo/hpo_v5_fusion.json --dry_run
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from typing import Any, Dict, Tuple

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REPO_ROOT = os.path.dirname(MODEL_DIR)
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
TRAIN_SCRIPT = os.path.join(MODEL_DIR, "scripts", "train", "train.py")
DEFAULT_BASE_TRAIN_CONFIG = os.path.join(MODEL_DIR, "args", "MAGIKS_BNS_NSBH_full.json")

MAGIKS_BOOL_KEYS = {
    "cache_in_memory",
    "enable_ood_monitoring",
    "mask_itc",
    "semi_hard",
    "hardneg_memory_bank_enable",
    "use_lightweight_gw",
    "dual_fusion",
    "use_time_delta_cls_feature",
    "use_similarity_as_cls_input",
    "use_cred_level_feature",
    "compute_cls_metrics",
    "compute_fusion_gallery_metrics",
    "gallery_include_extra_negatives",
    "gallery_hard_neg_enable",
    "skip_epoch_checkpoints",
}

MAGIKS_ARG_KEYS = {
    "data_path",
    "test_data_path",
    "test_steps",
    "enable_ood_monitoring",
    "neg_data_path",
    "neg_group",
    "epochs",
    "batch_size",
    "steps_per_epoch",
    "ckpt_path",
    "resume",
    "lr",
    "weight_decay",
    "grad_clip_norm",
    "lr_scheduler",
    "warmup_epochs",
    "min_lr",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "prefetch_factor",
    "cache_in_memory",
    "val_split",
    "val_batch_size",
    "val_steps_per_epoch",
    "split_seed",
    "seed",
    "early_stop_patience",
    "early_stop_min_delta",
    "best_ckpt_metric",
    "n_ref",
    "ref_start",
    "ref_end",
    "ref_dim",
    "enc_dim",
    "proj_dim",
    "optical_curve_dim",
    "optical_coord_dim",
    "optical_curve_hidden_dim",
    "contrastive_hidden_dim",
    "fusion_attn_dim",
    "fusion_hidden_dim",
    "fusion_dropout",
    "label_smoothing",
    "temp_init",
    "temp_final",
    "temp_min",
    "temp_max",
    "temp_schedule",
    "time_compat_weight",
    "time_compat_tau_days",
    "time_compat_power",
    "time_compat_max_penalty",
    "use_time_delta_cls_feature",
    "time_delta_cls_scale_days",
    "time_delta_cls_clip",
    "fusion_physical_weight",
    "fusion_spatial_weight",
    "mtan_snr_s0",
    "mtan_snr_beta",
    "mtan_snr_clip_min",
    "mtan_snr_clip_max",
    "mtan_snr_eps",
    "mtan_lupt_psfflux_zp",
    "mtan_lupt_k",
    "mtan_lupt_m5_mag",
    "nonkn_cls_base_field",
    "gw_dropout",
    "opt_dropout",
    "proj_dropout",
    "feature_dropout",
    "itc_weight",
    "itc_extra_negative_enable",
    "cls_weight",
    "cls_pos_weight",
    "cls_neg_weight",
    "cls_extra_neg_weight",
    "cls_aligned_pos_weight",
    "cls_neg_gw_weight",
    "cls_mismatched_weight",
    "cls_external_neg_weight",
    "cls_ramp_epochs",
    "staged_training_enable",
    "stage_itc_epochs",
    "stage_cls_ramp_epochs",
    "stage_retrieval_ramp_epochs",
    "stage_joint_itc_start_weight",
    "stage_joint_itc_end_weight",
    "encoder_lr_ratio",
    "neg_gw_guardrail_enable",
    "neg_gw_guardrail_recall",
    "retrieval_start_epoch",
    "gallery_loss_weight",
    "gallery_loss_ramp_epochs",
    "compute_cls_metrics",
    "compute_fusion_gallery_metrics",
    "fusion_gallery_metrics_start_epoch",
    "val_split_stratify_by_source",
    "validation_gallery_enable",
    "validation_gallery_mode",
    "validation_gallery_sizes",
    "validation_gallery_queries_per_source",
    "validation_gallery_trials",
    "validation_gallery_seed",
    "validation_gallery_time_window_days",
    "validation_gallery_credible_level_max",
    "validation_gallery_include_undersized",
    "validation_gallery_mrr_weight",
    "validation_gallery_recall_at_1_weight",
    "gallery_score_chunk_size",
    "max_gallery_queries",
    "gallery_include_extra_negatives",
    "gallery_distractor_time_mode",
    "gallery_distractor_time_window_days",
    "gallery_hard_neg_enable",
    "gallery_hard_neg_topk",
    "gallery_hard_neg_weight",
    "gallery_hard_neg_start_after_retrieval_epochs",
    "gallery_hard_neg_ramp_epochs",
    "itc_decay_start_epoch",
    "itc_decay_epochs",
    "itc_decay_ratio",
    "itc_label_smoothing",
    "itc_loss_type",
    "supcon_margin",
    "samples_per_gw",
    "min_lc_per_gw",
    "neg_gw_pair_ratio",
    "mis_neg_dt_window_days",
    "mask_itc",
    "hard_neg_start_epoch",
    "hard_neg_ramp_epochs",
    "semi_hard",
    "semi_hard_margin",
    "hardneg_time_window_days",
    "hardneg_min_candidates",
    "hardneg_fallback_mode",
    "hardneg_memory_bank_enable",
    "hardneg_memory_bank_size",
    "hardneg_memory_topk",
    "hardneg_memory_warmup_steps",
    "hardneg_memory_interval",
    "hardneg_memory_max_rows",
    "cls_start_epoch",
    "use_lightweight_gw",
    "dual_fusion",
    "fusion_mode",
    "use_similarity_as_cls_input",
    "use_cred_level_feature",
    "gw_aug_noise",
    "gw_aug_jitter",
    "gw_aug_dropout",
    "opt_aug_noise",
    "opt_aug_time_jitter",
    "opt_aug_dropout",
    "opt_aug_band_dropout",
    "hpo_trial_number",
    "skip_epoch_checkpoints",
}

STALE_KEYS = {
    "supcon_temperature",
    "stage_to_jobfs",
    "val_data_path",
    "ood_val_steps",
    "pretrained",
    "freeze_encoder_epochs",
    "freeze_itc_epochs",
    "use_neg_gw",
    "neg_gw_ratio",
    "hard_neg_top_k",
    "gallery_force_include_positives",
    "gallery_detach_encoder_inputs",
    "fusion_rerank_topk",
    "fusion_rerank_lambda",
    "_hardneg_window_days",
    "_dataset_window_metadata",
    "_effective_input_window_metadata",
}

DEFAULT_TUNABLE_PARAMS = [
    "lr",
    "weight_decay",
    "warmup_epochs",
    "cls_start_epoch",
    "cls_ramp_epochs",
    "retrieval_start_after_cls_epochs",
    "gallery_loss_ramp_epochs",
    "gallery_hard_neg_start_after_retrieval_epochs",
    "gallery_hard_neg_ramp_epochs",
    "gallery_hard_neg_weight",
    "gallery_hard_neg_topk",
    "time_compat_weight",
]

DERIVED_PARAM_KEYS = {
    "ref_shared_dim",
    "augment_enable",
    "retrieval_start_after_cls_epochs",
    "balanced_model_capacity",
}

BALANCED_MODEL_CAPACITY_PRESETS = {
    "baseline": {
        "enc_dim": 128,
        "proj_dim": 128,
        "optical_curve_dim": 192,
        "optical_coord_dim": 64,
        "optical_curve_hidden_dim": 256,
        "contrastive_hidden_dim": 128,
        "fusion_attn_dim": 128,
        "fusion_hidden_dim": 256,
    },
    "wide_balanced": {
        "enc_dim": 160,
        "proj_dim": 160,
        "optical_curve_dim": 216,
        "optical_coord_dim": 72,
        "optical_curve_hidden_dim": 288,
        "contrastive_hidden_dim": 160,
        "fusion_attn_dim": 144,
        "fusion_hidden_dim": 288,
    },
}
BALANCED_MODEL_CAPACITY_KEYS = frozenset(
    next(iter(BALANCED_MODEL_CAPACITY_PRESETS.values())).keys()
)

AUGMENT_PRESET_KEYS = (
    "gw_aug_noise",
    "gw_aug_jitter",
    "gw_aug_dropout",
    "opt_aug_noise",
    "opt_aug_time_jitter",
    "opt_aug_dropout",
    "opt_aug_band_dropout",
)

OBJECTIVE_PRESETS = {
    # Historical objective used in earlier experiments.
    "combined_auroc_g2o_r5": {"val_auroc": 0.5, "val_recall_at_5": 0.5},
    # Pure classification objective.
    "cls_auroc_auprc": {"val_auroc": 0.5, "val_auprc": 0.5},
    # Classification-priority objective with retrieval as a soft guard.
    "cls_priority_auprc_auroc_r5": {
        "val_auprc": 0.45,
        "val_auroc": 0.35,
        "val_recall_at_5": 0.20,
    },
    # Fusion-gallery retrieval priority for current three-stage training.
    "fusion_gallery_priority": {
        "val_fusion_gallery_mrr": 0.70,
        "val_fusion_gallery_recall_at_1": 0.15,
        "val_auprc": 0.10,
        "val_auroc": 0.05,
    },
    "hard_gallery_macro_retrieval": {
        "val_hard_gallery_macro_retrieval_score": 1.0,
    },
    # Fully user-defined weighted sum via `objective_weights`.
    "weighted_sum": None,
}

SUPPORTED_OBJECTIVE_COMPONENTS = {
    "val_auroc",
    "val_auprc",
    "val_recall_at_1",
    "val_recall_at_5",
    "val_mrr",
    "val_fusion_gallery_recall_at_1",
    "val_fusion_gallery_recall_at_5",
    "val_fusion_gallery_mrr",
    "val_hard_gallery_macro_mrr",
    "val_hard_gallery_macro_recall_at_1",
    "val_hard_gallery_macro_retrieval_score",
    "val_itc_acc",
    "val_neg_gw_min_recall",
    "val_neg_gw_guardrail_met",
}

METRIC_SHORT_NAMES = {
    "val_auroc": "AUROC",
    "val_auprc": "AUPRC",
    "val_recall_at_1": "R@1",
    "val_recall_at_5": "R@5",
    "val_mrr": "MRR",
    "val_fusion_gallery_recall_at_1": "FG_R@1",
    "val_fusion_gallery_recall_at_5": "FG_R@5",
    "val_fusion_gallery_mrr": "FG_MRR",
    "val_itc_acc": "ITC_ACC",
    "val_hard_gallery_macro_mrr": "HardMacroMRR",
    "val_hard_gallery_macro_recall_at_1": "HardMacroR@1",
    "val_hard_gallery_macro_retrieval_score": "HardMacroScore",
    "val_neg_gw_min_recall": "NegGWMinRecall",
    "val_neg_gw_guardrail_met": "NegGWGuardrail",
}


def _resolve_path(path: str) -> str:
    path = str(path)
    path = path.replace("<REPO_ROOT>", REPO_ROOT).replace("<BASE_DIR>", WORKSPACE_ROOT)
    return os.path.abspath(os.path.expanduser(path))


def _resolve_config_path(path: str, config_dir: str) -> str:
    path = str(path)
    path = path.replace("<REPO_ROOT>", REPO_ROOT).replace("<BASE_DIR>", WORKSPACE_ROOT)
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(config_dir, path)
    return os.path.abspath(path)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _validate_search_space_spec(name: str, spec: Dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise ValueError(f"search_space['{name}'] must be an object")
    kind = spec.get("type")
    if kind not in {"float", "int", "categorical"}:
        raise ValueError(f"search_space['{name}'].type must be float/int/categorical")
    if kind == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"search_space['{name}'].choices must be a non-empty list")
        return
    if "low" not in spec or "high" not in spec:
        raise ValueError(f"search_space['{name}'] must define low/high")
    if spec["low"] >= spec["high"]:
        raise ValueError(f"search_space['{name}'] requires low < high")


def _validate_balanced_model_capacity_config(cfg: Dict[str, Any]) -> None:
    tunable_params = set(cfg["tunable_params"])
    fixed_params = set(cfg["fixed_overrides"])
    capacity_is_tunable = "balanced_model_capacity" in tunable_params
    capacity_is_fixed = "balanced_model_capacity" in fixed_params
    if not capacity_is_tunable and not capacity_is_fixed:
        return

    conflicts = sorted(BALANCED_MODEL_CAPACITY_KEYS & (tunable_params | fixed_params))
    if conflicts:
        raise ValueError(
            "balanced_model_capacity controls model dimensions as one preset and cannot "
            f"be combined with independently tuned or fixed dimensions: {conflicts}"
        )

    if capacity_is_tunable:
        spec = cfg["search_space"]["balanced_model_capacity"]
        if spec["type"] != "categorical":
            raise ValueError(
                "search_space['balanced_model_capacity'] must use type='categorical'"
            )
        capacity_values = spec["choices"]
    else:
        capacity_values = [cfg["fixed_overrides"]["balanced_model_capacity"]]

    invalid = [
        value
        for value in capacity_values
        if not isinstance(value, str) or value not in BALANCED_MODEL_CAPACITY_PRESETS
    ]
    if invalid:
        raise ValueError(
            f"Unknown balanced_model_capacity preset(s): {sorted(invalid, key=str)}. "
            f"Supported: {sorted(BALANCED_MODEL_CAPACITY_PRESETS)}"
        )


def _apply_balanced_model_capacity(config: Dict[str, Any], preset_name: Any) -> None:
    if (
        not isinstance(preset_name, str)
        or preset_name not in BALANCED_MODEL_CAPACITY_PRESETS
    ):
        raise ValueError(
            f"Unknown balanced_model_capacity preset '{preset_name}'. "
            f"Supported: {sorted(BALANCED_MODEL_CAPACITY_PRESETS)}"
        )
    config.update(BALANCED_MODEL_CAPACITY_PRESETS[preset_name])


def _validate_metric_weights(
    weights: Dict[str, Any], field_name: str
) -> Dict[str, float]:
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{field_name} must be a non-empty object")

    normalized: Dict[str, float] = {}
    for key, value in weights.items():
        if key not in SUPPORTED_OBJECTIVE_COMPONENTS:
            raise ValueError(
                f"Unsupported metric '{key}' in {field_name}. "
                f"Supported: {sorted(SUPPORTED_OBJECTIVE_COMPONENTS)}"
            )
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}['{key}'] must be numeric") from exc
        if value < 0:
            raise ValueError(f"{field_name}['{key}'] must be >= 0")
        normalized[key] = value

    if sum(normalized.values()) <= 0:
        raise ValueError(f"Sum of {field_name} values must be > 0")
    return normalized


def _resolve_objective_weights(cfg: Dict[str, Any]) -> Dict[str, float]:
    metric_name = cfg["objective_metric"]
    if metric_name not in OBJECTIVE_PRESETS:
        raise ValueError(
            f"Unsupported objective_metric='{metric_name}'. "
            f"Supported: {sorted(OBJECTIVE_PRESETS.keys())}"
        )

    custom = cfg.get("objective_weights")
    if metric_name == "weighted_sum":
        if custom is None:
            raise ValueError(
                "objective_metric='weighted_sum' requires objective_weights"
            )
        return _validate_metric_weights(custom, "objective_weights")

    preset = OBJECTIVE_PRESETS[metric_name]
    if custom is None:
        return dict(preset)
    merged = dict(preset)
    merged.update(custom)
    return _validate_metric_weights(merged, "objective_weights")


def _resolve_objective_min_metrics(cfg: Dict[str, Any]) -> Dict[str, float]:
    raw = cfg.get("objective_min_metrics", {})
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError("objective_min_metrics must be an object")
    resolved: Dict[str, float] = {}
    for key, value in raw.items():
        if key not in SUPPORTED_OBJECTIVE_COMPONENTS:
            raise ValueError(
                f"Unsupported metric '{key}' in objective_min_metrics. "
                f"Supported: {sorted(SUPPORTED_OBJECTIVE_COMPONENTS)}"
            )
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"objective_min_metrics['{key}'] must be numeric") from exc
        if value < 0:
            raise ValueError(f"objective_min_metrics['{key}'] must be >= 0")
        resolved[key] = value
    for key, value in resolved.items():
        if value > 1.0:
            raise ValueError(
                f"objective_min_metrics['{key}'] should be <= 1.0, got {value}"
            )
    return resolved


def _objective_formula_str(weights: Dict[str, float]) -> str:
    total = sum(weights.values())
    terms = []
    for key, weight in weights.items():
        w = weight / total
        terms.append(f"{w:.3f}*{key}")
    return " + ".join(terms)


def _validate_existing_file(path_value: Any, field_name: str) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"{field_name} must be a non-empty string path")
    resolved = _resolve_path(path_value)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"{field_name} not found: {resolved}")
    return resolved


def load_hpo_config(config_path: str) -> Dict[str, Any]:
    cfg = _load_json(config_path)

    required = [
        "study_name",
        "n_trials",
        "output_dir",
        "epochs_per_trial",
        "search_space",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    cfg["config_path"] = _resolve_path(config_path)
    config_dir = os.path.dirname(cfg["config_path"])
    cfg["output_dir"] = _resolve_path(cfg["output_dir"])
    cfg["base_train_config"] = _resolve_config_path(
        cfg.get("base_train_config", DEFAULT_BASE_TRAIN_CONFIG), config_dir
    )
    default_train_config = cfg.get("default_train_config")
    cfg["default_train_config"] = (
        _resolve_config_path(default_train_config, config_dir)
        if default_train_config not in (None, "")
        else None
    )

    if not os.path.exists(cfg["base_train_config"]):
        raise FileNotFoundError(
            f"base_train_config not found: {cfg['base_train_config']}"
        )
    if cfg["default_train_config"] and not os.path.exists(cfg["default_train_config"]):
        raise FileNotFoundError(
            f"default_train_config not found: {cfg['default_train_config']}"
        )

    cfg.setdefault("n_startup_trials", 10)
    cfg.setdefault("storage", None)
    cfg.setdefault("objective_metric", "combined_auroc_g2o_r5")
    cfg.setdefault("objective_direction", "maximize")
    cfg.setdefault("objective_weights", None)
    cfg.setdefault("objective_min_metrics", {})
    cfg.setdefault("tunable_params", list(DEFAULT_TUNABLE_PARAMS))
    cfg.setdefault("fixed_overrides", {})
    cfg.setdefault("retrain_overrides", {})
    cfg.setdefault("num_workers", 4)

    if cfg["objective_direction"] not in {"maximize", "minimize"}:
        raise ValueError("objective_direction must be 'maximize' or 'minimize'")

    cfg["objective_weights"] = _resolve_objective_weights(cfg)
    cfg["objective_min_metrics"] = _resolve_objective_min_metrics(cfg)

    if "data_path" in cfg:
        cfg["data_path"] = _validate_existing_file(cfg["data_path"], "data_path")
    if "neg_data_path" in cfg and cfg["neg_data_path"] not in (None, ""):
        cfg["neg_data_path"] = _validate_existing_file(
            cfg["neg_data_path"], "neg_data_path"
        )
    if "test_data_path" in cfg and cfg["test_data_path"] not in (None, ""):
        cfg["test_data_path"] = _validate_existing_file(
            cfg["test_data_path"], "test_data_path"
        )

    if not isinstance(cfg["tunable_params"], list) or not cfg["tunable_params"]:
        raise ValueError("tunable_params must be a non-empty list")
    if not isinstance(cfg["fixed_overrides"], dict):
        raise ValueError("fixed_overrides must be an object")
    if not isinstance(cfg["retrain_overrides"], dict):
        raise ValueError("retrain_overrides must be an object")

    valid_tunable_names = (MAGIKS_ARG_KEYS - {"hpo_trial_number"}) | DERIVED_PARAM_KEYS
    unknown_tunable = sorted(set(cfg["tunable_params"]) - valid_tunable_names)
    if unknown_tunable:
        raise ValueError(f"Unknown tunable parameter(s): {unknown_tunable}")

    valid_fixed_names = (MAGIKS_ARG_KEYS - {"hpo_trial_number"}) | DERIVED_PARAM_KEYS
    unknown_fixed = sorted(set(cfg["fixed_overrides"].keys()) - valid_fixed_names)
    if unknown_fixed:
        raise ValueError(f"Unknown fixed_overrides parameter(s): {unknown_fixed}")

    valid_retrain_names = MAGIKS_ARG_KEYS - {"hpo_trial_number"}
    unknown_retrain = sorted(set(cfg["retrain_overrides"].keys()) - valid_retrain_names)
    if unknown_retrain:
        raise ValueError(f"Unknown retrain_overrides parameter(s): {unknown_retrain}")

    overlap = set(cfg["tunable_params"]) & set(cfg["fixed_overrides"].keys())
    if overlap:
        raise ValueError(
            f"Parameters cannot be both tuned and fixed: {sorted(overlap)}"
        )

    for p in cfg["tunable_params"]:
        if p not in cfg["search_space"]:
            raise ValueError(f"Missing search_space for tunable parameter '{p}'")
        _validate_search_space_spec(p, cfg["search_space"][p])

    _validate_balanced_model_capacity_config(cfg)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    if cfg["storage"] is None:
        db_path = os.path.join(cfg["output_dir"], "optuna_study.db")
        cfg["storage"] = f"sqlite:///{db_path}"

    return cfg


def suggest_from_space(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "float":
        kwargs = {"log": bool(spec.get("log", False))}
        if "step" in spec and spec["step"] is not None:
            kwargs["step"] = spec["step"]
            kwargs["log"] = False
        return trial.suggest_float(name, spec["low"], spec["high"], **kwargs)
    kwargs = {"log": bool(spec.get("log", False))}
    if "step" in spec and spec["step"] is not None:
        kwargs["step"] = int(spec["step"])
    return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), **kwargs)


def _filter_train_config(config: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in config.items():
        if key in STALE_KEYS:
            continue
        if key in MAGIKS_ARG_KEYS:
            out[key] = value
    return out


def load_base_train_config(hpo_cfg: Dict[str, Any]) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if hpo_cfg.get("default_train_config"):
        config.update(_load_json(hpo_cfg["default_train_config"]))
    config.update(_load_json(hpo_cfg["base_train_config"]))
    return config


def _curriculum_full_epoch(start_epoch: int, ramp_epochs: int) -> int:
    if int(ramp_epochs) <= 0:
        return int(start_epoch)
    return int(start_epoch) + int(ramp_epochs) - 1


def apply_batch_size_step_derivation(config: Dict[str, Any]) -> None:
    """Derive event-level train/val steps when batch_size changes.

    Keeps the same approximate number of positive-event draws per epoch as the
    default 1024-batch / 100-step setup (about 20,500 positive events).
    """
    if "batch_size" not in config:
        return
    batch_size = int(config["batch_size"])
    samples_per_gw = int(config.get("samples_per_gw", 4))
    neg_ratio = float(config.get("neg_gw_pair_ratio", 0.2))
    if batch_size <= 0 or samples_per_gw <= 0:
        raise ValueError("batch_size and samples_per_gw must be > 0")
    if not 0.0 <= neg_ratio < 1.0:
        raise ValueError("neg_gw_pair_ratio must be in [0, 1)")
    events_per_batch = int(batch_size * (1.0 - neg_ratio)) // samples_per_gw
    if events_per_batch < 1:
        raise ValueError(
            "batch_size is too small for the requested samples_per_gw and neg_gw_pair_ratio"
        )
    config["val_batch_size"] = batch_size
    config["steps_per_epoch"] = max(1, int(round(20500.0 / events_per_batch)))
    config["val_steps_per_epoch"] = max(1, int(round(config["steps_per_epoch"] * 0.16)))


def _sequential_phase_boundaries(config: Dict[str, Any]) -> Tuple[int, int]:
    cursor = 0
    if float(config.get("itc_weight", 0.0) or 0.0) > 0.0:
        cursor += int(config.get("stage_itc_epochs", 8))
    if float(config.get("cls_weight", 0.0) or 0.0) > 0.0:
        cursor += int(config.get("stage_cls_ramp_epochs", 4))
    retrieval_start = cursor
    if float(config.get("gallery_loss_weight", 0.0) or 0.0) > 0.0:
        cursor += int(config.get("stage_retrieval_ramp_epochs", 4))
    return retrieval_start, cursor


def _apply_curriculum_constraints(config: Dict[str, Any]) -> None:
    durations = {
        key: int(config.get(key, default))
        for key, default in (
            ("stage_itc_epochs", 8),
            ("stage_cls_ramp_epochs", 4),
            ("stage_retrieval_ramp_epochs", 4),
        )
    }
    if any(value < 0 for value in durations.values()):
        raise ValueError("Sequential curriculum stage durations must be >= 0.")
    if not any(
        float(config.get(key, 0.0) or 0.0) > 0.0
        for key in ("itc_weight", "cls_weight", "gallery_loss_weight")
    ):
        raise ValueError("Sequential curriculum has no enabled loss.")
    retrieval_start, joint_start = _sequential_phase_boundaries(config)
    epochs = int(config.get("epochs", 1))
    if joint_start >= epochs:
        raise ValueError(
            f"Sequential curriculum leaves no joint epoch: epochs={epochs}, joint_start={joint_start}."
        )

    guardrail_recall = float(config.get("neg_gw_guardrail_recall", 0.9))
    if not 0.0 < guardrail_recall <= 1.0:
        raise ValueError("neg_gw_guardrail_recall must be in (0, 1].")

    lr = float(config.get("lr", 0.0) or 0.0)
    min_lr = float(config.get("min_lr", 0.0) or 0.0)
    if lr > 0.0 and min_lr >= lr:
        raise ValueError(f"min_lr ({min_lr}) must be strictly less than lr ({lr})")

    gallery_ramp = durations["stage_retrieval_ramp_epochs"]
    gallery_full = _curriculum_full_epoch(retrieval_start, gallery_ramp)
    if (
        float(config.get("gallery_loss_weight", 0.0) or 0.0) > 0.0
        and gallery_full >= epochs
    ):
        raise ValueError(
            "gallery loss does not reach full activation within this trial: "
            f"epochs={epochs}, retrieval_start={retrieval_start}, "
            f"stage_retrieval_ramp_epochs={gallery_ramp}, full_epoch={gallery_full}"
        )

    hard_enabled = bool(config.get("gallery_hard_neg_enable", False))
    hard_weight = float(config.get("gallery_hard_neg_weight", 0.0) or 0.0)
    if hard_enabled and hard_weight > 0.0:
        hard_offset = int(
            config.get("gallery_hard_neg_start_after_retrieval_epochs", 0)
        )
        hard_ramp = int(config.get("gallery_hard_neg_ramp_epochs", 0))
        hard_start = retrieval_start + hard_offset
        hard_full = _curriculum_full_epoch(hard_start, hard_ramp)
        if hard_full >= epochs:
            raise ValueError(
                "gallery hard-negative loss does not reach full activation within this trial: "
                f"epochs={epochs}, retrieval_start={retrieval_start}, "
                f"gallery_hard_neg_start_after_retrieval_epochs={hard_offset}, "
                f"gallery_hard_neg_ramp_epochs={hard_ramp}, full_epoch={hard_full}"
            )


def build_trial_config(
    trial: optuna.Trial, hpo_cfg: Dict[str, Any], base_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    config = dict(base_cfg)
    default_aug_preset = {k: base_cfg.get(k, 0.0) for k in AUGMENT_PRESET_KEYS}

    # Runtime overrides from HPO config.
    for key in (
        "data_path",
        "neg_data_path",
        "neg_group",
        "test_data_path",
        "test_steps",
        "enable_ood_monitoring",
        "best_ckpt_metric",
        "cache_in_memory",
        "batch_size",
        "val_batch_size",
        "steps_per_epoch",
        "val_steps_per_epoch",
    ):
        if key in hpo_cfg:
            config[key] = hpo_cfg[key]

    config["num_workers"] = int(
        hpo_cfg.get("num_workers", config.get("num_workers", 4))
    )
    config["epochs"] = int(hpo_cfg["epochs_per_trial"])

    # Required HPO runtime behavior.
    config["early_stop_patience"] = 0
    config["skip_epoch_checkpoints"] = True

    sampled: Dict[str, Any] = {}
    for name in hpo_cfg["tunable_params"]:
        sampled[name] = suggest_from_space(trial, name, hpo_cfg["search_space"][name])

    augment_enable = None
    balanced_model_capacity = None
    for name, value in sampled.items():
        if name == "ref_shared_dim":
            shared = int(value)
            config["n_ref"] = shared
            config["ref_dim"] = shared
        elif name == "augment_enable":
            augment_enable = bool(value)
        elif name == "balanced_model_capacity":
            balanced_model_capacity = value
        else:
            config[name] = value

    fixed_overrides = dict(hpo_cfg.get("fixed_overrides", {}))
    if "augment_enable" in fixed_overrides:
        augment_enable = bool(fixed_overrides.pop("augment_enable"))
    if "balanced_model_capacity" in fixed_overrides:
        if balanced_model_capacity is not None:
            raise ValueError("balanced_model_capacity cannot be both tuned and fixed")
        balanced_model_capacity = fixed_overrides.pop("balanced_model_capacity")

    if balanced_model_capacity is not None:
        _apply_balanced_model_capacity(config, balanced_model_capacity)

    config.update(fixed_overrides)

    if "batch_size" in sampled or "batch_size" in fixed_overrides:
        apply_batch_size_step_derivation(config)

    if augment_enable is not None:
        if augment_enable:
            for key, value in default_aug_preset.items():
                config[key] = value
        else:
            for key in AUGMENT_PRESET_KEYS:
                config[key] = 0.0

    if int(config.get("proj_dim", 0)) < int(config.get("enc_dim", 0)):
        config["proj_dim"] = int(config["enc_dim"])

    _apply_curriculum_constraints(config)

    trial_ckpt = os.path.join(
        hpo_cfg["output_dir"], "results", f"trial_{trial.number}", "checkpoints"
    )
    os.makedirs(trial_ckpt, exist_ok=True)
    config["ckpt_path"] = trial_ckpt
    config["resume"] = None
    config["hpo_trial_number"] = trial.number

    return _filter_train_config(config)


def run_trial_subprocess(
    config: Dict[str, Any], trial_number: int, output_dir: str
) -> Dict[str, Any]:
    """Run a single training trial as a subprocess and return parsed trial_results.json."""

    config_dir = os.path.join(output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"trial_{trial_number}.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    cmd = [sys.executable, "-u", TRAIN_SCRIPT, "--json_config", config_path]

    print(f"\n{'=' * 60}")
    print(f"Trial {trial_number}: Starting training")
    print(
        "  "
        f"lr={config.get('lr', 'N/A')}, weight_decay={config.get('weight_decay', 'N/A')}, "
        f"warmup_epochs={config.get('warmup_epochs', 'N/A')}"
    )
    print(
        "  "
        f"stage_itc_epochs={config.get('stage_itc_epochs', 'N/A')}, "
        f"stage_cls_ramp_epochs={config.get('stage_cls_ramp_epochs', 'N/A')}, "
        f"stage_retrieval_ramp_epochs={config.get('stage_retrieval_ramp_epochs', 'N/A')}"
    )
    print(
        "  "
        f"encoder_lr_ratio={config.get('encoder_lr_ratio', 'N/A')}, "
        f"gallery_hard_neg_topk={config.get('gallery_hard_neg_topk', 'N/A')}, "
        f"gallery_hard_neg_weight={config.get('gallery_hard_neg_weight', 'N/A')}"
    )
    print(f"{'=' * 60}")

    log_path = os.path.join(output_dir, "results", f"trial_{trial_number}", "train.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    t0 = time.time()
    tail_lines = []
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            log_f.write(line)
            tail_lines.append(line.rstrip())
            if len(tail_lines) > 40:
                tail_lines.pop(0)
            if (
                ("Epoch " in line and "Complete" in line)
                or "Val Avg Total" in line
                or "Retrieval:" in line
                or "Saved best" in line
                or "Trial results saved" in line
            ):
                print(f"  [T{trial_number}] {line.rstrip()}", flush=True)
        proc.wait()

    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"Trial {trial_number}: FAILED (exit code {proc.returncode})")
        for line in tail_lines:
            print(f"  {line}")
        raise optuna.TrialPruned(f"Training failed with exit code {proc.returncode}")

    print(f"Trial {trial_number}: Completed in {elapsed:.0f}s")

    result_path = os.path.join(config["ckpt_path"], "ALBEF", "trial_results.json")
    if not os.path.exists(result_path):
        print(f"Trial {trial_number}: missing {result_path}")
        raise optuna.TrialPruned("Missing trial_results.json")

    with open(result_path) as f:
        return json.load(f)


def worst_value(direction: str) -> float:
    return float("-inf") if direction == "maximize" else float("inf")


def _read_metric(results: Dict[str, Any], key: str) -> float:
    try:
        return float(results.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def compute_objective_score(
    results: Dict[str, Any], weights: Dict[str, float]
) -> float:
    weighted_sum = 0.0
    weight_total = 0.0
    for key, weight in weights.items():
        value = _read_metric(results, key)
        if math.isnan(value):
            return float("nan")
        weighted_sum += weight * value
        weight_total += weight
    if weight_total <= 0:
        return float("nan")
    return weighted_sum / weight_total


def check_objective_min_metrics(
    results: Dict[str, Any], min_metrics: Dict[str, float]
) -> Tuple[bool, str]:
    for key, min_value in min_metrics.items():
        value = _read_metric(results, key)
        if math.isnan(value):
            return False, f"{key}=nan < min({min_value:.6f})"
        if value < min_value:
            return False, f"{key}={value:.6f} < min({min_value:.6f})"
    return True, ""


def format_metric_summary(results: Dict[str, Any], metric_keys) -> str:
    parts = []
    for key in metric_keys:
        value = _read_metric(results, key)
        name = METRIC_SHORT_NAMES.get(key, key)
        if math.isnan(value):
            parts.append(f"{name}=nan")
        else:
            parts.append(f"{name}={value:.6f}")
    return ", ".join(parts)


def objective(
    trial: optuna.Trial, hpo_cfg: Dict[str, Any], base_cfg: Dict[str, Any]
) -> float:
    config = build_trial_config(trial, hpo_cfg, base_cfg)
    results = run_trial_subprocess(config, trial.number, hpo_cfg["output_dir"])

    score = compute_objective_score(results, hpo_cfg["objective_weights"])
    passed_min_metrics, min_fail_reason = check_objective_min_metrics(
        results, hpo_cfg["objective_min_metrics"]
    )
    if math.isnan(score) or not passed_min_metrics:
        score = worst_value(hpo_cfg["objective_direction"])

    # Persist useful metrics
    trial.set_user_attr("objective_score", score)
    trial.set_user_attr("objective_metric", hpo_cfg["objective_metric"])
    trial.set_user_attr("objective_min_metrics_passed", int(passed_min_metrics))
    if not passed_min_metrics:
        trial.set_user_attr("objective_min_metrics_fail", min_fail_reason)
    trial.set_user_attr("best_epoch", results.get("best_epoch", -1))
    trial.set_user_attr("final_epoch", results.get("final_epoch", -1))
    for metric_key in sorted(SUPPORTED_OBJECTIVE_COMPONENTS):
        trial.set_user_attr(metric_key, results.get(metric_key, 0.0))
    hard_gallery = results.get("val_hard_gallery", {})
    if isinstance(hard_gallery, dict) and hard_gallery:
        trial.set_user_attr(
            "val_hard_gallery_by_source",
            hard_gallery.get("by_source", {}),
        )
        trial.set_user_attr(
            "val_hard_gallery_source_gap",
            hard_gallery.get("source_gap", {}),
        )

    print(
        f"Trial {trial.number}: objective={score:.6f} "
        f"({format_metric_summary(results, hpo_cfg['objective_weights'].keys())})"
    )
    if not passed_min_metrics:
        print(
            f"  [T{trial.number}] objective min-metric check failed: {min_fail_reason}"
        )
    return score


def dry_run(hpo_cfg: Dict[str, Any], base_cfg: Dict[str, Any]) -> None:
    print("Dry run: sampling one trial config without launching training.")
    sampler = TPESampler(seed=42, n_startup_trials=1)
    study = optuna.create_study(
        direction=hpo_cfg["objective_direction"], sampler=sampler
    )
    trial = study.ask()
    config = build_trial_config(trial, hpo_cfg, base_cfg)

    print("\nSampled trial params:")
    for k, v in sorted(trial.params.items()):
        print(f"  {k}: {v}")

    retrieval_start, _ = _sequential_phase_boundaries(config)
    gallery_full = _curriculum_full_epoch(
        retrieval_start, int(config.get("stage_retrieval_ramp_epochs", 4))
    )
    hard_start = retrieval_start + int(
        config.get("gallery_hard_neg_start_after_retrieval_epochs", 0)
    )
    hard_full = _curriculum_full_epoch(
        hard_start, int(config.get("gallery_hard_neg_ramp_epochs", 0))
    )

    print("\nConstraint checks:")
    capacity_name = trial.params.get(
        "balanced_model_capacity",
        hpo_cfg.get("fixed_overrides", {}).get("balanced_model_capacity"),
    )
    if capacity_name is not None:
        dims = ", ".join(
            f"{key}={config.get(key)}" for key in sorted(BALANCED_MODEL_CAPACITY_KEYS)
        )
        print(f"  balanced_model_capacity: {capacity_name} ({dims})")
    print(f"  n_ref == ref_dim: {config.get('n_ref')} == {config.get('ref_dim')}")
    print(f"  sequential retrieval phase start (0-based): {retrieval_start}")
    print(f"  gallery full activation epoch (0-based): {gallery_full}")
    print(f"  gallery hard-negative full activation epoch (0-based): {hard_full}")
    print(f"  epochs = {config.get('epochs')}")

    print("\nResolved trial training config:")
    print(json.dumps(config, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Config-driven Optuna HPO for MAGIKS training"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to HPO config JSON"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate config and print one sampled trial",
    )
    args = parser.parse_args()

    hpo_cfg = load_hpo_config(args.config)
    base_cfg = load_base_train_config(hpo_cfg)

    print(f"Study: {hpo_cfg['study_name']}")
    print(f"Storage: {hpo_cfg['storage']}")
    print(f"Output: {hpo_cfg['output_dir']}")
    print(f"Trials: {hpo_cfg['n_trials']} ({hpo_cfg['epochs_per_trial']} epochs each)")
    print(
        f"Objective: {hpo_cfg['objective_metric']} ({hpo_cfg['objective_direction']})"
    )
    print(f"Objective formula: {_objective_formula_str(hpo_cfg['objective_weights'])}")
    if hpo_cfg["objective_min_metrics"]:
        print(f"Objective minimum metrics: {hpo_cfg['objective_min_metrics']}")
    print(f"Config: {hpo_cfg['config_path']}")
    if hpo_cfg.get("default_train_config"):
        print(f"Default train config: {hpo_cfg['default_train_config']}")
    print(f"Base train config: {hpo_cfg['base_train_config']}")

    if args.dry_run:
        dry_run(hpo_cfg, base_cfg)
        return

    sampler = TPESampler(n_startup_trials=hpo_cfg["n_startup_trials"], seed=42)
    pruner = MedianPruner(
        n_startup_trials=hpo_cfg["n_startup_trials"], n_warmup_steps=0
    )

    study = optuna.create_study(
        study_name=hpo_cfg["study_name"],
        storage=hpo_cfg["storage"],
        sampler=sampler,
        pruner=pruner,
        direction=hpo_cfg["objective_direction"],
        load_if_exists=True,
    )

    study.set_user_attr("objective_metric", hpo_cfg["objective_metric"])
    study.set_user_attr("objective_direction", hpo_cfg["objective_direction"])
    study.set_user_attr("objective_weights", hpo_cfg["objective_weights"])
    study.set_user_attr("objective_min_metrics", hpo_cfg["objective_min_metrics"])
    study.set_user_attr("retrain_overrides", hpo_cfg.get("retrain_overrides", {}))

    study.optimize(
        lambda trial: objective(trial, hpo_cfg, base_cfg),
        n_trials=int(hpo_cfg["n_trials"]),
        catch=(Exception,),
    )

    print("\n" + "=" * 60)
    print("HPO COMPLETE")
    print("=" * 60)
    total_trials = len(study.trials)
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    failed_trials = [
        t for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE
    ]
    print(f"Total trials: {total_trials}")
    print(f"Completed: {len(completed_trials)}")
    print(f"Pruned/Failed: {len(failed_trials)}")

    if not completed_trials:
        print("\nNo completed trials. Skipping best-trial export.")
        sys.exit(2)

    best = study.best_trial
    print(f"\nBest trial: #{best.number}")
    print(f"  objective: {best.value:.6f}")
    print("  parameters:")
    for key, value in sorted(best.params.items()):
        print(f"    {key}: {value}")

    best_config_path = os.path.join(hpo_cfg["output_dir"], "best_config.json")
    config_path = os.path.join(
        hpo_cfg["output_dir"], "configs", f"trial_{best.number}.json"
    )
    if os.path.exists(config_path):
        with open(config_path) as f:
            best_config = json.load(f)
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"\nBest config saved to: {best_config_path}")


if __name__ == "__main__":
    main()
