#!/usr/bin/env python3
"""
Config-driven Optuna hyperparameter optimization for ALBEF GW-Optical training.

Usage:
    python hpo_optuna.py --config args/hpo_v1.json
    python hpo_optuna.py --config args/hpo_v1.json --dry_run
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
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "ALBEF_train.py")
DEFAULT_BASE_TRAIN_CONFIG = os.path.join(SCRIPT_DIR, "args", "ALBEF_BNS_NSBH.json")

ALBEF_BOOL_KEYS = {
    "cache_in_memory",
    "enable_ood_monitoring",
    "neg_time_offset_enable",
    "cls_time_delta_enable",
    "cls_dt_aug_enable",
    "extra_dt_dropout_enable",
    "mask_itc",
    "semi_hard",
    "hardneg_memory_bank_enable",
    "use_lightweight_gw",
    "dual_fusion",
    "skip_epoch_checkpoints",
}

ALBEF_ARG_KEYS = {
    "data_path",
    "test_data_path",
    "test_steps",
    "enable_ood_monitoring",
    "neg_data_path",
    "neg_group",
    "neg_time_offset_enable",
    "neg_offset_dist_npz",
    "neg_offset_dist_key",
    "neg_offset_train_sampling",
    "neg_offset_eval_mode",
    "neg_offset_eval_quantiles",
    "neg_offset_scale_days_divisor",
    "neg_offset_seed",
    "neg_offset_bank_size",
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
    "cls_time_delta_enable",
    "cls_time_delta_scale_days",
    "cls_time_delta_clip",
    "nonkn_cls_base_field",
    "cls_dt_aug_enable",
    "cls_dt_aug_start_epoch",
    "cls_dt_jitter_sigma_days",
    "cls_dt_jitter_clip_days",
    "cls_dt_dropout_p_pos",
    "cls_dt_dropout_p_hard",
    "cls_dt_dropout_p_extra",
    "extra_dt_dropout_enable",
    "extra_dt_dropout_p",
    "extra_dt_dropout_apply_to",
    "gw_dropout",
    "opt_dropout",
    "proj_dropout",
    "feature_dropout",
    "itc_weight",
    "cls_weight",
    "cls_pos_weight",
    "cls_neg_weight",
    "cls_extra_neg_weight",
    "cls_pos_nodt_aux_weight",
    "cls_ramp_epochs",
    "itc_decay_start_epoch",
    "itc_decay_epochs",
    "itc_decay_ratio",
    "itc_label_smoothing",
    "itc_loss_type",
    "supcon_margin",
    "samples_per_gw",
    "min_lc_per_gw",
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
}

DEFAULT_TUNABLE_PARAMS = [
    "lr",
    "weight_decay",
    "warmup_epochs",
    "enc_dim",
    "proj_dim",
    "gw_dropout",
    "opt_dropout",
    "fusion_dropout",
    "proj_dropout",
    "feature_dropout",
    "label_smoothing",
    "itc_weight",
    "cls_weight",
    "cls_neg_weight",
    "cls_extra_neg_weight",
    "ref_shared_dim",
    "samples_per_gw",
    "cls_start_epoch",
    "semi_hard_margin",
]

OBJECTIVE_PRESETS = {
    # Historical objective used in earlier experiments.
    "combined_auroc_g2o_r5": {"val_auroc": 0.5, "val_recall_at_5": 0.5},
    # Pure classification objective.
    "cls_auroc_auprc": {"val_auroc": 0.5, "val_auprc": 0.5},
    # Classification-priority objective with retrieval as a soft guard.
    "cls_priority_auprc_auroc_r5": {"val_auprc": 0.45, "val_auroc": 0.35, "val_recall_at_5": 0.20},
    # Fully user-defined weighted sum via `objective_weights`.
    "weighted_sum": None,
}

SUPPORTED_OBJECTIVE_COMPONENTS = {
    "val_auroc",
    "val_auprc",
    "val_recall_at_1",
    "val_recall_at_5",
    "val_mrr",
    "val_itc_acc",
}

METRIC_SHORT_NAMES = {
    "val_auroc": "AUROC",
    "val_auprc": "AUPRC",
    "val_recall_at_1": "R@1",
    "val_recall_at_5": "R@5",
    "val_mrr": "MRR",
    "val_itc_acc": "ITC_ACC",
}


def _resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


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


def _validate_metric_weights(weights: Dict[str, Any], field_name: str) -> Dict[str, float]:
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
            raise ValueError("objective_metric='weighted_sum' requires objective_weights")
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
            raise ValueError(f"objective_min_metrics['{key}'] should be <= 1.0, got {value}")
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

    required = ["study_name", "n_trials", "output_dir", "epochs_per_trial", "search_space"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    cfg["config_path"] = _resolve_path(config_path)
    cfg["output_dir"] = _resolve_path(cfg["output_dir"])
    cfg["base_train_config"] = _resolve_path(cfg.get("base_train_config", DEFAULT_BASE_TRAIN_CONFIG))

    if not os.path.exists(cfg["base_train_config"]):
        raise FileNotFoundError(f"base_train_config not found: {cfg['base_train_config']}")

    cfg.setdefault("n_startup_trials", 10)
    cfg.setdefault("storage", None)
    cfg.setdefault("objective_metric", "combined_auroc_g2o_r5")
    cfg.setdefault("objective_direction", "maximize")
    cfg.setdefault("objective_weights", None)
    cfg.setdefault("objective_min_metrics", {})
    cfg.setdefault("tunable_params", list(DEFAULT_TUNABLE_PARAMS))
    cfg.setdefault("fixed_overrides", {})
    cfg.setdefault("num_workers", 4)

    if cfg["objective_direction"] not in {"maximize", "minimize"}:
        raise ValueError("objective_direction must be 'maximize' or 'minimize'")

    cfg["objective_weights"] = _resolve_objective_weights(cfg)
    cfg["objective_min_metrics"] = _resolve_objective_min_metrics(cfg)

    if "data_path" in cfg:
        cfg["data_path"] = _validate_existing_file(cfg["data_path"], "data_path")
    if "neg_data_path" in cfg and cfg["neg_data_path"] not in (None, ""):
        cfg["neg_data_path"] = _validate_existing_file(cfg["neg_data_path"], "neg_data_path")
    if "test_data_path" in cfg and cfg["test_data_path"] not in (None, ""):
        cfg["test_data_path"] = _validate_existing_file(cfg["test_data_path"], "test_data_path")

    if not isinstance(cfg["tunable_params"], list) or not cfg["tunable_params"]:
        raise ValueError("tunable_params must be a non-empty list")

    valid_tunable_names = (ALBEF_ARG_KEYS - {"hpo_trial_number"}) | {"ref_shared_dim"}
    unknown_tunable = sorted(set(cfg["tunable_params"]) - valid_tunable_names)
    if unknown_tunable:
        raise ValueError(f"Unknown tunable parameter(s): {unknown_tunable}")

    unknown_fixed = sorted(set(cfg["fixed_overrides"].keys()) - (ALBEF_ARG_KEYS - {"hpo_trial_number"}))
    if unknown_fixed:
        raise ValueError(f"Unknown fixed_overrides parameter(s): {unknown_fixed}")

    overlap = set(cfg["tunable_params"]) & set(cfg["fixed_overrides"].keys())
    if overlap:
        raise ValueError(f"Parameters cannot be both tuned and fixed: {sorted(overlap)}")

    for p in cfg["tunable_params"]:
        if p not in cfg["search_space"]:
            raise ValueError(f"Missing search_space for tunable parameter '{p}'")
        _validate_search_space_spec(p, cfg["search_space"][p])

    # Fill default storage path
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
        if key in ALBEF_ARG_KEYS:
            out[key] = value
    return out


def build_trial_config(trial: optuna.Trial, hpo_cfg: Dict[str, Any], base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(base_cfg)

    # Runtime overrides from HPO config
    for key in (
        "data_path",
        "neg_data_path",
        "neg_group",
        "test_data_path",
        "test_steps",
        "enable_ood_monitoring",
        "best_ckpt_metric",
        "cache_in_memory",
    ):
        if key in hpo_cfg:
            config[key] = hpo_cfg[key]

    config["num_workers"] = int(hpo_cfg.get("num_workers", config.get("num_workers", 4)))
    config["epochs"] = int(hpo_cfg["epochs_per_trial"])

    # Required HPO runtime behavior
    config["early_stop_patience"] = 0
    config["skip_epoch_checkpoints"] = True

    # Sample tunable params
    sampled: Dict[str, Any] = {}
    for name in hpo_cfg["tunable_params"]:
        sampled[name] = suggest_from_space(trial, name, hpo_cfg["search_space"][name])

    # Apply sampled values (with tied params)
    for name, value in sampled.items():
        if name == "ref_shared_dim":
            shared = int(value)
            config["n_ref"] = shared
            config["ref_dim"] = shared
        else:
            config[name] = value

    # Fixed overrides from HPO config
    config.update(hpo_cfg.get("fixed_overrides", {}))

    # Hard constraints requested by user
    cls_start = int(config.get("cls_start_epoch", 0))
    config["hard_neg_start_epoch"] = cls_start + 5
    config["cls_ramp_epochs"] = 5
    config["hard_neg_ramp_epochs"] = 5

    # Keep proj head width valid
    if int(config.get("proj_dim", 0)) < int(config.get("enc_dim", 0)):
        config["proj_dim"] = int(config["enc_dim"])

    if config["epochs"] < int(config["hard_neg_start_epoch"]):
        raise ValueError(
            "epochs_per_trial is smaller than derived hard_neg_start_epoch; "
            f"got epochs={config['epochs']} and hard_neg_start_epoch={config['hard_neg_start_epoch']}"
        )

    # Trial-specific checkpoint path
    trial_ckpt = os.path.join(hpo_cfg["output_dir"], "results", f"trial_{trial.number}", "checkpoints")
    os.makedirs(trial_ckpt, exist_ok=True)
    config["ckpt_path"] = trial_ckpt
    config["resume"] = None
    config["hpo_trial_number"] = trial.number

    return _filter_train_config(config)


def run_trial_subprocess(config: Dict[str, Any], trial_number: int, output_dir: str) -> Dict[str, Any]:
    """Run a single training trial as a subprocess and return parsed trial_results.json."""

    config_dir = os.path.join(output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"trial_{trial_number}.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    cmd = [sys.executable, "-u", TRAIN_SCRIPT]
    for key, value in config.items():
        if key == "hpo_trial_number":
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            # Only true CLI flags are passed without an explicit value.
            # Bool-typed values for non-flag args (e.g., persistent_workers)
            # must still be serialized as 0/1.
            if key in ALBEF_BOOL_KEYS:
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append("1" if value else "0")
            continue
        cmd.append(f"--{key}")
        cmd.append(str(value))

    print(f"\n{'=' * 60}")
    print(f"Trial {trial_number}: Starting training")
    print(f"  lr={config.get('lr', 'N/A')}, enc_dim={config.get('enc_dim', 'N/A')}, proj_dim={config.get('proj_dim', 'N/A')}")
    print(f"  n_ref={config.get('n_ref', 'N/A')}, ref_dim={config.get('ref_dim', 'N/A')}, samples_per_gw={config.get('samples_per_gw', 'N/A')}")
    print(f"  cls_start_epoch={config.get('cls_start_epoch', 'N/A')}, hard_neg_start_epoch={config.get('hard_neg_start_epoch', 'N/A')}")
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


def compute_objective_score(results: Dict[str, Any], weights: Dict[str, float]) -> float:
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


def check_objective_min_metrics(results: Dict[str, Any], min_metrics: Dict[str, float]) -> Tuple[bool, str]:
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


def objective(trial: optuna.Trial, hpo_cfg: Dict[str, Any], base_cfg: Dict[str, Any]) -> float:
    config = build_trial_config(trial, hpo_cfg, base_cfg)
    results = run_trial_subprocess(config, trial.number, hpo_cfg["output_dir"])

    score = compute_objective_score(results, hpo_cfg["objective_weights"])
    passed_min_metrics, min_fail_reason = check_objective_min_metrics(results, hpo_cfg["objective_min_metrics"])
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
    trial.set_user_attr("val_recall_at_1", results.get("val_recall_at_1", 0.0))
    trial.set_user_attr("val_recall_at_5", results.get("val_recall_at_5", 0.0))
    trial.set_user_attr("val_auroc", results.get("val_auroc", 0.0))
    trial.set_user_attr("val_auprc", results.get("val_auprc", 0.0))
    trial.set_user_attr("val_itc_acc", results.get("val_itc_acc", 0.0))

    print(
        f"Trial {trial.number}: objective={score:.6f} "
        f"({format_metric_summary(results, hpo_cfg['objective_weights'].keys())})"
    )
    if not passed_min_metrics:
        print(f"  [T{trial.number}] objective min-metric check failed: {min_fail_reason}")
    return score


def dry_run(hpo_cfg: Dict[str, Any], base_cfg: Dict[str, Any]) -> None:
    print("Dry run: sampling one trial config without launching training.")
    sampler = TPESampler(seed=42, n_startup_trials=1)
    study = optuna.create_study(direction=hpo_cfg["objective_direction"], sampler=sampler)
    trial = study.ask()
    config = build_trial_config(trial, hpo_cfg, base_cfg)

    print("\nSampled trial params:")
    for k, v in sorted(trial.params.items()):
        print(f"  {k}: {v}")

    print("\nConstraint checks:")
    print(f"  n_ref == ref_dim: {config.get('n_ref')} == {config.get('ref_dim')}")
    print(f"  hard_neg_start_epoch = cls_start_epoch + 5: {config.get('hard_neg_start_epoch')} = {config.get('cls_start_epoch')} + 5")
    print(f"  cls_ramp_epochs = {config.get('cls_ramp_epochs')}")
    print(f"  hard_neg_ramp_epochs = {config.get('hard_neg_ramp_epochs')}")
    print(f"  epochs = {config.get('epochs')}")

    print("\nResolved trial training config:")
    print(json.dumps(config, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Config-driven Optuna HPO for ALBEF training")
    parser.add_argument("--config", type=str, required=True, help="Path to HPO config JSON")
    parser.add_argument("--dry_run", action="store_true", help="Validate config and print one sampled trial")
    args = parser.parse_args()

    hpo_cfg = load_hpo_config(args.config)
    base_cfg = _load_json(hpo_cfg["base_train_config"])

    print(f"Study: {hpo_cfg['study_name']}")
    print(f"Storage: {hpo_cfg['storage']}")
    print(f"Output: {hpo_cfg['output_dir']}")
    print(f"Trials: {hpo_cfg['n_trials']} ({hpo_cfg['epochs_per_trial']} epochs each)")
    print(f"Objective: {hpo_cfg['objective_metric']} ({hpo_cfg['objective_direction']})")
    print(f"Objective formula: {_objective_formula_str(hpo_cfg['objective_weights'])}")
    if hpo_cfg["objective_min_metrics"]:
        print(f"Objective minimum metrics: {hpo_cfg['objective_min_metrics']}")
    print(f"Config: {hpo_cfg['config_path']}")

    if args.dry_run:
        dry_run(hpo_cfg, base_cfg)
        return

    sampler = TPESampler(n_startup_trials=hpo_cfg["n_startup_trials"], seed=42)
    pruner = MedianPruner(n_startup_trials=hpo_cfg["n_startup_trials"], n_warmup_steps=0)

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

    study.optimize(
        lambda trial: objective(trial, hpo_cfg, base_cfg),
        n_trials=int(hpo_cfg["n_trials"]),
        catch=(Exception,),
    )

    print("\n" + "=" * 60)
    print("HPO COMPLETE")
    print("=" * 60)
    total_trials = len(study.trials)
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed_trials = [t for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE]
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
    config_path = os.path.join(hpo_cfg["output_dir"], "configs", f"trial_{best.number}.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            best_config = json.load(f)
        with open(best_config_path, "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"\nBest config saved to: {best_config_path}")


if __name__ == "__main__":
    main()
