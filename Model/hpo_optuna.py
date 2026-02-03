#!/usr/bin/env python3
"""
Optuna-based hyperparameter optimization for ALBEF GW-Optical training.

Usage (sequential, within a GPU SLURM job):
    python hpo_optuna.py --n_trials 50 --study_name albef_hpo_v1

Usage (resume existing study):
    python hpo_optuna.py --n_trials 50 --study_name albef_hpo_v1 --storage sqlite:///hpo_results/optuna_study.db
"""

import argparse
import json
import os
import subprocess
import sys
import time

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# ─── paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "ALBEF_train.py")
TRAIN_SHELL = os.path.join(SCRIPT_DIR, "ALBEF_train.sh")

# Default data paths — override with CLI args
DEFAULT_DATA_PATH = "/fred/oz016/bgao_kn/data/LSST_KN_BNS_AUG/combined_dataset_with_neg_gw.h5"
DEFAULT_NEG_DATA_PATH = "/fred/oz016/bgao_kn/data/ELASTICC2_TRAIN/negative_dataset.h5"
DEFAULT_NEG_GROUP = "ELASTICC2_TRAIN/optical_data"


def build_trial_config(trial, args):
    """Sample hyperparameters from Optuna and build a full training config dict."""

    # ── Group 1: Training Dynamics ────────────────────────────────────────
    batch_size = 1024            # Fixed — larger batch benefits contrastive learning
    val_batch_size = 1024        # Fixed — same as train
    steps_per_epoch = 1_024_000 // batch_size       # = 1000
    val_steps_per_epoch = 25600 // val_batch_size  # = 25

    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.1, log=True)
    warmup_epochs = trial.suggest_categorical("warmup_epochs", [0, 3, 5, 10])

    # ── Group 2: Contrastive Learning ─────────────────────────────────────
    supcon_temperature = trial.suggest_float("supcon_temperature", 0.03, 0.3)
    supcon_margin = trial.suggest_float("supcon_margin", 0.0, 0.2)
    samples_per_gw = trial.suggest_categorical("samples_per_gw", [2, 4, 8])
    neg_gw_ratio = trial.suggest_float("neg_gw_ratio", 0.05, 0.4)

    # ── Group 3: Model Architecture ───────────────────────────────────────
    enc_dim = trial.suggest_categorical("enc_dim", [64, 128, 256])
    proj_dim = trial.suggest_categorical("proj_dim", [128, 256, 512])
    ref_dim = trial.suggest_categorical("ref_dim", [32, 64, 128])
    n_ref = trial.suggest_categorical("n_ref", [32, 64, 128])

    # Enforce proj_dim >= enc_dim
    if proj_dim < enc_dim:
        proj_dim = enc_dim

    # ── Group 4: Regularization ───────────────────────────────────────────
    gw_dropout = trial.suggest_float("gw_dropout", 0.05, 0.6)
    opt_dropout = trial.suggest_float("opt_dropout", 0.05, 0.5)
    fusion_dropout = trial.suggest_float("fusion_dropout", 0.05, 0.6)
    proj_dropout = trial.suggest_float("proj_dropout", 0.0, 0.4)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    # ── Group 5: Loss Weights & Scheduling ────────────────────────────────
    itc_weight = trial.suggest_float("itc_weight", 0.5, 2.0)
    cls_weight = trial.suggest_float("cls_weight", 0.5, 2.0)
    cls_ramp_epochs = trial.suggest_categorical("cls_ramp_epochs", [0, 5, 10, 20])
    cls_neg_weight = trial.suggest_float("cls_neg_weight", 0.2, 1.0)
    cls_extra_neg_weight = trial.suggest_float("cls_extra_neg_weight", 0.2, 1.0)

    # ── Build full config ─────────────────────────────────────────────────
    trial_ckpt = os.path.join(args.output_dir, "results", f"trial_{trial.number}", "checkpoints")
    os.makedirs(trial_ckpt, exist_ok=True)

    config = {
        # Data
        "data_path": args.data_path,
        "neg_data_path": args.neg_data_path,
        "neg_group": args.neg_group,

        # Training
        "epochs": args.epochs_per_trial,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip_norm": 1.0,
        "lr_scheduler": "cosine",
        "warmup_epochs": warmup_epochs,
        "min_lr": 2e-5,

        # Data loading
        "num_workers": args.num_workers,
        "pin_memory": 1,
        "persistent_workers": 1,
        "prefetch_factor": 4,
        "stage_to_jobfs": args.stage_to_jobfs,
        "cache_in_memory": args.cache_in_memory,

        # Validation
        "val_split": 0.2,
        "val_batch_size": val_batch_size,
        "val_steps_per_epoch": val_steps_per_epoch,
        "split_seed": 42,
        "early_stop_patience": 10,
        "early_stop_min_delta": 0.005,

        # Model architecture
        "enc_dim": enc_dim,
        "proj_dim": proj_dim,
        "ref_dim": ref_dim,
        "n_ref": n_ref,
        "ref_start": -0.3,
        "ref_end": 0.6,
        "fusion_attn_dim": None,
        "fusion_hidden_dim": None,
        "use_lightweight_gw": True,

        # Contrastive learning
        "itc_loss_type": "supcon",
        "supcon_temperature": supcon_temperature,
        "supcon_margin": supcon_margin,
        "samples_per_gw": samples_per_gw,
        "min_lc_per_gw": 2,
        "temp_init": 0.1,
        "temp_final": 0.1,
        "temp_min": 0.01,
        "temp_max": 1.0,
        "temp_schedule": "fixed",

        # Negative GW
        "use_neg_gw": True,
        "neg_gw_ratio": neg_gw_ratio,
        "mask_itc": False,

        # Regularization
        "gw_dropout": gw_dropout,
        "opt_dropout": opt_dropout,
        "fusion_dropout": fusion_dropout,
        "proj_dropout": proj_dropout,
        "feature_dropout": 0.0,
        "label_smoothing": label_smoothing,

        # Loss weights
        "itc_weight": itc_weight,
        "cls_weight": cls_weight,
        "cls_pos_weight": 1.0,
        "cls_neg_weight": cls_neg_weight,
        "cls_extra_neg_weight": cls_extra_neg_weight,
        "cls_ramp_epochs": cls_ramp_epochs,
        "cls_start_epoch": 0,

        # ITC decay (disabled for HPO)
        "itc_decay_start_epoch": 0,
        "itc_decay_epochs": 0,
        "itc_decay_ratio": 0.0,
        "itc_label_smoothing": 0.0,

        # Hard negatives — DISABLED during HPO (3x per-step cost)
        "hard_neg_start_epoch": 999,
        "hard_neg_ramp_epochs": 0,
        "hard_neg_top_k": 5,

        # Data augmentation — DISABLED during HPO
        "gw_aug_noise": 0.0,
        "gw_aug_jitter": 0.0,
        "gw_aug_dropout": 0.0,
        "opt_aug_noise": 0.0,
        "opt_aug_time_jitter": 0.0,
        "opt_aug_dropout": 0.0,
        "opt_aug_band_dropout": 0.0,

        # Checkpoint
        "ckpt_path": trial_ckpt,
        "resume": None,
        "pretrained": None,
        "skip_epoch_checkpoints": True,

        # HPO metadata
        "hpo_trial_number": trial.number,
    }

    return config


def run_trial_subprocess(config, trial_number, output_dir):
    """Run a single training trial as a subprocess and return best val_loss."""

    config_dir = os.path.join(output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    config_path = os.path.join(config_dir, f"trial_{trial_number}.json")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Build command — call ALBEF_train.py directly (not via shell script)
    # The shell script is for SLURM submission; here we run directly
    cmd = [sys.executable, "-u", TRAIN_SCRIPT]

    # Convert config dict to CLI args
    for key, value in config.items():
        if key == "hpo_trial_number":
            continue  # Not a CLI arg for ALBEF_train.py (yet)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        cmd.append(f"--{key}")
        cmd.append(str(value))

    print(f"\n{'='*60}")
    print(f"Trial {trial_number}: Starting training")
    print(f"  batch_size={config['batch_size']}, lr={config['lr']:.2e}, "
          f"enc_dim={config['enc_dim']}, proj_dim={config['proj_dim']}")
    print(f"  steps_per_epoch={config['steps_per_epoch']}, "
          f"val_steps_per_epoch={config['val_steps_per_epoch']}")
    print(f"{'='*60}")

    # Stream subprocess output to log file + print epoch summaries
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
            # Keep last 20 lines for error reporting
            tail_lines.append(line.rstrip())
            if len(tail_lines) > 20:
                tail_lines.pop(0)
            # Print epoch progress lines to parent stdout
            if ("Epoch " in line and "Complete" in line) or \
               "Val Avg Total" in line or \
               "Retrieval:" in line or \
               "Early stopping" in line or \
               "Saved best" in line:
                print(f"  [T{trial_number}] {line.rstrip()}", flush=True)
        proc.wait()
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"Trial {trial_number}: FAILED (exit code {proc.returncode})")
        for line in tail_lines:
            print(f"  {line}")
        raise optuna.TrialPruned(f"Training failed with exit code {proc.returncode}")

    print(f"Trial {trial_number}: Completed in {elapsed:.0f}s")

    # Read result from trial_results.json written by ALBEF_train.py
    result_path = os.path.join(config["ckpt_path"], "ALBEF", "trial_results.json")
    if not os.path.exists(result_path):
        # Fallback: try to parse from best checkpoint
        best_ckpt = os.path.join(config["ckpt_path"], "ALBEF", "albef_best.pth")
        if os.path.exists(best_ckpt):
            import torch
            ckpt = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            return ckpt.get("loss", float("inf"))
        print(f"Trial {trial_number}: No results found")
        return float("inf")

    with open(result_path) as f:
        results = json.load(f)

    return results


def objective(trial, args):
    """Optuna objective: sample hyperparams, run training, return val_loss."""

    config = build_trial_config(trial, args)
    results = run_trial_subprocess(config, trial.number, args.output_dir)

    if isinstance(results, float):
        # Fallback: only got val_loss from checkpoint
        val_loss = results
    else:
        val_loss = results.get("best_val_loss", float("inf"))

        # Store secondary metrics as user attributes
        trial.set_user_attr("best_epoch", results.get("best_epoch", -1))
        trial.set_user_attr("final_epoch", results.get("final_epoch", -1))
        trial.set_user_attr("val_recall_at_1", results.get("val_recall_at_1", 0))
        trial.set_user_attr("val_auroc", results.get("val_auroc", 0))
        trial.set_user_attr("val_auprc", results.get("val_auprc", 0))
        trial.set_user_attr("val_itc_acc", results.get("val_itc_acc", 0))

    print(f"Trial {trial.number}: val_loss = {val_loss:.6f}")
    return val_loss


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO for ALBEF training")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of Optuna trials")
    parser.add_argument("--study_name", type=str, default="albef_hpo_v1",
                        help="Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///hpo_results/optuna_study.db)")
    parser.add_argument("--output_dir", type=str, default="hpo_results",
                        help="Directory for HPO configs, results, and analysis")
    parser.add_argument("--epochs_per_trial", type=int, default=30,
                        help="Max epochs per trial")
    parser.add_argument("--n_startup_trials", type=int, default=10,
                        help="Random trials before TPE kicks in")

    # Data paths
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--neg_data_path", type=str, default=DEFAULT_NEG_DATA_PATH)
    parser.add_argument("--neg_group", type=str, default=DEFAULT_NEG_GROUP)

    # Resource config
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--stage_to_jobfs", action="store_true")
    parser.add_argument("--cache_in_memory", action="store_true")

    args = parser.parse_args()

    # Resolve output_dir to absolute path
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Default storage: SQLite in output_dir
    if args.storage is None:
        db_path = os.path.join(args.output_dir, "optuna_study.db")
        args.storage = f"sqlite:///{db_path}"

    print(f"Study: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Output: {args.output_dir}")
    print(f"Trials: {args.n_trials} ({args.epochs_per_trial} epochs each)")
    print(f"Data: {args.data_path}")
    print()

    # Create Optuna study
    sampler = TPESampler(
        n_startup_trials=args.n_startup_trials,
        seed=42,
    )
    pruner = MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=0,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("HPO COMPLETE")
    print("=" * 60)
    print(f"Total trials: {len(study.trials)}")
    print(f"Completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"Pruned/Failed: {len([t for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE])}")

    if study.best_trial:
        best = study.best_trial
        print(f"\nBest trial: #{best.number}")
        print(f"  val_loss: {best.value:.6f}")
        print(f"  Parameters:")
        for key, value in best.params.items():
            print(f"    {key}: {value}")

        # Fixed params (not tuned)
        print(f"\n  Fixed:")
        print(f"    batch_size: 1024")
        print(f"    steps_per_epoch: {1_024_000 // 1024}")
        print(f"    val_steps_per_epoch: {25600 // 1024}")

        # Save best config
        best_config_path = os.path.join(args.output_dir, "best_config.json")
        config_path = os.path.join(args.output_dir, "configs", f"trial_{best.number}.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                best_config = json.load(f)
            with open(best_config_path, "w") as f:
                json.dump(best_config, f, indent=2)
            print(f"\n  Best config saved to: {best_config_path}")


if __name__ == "__main__":
    main()
