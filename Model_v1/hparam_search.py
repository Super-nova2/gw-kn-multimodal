#!/usr/bin/env python3
"""
Hyperparameter optimization for GW-Optical Fusion Model using Optuna.

This script performs Bayesian optimization (TPE) over 19 hyperparameters:
- Training: lr, weight_decay, batch_size, grad_clip_norm, label_smoothing
- Architecture: enc_dim, ref_dim, n_ref, fusion_attn_dim, fusion_hidden_dim
- Regularization: fusion_dropout, gw_dropout, opt_dropout
- Loss weights: pos_weight, neg_weight, extra_neg_weight
- Data sampling: samples_per_gw, neg_gw_ratio
- LR scheduler: lr_scheduler, warmup_epochs, min_lr

Usage:
    python hparam_search.py --data_path /path/to/data.h5 --n_trials 100

    # Resume a previous study
    python hparam_search.py --data_path /path/to/data.h5 --n_trials 50 --resume
"""

import argparse
import json
import os
from datetime import datetime

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from train_v1 import train_with_config


def create_objective(data_path, neg_data_path, neg_group, max_epochs):
    """Create an Optuna objective function with the given data paths."""

    def objective(trial):
        config = {
            # Training hyperparameters
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "grad_clip_norm": trial.suggest_float("grad_clip_norm", 0.5, 2.0),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),

            # Architecture hyperparameters
            "enc_dim": trial.suggest_categorical("enc_dim", [64, 128, 256, 512]),
            "ref_dim": trial.suggest_categorical("ref_dim", [32, 64, 128]),
            "n_ref": trial.suggest_categorical("n_ref", [32, 64, 128]),
            "fusion_attn_dim": trial.suggest_categorical("fusion_attn_dim", [64, 128, 256]),
            "fusion_hidden_dim": trial.suggest_categorical("fusion_hidden_dim", [128, 256, 512]),

            # Regularization hyperparameters
            "fusion_dropout": trial.suggest_float("fusion_dropout", 0.0, 0.5),
            "gw_dropout": trial.suggest_float("gw_dropout", 0.0, 0.5),
            "opt_dropout": trial.suggest_float("opt_dropout", 0.0, 0.5),

            # Loss weighting hyperparameters
            "pos_weight": trial.suggest_float("pos_weight", 0.5, 2.0),
            "neg_weight": trial.suggest_float("neg_weight", 0.25, 1.0),
            "extra_neg_weight": trial.suggest_float("extra_neg_weight", 0.25, 1.0),

            # Data sampling hyperparameters
            "samples_per_gw": trial.suggest_categorical("samples_per_gw", [4, 8, 16]),
            "neg_gw_ratio": trial.suggest_float("neg_gw_ratio", 0.1, 0.3),

            # LR scheduler (conditional)
            "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["none", "cosine"]),

            # Fixed parameters
            "use_neg_gw": True,
            "val_split": 0.2,
            "ref_start": -0.3,
            "ref_end": 0.6,
            "optical_dim": 6,
        }

        # Conditional: warmup only if using cosine scheduler
        if config["lr_scheduler"] == "cosine":
            config["warmup_epochs"] = trial.suggest_int("warmup_epochs", 0, 10)
            config["min_lr"] = trial.suggest_float("min_lr", 0.0, config["lr"] / 10)
        else:
            config["warmup_epochs"] = 0
            config["min_lr"] = 0.0

        val_loss = train_with_config(
            config=config,
            data_path=data_path,
            neg_data_path=neg_data_path,
            neg_group=neg_group,
            max_epochs=max_epochs,
            trial=trial,
            verbose=False
        )
        return val_loss

    return objective


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data HDF5 file")
    parser.add_argument("--neg_data_path", type=str, default=None,
                        help="Path to negative samples HDF5 file")
    parser.add_argument("--neg_group", type=str, default="events/optical_data",
                        help="Group name in negative HDF5 file")
    parser.add_argument("--n_trials", type=int, default=100,
                        help="Number of optimization trials")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum epochs per trial (reduced for HPO)")
    parser.add_argument("--study_name", type=str, default="gw_optical_fusion_hpo",
                        help="Name of the Optuna study")
    parser.add_argument("--storage", type=str, default=None,
                        help="Database URL for study persistence (e.g., sqlite:///hpo.db)")
    parser.add_argument("--output_dir", type=str, default="./hpo_results",
                        help="Directory to save results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an existing study")
    parser.add_argument("--n_startup_trials", type=int, default=20,
                        help="Number of random trials before TPE kicks in")
    parser.add_argument("--pruner_warmup", type=int, default=10,
                        help="Number of epochs before pruning can occur")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Timeout in seconds for the entire study")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up storage
    if args.storage is None:
        db_path = os.path.join(args.output_dir, f"{args.study_name}.db")
        storage = f"sqlite:///{db_path}"
    else:
        storage = args.storage

    # Create sampler and pruner
    sampler = TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,  # Model parameter correlations
    )
    pruner = MedianPruner(
        n_startup_trials=args.n_startup_trials // 2,
        n_warmup_steps=args.pruner_warmup,
        interval_steps=1,
    )

    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=args.resume
    )

    print(f"Study name: {args.study_name}")
    print(f"Storage: {storage}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Max epochs per trial: {args.max_epochs}")
    if args.resume and len(study.trials) > 0:
        print(f"Resuming from {len(study.trials)} existing trials")
        print(f"Current best value: {study.best_value:.6f}")

    # Create objective function
    objective = create_objective(
        data_path=args.data_path,
        neg_data_path=args.neg_data_path,
        neg_group=args.neg_group,
        max_epochs=args.max_epochs
    )

    # Run optimization
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
        gc_after_trial=True,  # Clean up GPU memory
    )

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best validation loss: {study.best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save best params to JSON
    best_params_path = os.path.join(args.output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    print(f"\nBest parameters saved to: {best_params_path}")

    # Also save to args directory for easy use with train_v1.py
    args_best_path = os.path.join(os.path.dirname(__file__), "args", "best_params.json")
    os.makedirs(os.path.dirname(args_best_path), exist_ok=True)
    with open(args_best_path, "w") as f:
        # Create a complete config by merging with defaults
        full_config = {
            "data_path": args.data_path,
            "neg_data_path": args.neg_data_path,
            "neg_group": args.neg_group,
            "ckpt_path": "/fred/oz016/bgao_kn/data/model/checkpoints_v1_optimized",
            "epochs": 150,
            "use_neg_gw": True,
            "val_split": 0.2,
            "val_batch_size": 32,
            "num_workers": 8,
            "pin_memory": 1,
            "persistent_workers": 1,
            "prefetch_factor": 4,
            "cache_in_memory": True,
            "ref_start": -0.3,
            "ref_end": 0.6,
            "optical_dim": 6,
            "early_stop_patience": 20,
            "early_stop_min_delta": 0.001,
            "log_every": 50,
            **study.best_trial.params
        }
        json.dump(full_config, f, indent=2)
    print(f"Full config saved to: {args_best_path}")

    # Save study statistics
    stats_path = os.path.join(args.output_dir, f"study_stats_{timestamp}.json")
    stats = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_trial.value,
        "best_params": study.best_trial.params,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Study statistics saved to: {stats_path}")

    # Generate importance analysis if enough trials
    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= 10:
        try:
            importance = optuna.importance.get_param_importances(study)
            print("\nParameter importance:")
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {param}: {imp:.4f}")

            importance_path = os.path.join(args.output_dir, f"param_importance_{timestamp}.json")
            with open(importance_path, "w") as f:
                json.dump(importance, f, indent=2)
        except Exception as e:
            print(f"Could not compute parameter importance: {e}")

    return study


if __name__ == "__main__":
    main()
