#!/usr/bin/env python3
"""
Post-HPO analysis: parameter importance, optimization history,
parallel coordinates, and top-K config export.

Usage:
    python hpo_analyze.py --study_name albef_hpo_v1 \
        --storage sqlite:///hpo_results/optuna_study.db \
        --output_dir hpo_results/analysis --top_k 5
"""

import argparse
import json
import os

import optuna


def print_study_summary(study):
    """Print overall study statistics."""
    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

    print("=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    print(f"Study name: {study.study_name}")
    print(f"Total trials: {len(trials)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Pruned:    {len(pruned)}")
    print(f"  Failed:    {len(failed)}")

    if not completed:
        print("\nNo completed trials. Nothing to analyze.")
        return False

    values = [t.value for t in completed]
    print(f"\nval_loss statistics:")
    print(f"  Best:   {min(values):.6f}")
    print(f"  Worst:  {max(values):.6f}")
    print(f"  Median: {sorted(values)[len(values)//2]:.6f}")
    print(f"  Mean:   {sum(values)/len(values):.6f}")

    return True


def print_best_trial(study):
    """Print best trial details."""
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST TRIAL: #{best.number}")
    print(f"{'='*60}")
    print(f"val_loss: {best.value:.6f}")

    print("\nHyperparameters:")
    for key, value in sorted(best.params.items()):
        if isinstance(value, float):
            print(f"  {key:30s} = {value:.6f}")
        else:
            print(f"  {key:30s} = {value}")

    # Derived params
    bs = best.params.get("batch_size", 1024)
    print(f"\nDerived:")
    print(f"  {'steps_per_epoch':30s} = {1_000_000 // bs}")
    print(f"  {'val_steps_per_epoch':30s} = {3 * 4000 // bs}")

    # User attributes (secondary metrics)
    if best.user_attrs:
        print(f"\nSecondary metrics:")
        for key, value in sorted(best.user_attrs.items()):
            if isinstance(value, float):
                print(f"  {key:30s} = {value:.6f}")
            else:
                print(f"  {key:30s} = {value}")


def print_top_k(study, k=5):
    """Print top-K trials."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(completed, key=lambda t: t.value)[:k]

    print(f"\n{'='*60}")
    print(f"TOP {k} TRIALS")
    print(f"{'='*60}")

    for rank, trial in enumerate(ranked, 1):
        bs = trial.params.get("batch_size", 1024)
        enc = trial.params.get("enc_dim", 64)
        lr = trial.params.get("lr", 0)
        r1 = trial.user_attrs.get("val_recall_at_1", "N/A")
        auroc = trial.user_attrs.get("val_auroc", "N/A")

        print(f"\n  Rank {rank}: Trial #{trial.number}")
        print(f"    val_loss={trial.value:.6f}  "
              f"R@1={r1 if isinstance(r1, str) else f'{r1:.4f}'}  "
              f"AUROC={auroc if isinstance(auroc, str) else f'{auroc:.4f}'}")
        print(f"    batch_size={bs}, enc_dim={enc}, lr={lr:.2e}, "
              f"steps_per_epoch={1_000_000 // bs}")


def generate_plots(study, output_dir):
    """Generate Optuna visualization plots."""
    os.makedirs(output_dir, exist_ok=True)

    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )
    except ImportError:
        print("\nWarning: optuna.visualization requires plotly. Skipping plots.")
        print("Install with: pip install plotly kaleido")
        return

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 3:
        print("\nToo few completed trials for meaningful plots. Skipping.")
        return

    plots = {}

    # Optimization history
    try:
        fig = plot_optimization_history(study)
        plots["optimization_history"] = fig
    except Exception as e:
        print(f"  Warning: optimization_history failed: {e}")

    # Parameter importances (needs >= 3 completed trials)
    try:
        fig = plot_param_importances(study)
        plots["param_importances"] = fig
    except Exception as e:
        print(f"  Warning: param_importances failed: {e}")

    # Parallel coordinate (top 10 trials)
    try:
        fig = plot_parallel_coordinate(study)
        plots["parallel_coordinate"] = fig
    except Exception as e:
        print(f"  Warning: parallel_coordinate failed: {e}")

    # Slice plots for key params
    key_params = ["batch_size", "lr", "enc_dim", "supcon_temperature",
                  "gw_dropout", "itc_weight", "cls_weight"]
    available_params = [p for p in key_params if p in study.best_params]
    if available_params:
        try:
            fig = plot_slice(study, params=available_params)
            plots["slice_plots"] = fig
        except Exception as e:
            print(f"  Warning: slice_plots failed: {e}")

    # Save plots
    try:
        import plotly.io as pio
        for name, fig in plots.items():
            # Save as HTML (interactive)
            html_path = os.path.join(output_dir, f"{name}.html")
            pio.write_html(fig, html_path)
            # Save as PNG (static)
            try:
                png_path = os.path.join(output_dir, f"{name}.png")
                pio.write_image(fig, png_path, width=1200, height=800)
            except Exception:
                pass  # kaleido may not be installed
        print(f"\nPlots saved to: {output_dir}")
    except ImportError:
        print("  Warning: plotly.io not available. Plots not saved.")


def export_top_configs(study, output_dir, config_source_dir, top_k=5):
    """Export top-K trial configs as ready-to-use JSON for full retraining."""
    export_dir = os.path.join(output_dir, "top_configs")
    os.makedirs(export_dir, exist_ok=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(completed, key=lambda t: t.value)[:top_k]

    exported = []
    for rank, trial in enumerate(ranked, 1):
        # Try to load the original trial config
        source = os.path.join(config_source_dir, f"trial_{trial.number}.json")
        if os.path.exists(source):
            with open(source) as f:
                config = json.load(f)
        else:
            # Reconstruct from trial params
            config = dict(trial.params)

        # Override for full retraining
        config["epochs"] = 100
        config["early_stop_patience"] = 25
        config["early_stop_min_delta"] = 0.005
        # Remove HPO metadata
        config.pop("hpo_trial_number", None)

        # Update checkpoint path for full retrain
        config["ckpt_path"] = config.get("ckpt_path", "").replace(
            f"trial_{trial.number}", f"retrain_rank{rank}"
        )

        out_path = os.path.join(export_dir, f"rank{rank}_trial{trial.number}.json")
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)

        exported.append(out_path)
        print(f"  Rank {rank} (trial #{trial.number}, val_loss={trial.value:.6f}): {out_path}")

    return exported


def export_results_csv(study, output_dir):
    """Export all trial results to CSV for external analysis."""
    csv_path = os.path.join(output_dir, "all_trials.csv")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return

    # Collect all param keys and user_attr keys
    param_keys = sorted(set(k for t in completed for k in t.params.keys()))
    attr_keys = sorted(set(k for t in completed for k in t.user_attrs.keys()))

    header = ["trial", "val_loss"] + param_keys + attr_keys
    rows = []
    for t in completed:
        row = [str(t.number), f"{t.value:.6f}"]
        for k in param_keys:
            v = t.params.get(k, "")
            row.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        for k in attr_keys:
            v = t.user_attrs.get(k, "")
            row.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        rows.append(row)

    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")

    print(f"\nAll results exported to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Optuna HPO results")
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True,
                        help="Optuna storage URL (e.g., sqlite:///hpo_results/optuna_study.db)")
    parser.add_argument("--output_dir", type=str, default="hpo_results/analysis")
    parser.add_argument("--config_dir", type=str, default=None,
                        help="Directory with trial configs (default: hpo_results/configs)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating plots")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.config_dir is None:
        # Infer from storage path
        storage_path = args.storage.replace("sqlite:///", "")
        args.config_dir = os.path.join(os.path.dirname(storage_path), "configs")

    # Load study
    study = optuna.load_study(
        study_name=args.study_name,
        storage=args.storage,
    )

    # Print summaries
    has_results = print_study_summary(study)
    if not has_results:
        return

    print_best_trial(study)
    print_top_k(study, args.top_k)

    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(study, args.output_dir)

    # Export top configs for retraining
    print(f"\nExporting top-{args.top_k} configs for full retraining:")
    export_top_configs(study, args.output_dir, args.config_dir, args.top_k)

    # Export CSV
    export_results_csv(study, args.output_dir)


if __name__ == "__main__":
    main()
