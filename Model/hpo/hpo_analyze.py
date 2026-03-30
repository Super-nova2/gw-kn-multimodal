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

METRIC_SHORT_NAMES = {
    "val_auroc": "AUROC",
    "val_auprc": "AUPRC",
    "val_recall_at_1": "R@1",
    "val_recall_at_5": "R@5",
    "val_mrr": "MRR",
    "val_itc_acc": "ITC_ACC",
}

PREFERRED_SLICE_PARAMS = [
    # Current v2 HPO defaults
    "lr",
    "weight_decay",
    "warmup_epochs",
    "enc_dim",
    "proj_dim",
    "ref_shared_dim",
    "cls_start_epoch",
    "time_compat_weight",
    "semi_hard_margin",
    "hardneg_min_candidates",
    "augment_enable",
    # Backward-compatible legacy params
    "samples_per_gw",
    "itc_weight",
    "cls_weight",
]


def _parse_dict_attr(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _fmt_attr_value(value, precision=4):
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    if isinstance(value, int):
        return str(value)
    return "N/A"


def get_objective_context(study):
    metric = study.user_attrs.get("objective_metric", "objective")
    direction = study.user_attrs.get("objective_direction")
    if direction is None:
        direction = str(study.direction).split(".")[-1].lower()
    direction = direction.lower()
    if direction not in {"maximize", "minimize"}:
        direction = "minimize"
    weights = _parse_dict_attr(study.user_attrs.get("objective_weights", {}))
    min_metrics = _parse_dict_attr(study.user_attrs.get("objective_min_metrics", {}))

    # Backward compatibility for older studies that only had the legacy objective name.
    if not weights and metric == "combined_auroc_g2o_r5":
        weights = {"val_auroc": 0.5, "val_recall_at_5": 0.5}

    return metric, direction, weights, min_metrics


def rank_trials(completed, direction):
    reverse = direction == "maximize"
    return sorted(completed, key=lambda t: t.value, reverse=reverse)


def print_study_summary(study):
    """Print overall study statistics."""
    metric, direction, weights, min_metrics = get_objective_context(study)

    trials = study.trials
    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

    print("=" * 60)
    print("STUDY SUMMARY")
    print("=" * 60)
    print(f"Study name: {study.study_name}")
    print(f"Objective: {metric} ({direction})")
    if weights:
        total = sum(weights.values())
        if total > 0:
            formula = " + ".join(
                f"{(w / total):.3f}*{k}" for k, w in weights.items()
            )
            print(f"Objective formula: {formula}")
    if min_metrics:
        print(f"Objective minimum metrics: {min_metrics}")
    print(f"Total trials: {len(trials)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Pruned:    {len(pruned)}")
    print(f"  Failed:    {len(failed)}")

    if not completed:
        print("\nNo completed trials. Nothing to analyze.")
        return False

    values = [t.value for t in completed]
    best = max(values) if direction == "maximize" else min(values)
    worst = min(values) if direction == "maximize" else max(values)

    print(f"\n{metric} statistics:")
    print(f"  Best:   {best:.6f}")
    print(f"  Worst:  {worst:.6f}")
    print(f"  Median: {sorted(values)[len(values)//2]:.6f}")
    print(f"  Mean:   {sum(values)/len(values):.6f}")

    return True


def print_best_trial(study):
    """Print best trial details."""
    metric, _, _, _ = get_objective_context(study)
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"BEST TRIAL: #{best.number}")
    print(f"{'='*60}")
    print(f"{metric}: {best.value:.6f}")

    print("\nHyperparameters:")
    for key, value in sorted(best.params.items()):
        if isinstance(value, float):
            print(f"  {key:30s} = {value:.6f}")
        else:
            print(f"  {key:30s} = {value}")

    # User attributes (secondary metrics)
    if best.user_attrs:
        print("\nSecondary metrics:")
        for key, value in sorted(best.user_attrs.items()):
            if isinstance(value, float):
                print(f"  {key:30s} = {value:.6f}")
            else:
                print(f"  {key:30s} = {value}")


def print_top_k(study, k=5):
    """Print top-K trials."""
    metric, direction, weights, _ = get_objective_context(study)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = rank_trials(completed, direction)[:k]

    print(f"\n{'='*60}")
    print(f"TOP {k} TRIALS")
    print(f"{'='*60}")

    for rank, trial in enumerate(ranked, 1):
        enc = trial.params.get("enc_dim", "N/A")
        lr = trial.params.get("lr", None)
        metric_keys = list(weights.keys()) if weights else ["val_recall_at_5", "val_auroc", "val_auprc"]
        metric_parts = []
        for key in metric_keys:
            label = METRIC_SHORT_NAMES.get(key, key)
            metric_parts.append(f"{label}={_fmt_attr_value(trial.user_attrs.get(key, 'N/A'))}")

        print(f"\n  Rank {rank}: Trial #{trial.number}")
        print(
            f"    {metric}={trial.value:.6f}  "
            + "  ".join(metric_parts)
        )
        if lr is None:
            print(f"    enc_dim={enc}")
        else:
            print(f"    enc_dim={enc}, lr={lr:.2e}")


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

    try:
        plots["optimization_history"] = plot_optimization_history(study)
    except Exception as e:
        print(f"  Warning: optimization_history failed: {e}")

    try:
        plots["param_importances"] = plot_param_importances(study)
    except Exception as e:
        print(f"  Warning: param_importances failed: {e}")

    try:
        plots["parallel_coordinate"] = plot_parallel_coordinate(study)
    except Exception as e:
        print(f"  Warning: parallel_coordinate failed: {e}")

    available_params = [p for p in PREFERRED_SLICE_PARAMS if p in study.best_params]
    if available_params:
        try:
            plots["slice_plots"] = plot_slice(study, params=available_params)
        except Exception as e:
            print(f"  Warning: slice_plots failed: {e}")

    try:
        import plotly.io as pio

        for name, fig in plots.items():
            html_path = os.path.join(output_dir, f"{name}.html")
            pio.write_html(fig, html_path)
            try:
                png_path = os.path.join(output_dir, f"{name}.png")
                pio.write_image(fig, png_path, width=1200, height=800)
            except Exception:
                pass
        print(f"\nPlots saved to: {output_dir}")
    except ImportError:
        print("  Warning: plotly.io not available. Plots not saved.")


def export_top_configs(study, output_dir, config_source_dir, top_k=5):
    """Export top-K trial configs as ready-to-use JSON for full retraining."""
    metric, direction, _, _ = get_objective_context(study)

    export_dir = os.path.join(output_dir, "top_configs")
    os.makedirs(export_dir, exist_ok=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = rank_trials(completed, direction)[:top_k]

    exported = []
    for rank, trial in enumerate(ranked, 1):
        source = os.path.join(config_source_dir, f"trial_{trial.number}.json")
        if os.path.exists(source):
            with open(source) as f:
                config = json.load(f)
        else:
            config = dict(trial.params)

        config["epochs"] = 100
        config["early_stop_patience"] = 25
        config["early_stop_min_delta"] = 0.005
        config.pop("hpo_trial_number", None)

        config["ckpt_path"] = config.get("ckpt_path", "").replace(
            f"trial_{trial.number}", f"retrain_rank{rank}"
        )

        out_path = os.path.join(export_dir, f"rank{rank}_trial{trial.number}.json")
        with open(out_path, "w") as f:
            json.dump(config, f, indent=2)

        exported.append(out_path)
        print(f"  Rank {rank} (trial #{trial.number}, {metric}={trial.value:.6f}): {out_path}")

    return exported


def export_results_csv(study, output_dir):
    """Export all trial results to CSV for external analysis."""
    metric, _, _, _ = get_objective_context(study)

    csv_path = os.path.join(output_dir, "all_trials.csv")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return

    param_keys = sorted(set(k for t in completed for k in t.params.keys()))
    attr_keys = sorted(set(k for t in completed for k in t.user_attrs.keys()))

    header = ["trial", metric] + param_keys + attr_keys
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
    parser.add_argument(
        "--storage",
        type=str,
        required=True,
        help="Optuna storage URL (e.g., sqlite:///hpo_results/optuna_study.db)",
    )
    parser.add_argument("--output_dir", type=str, default="hpo_results/analysis")
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Directory with trial configs (default: hpo_results/configs)",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.config_dir is None:
        storage_path = args.storage.replace("sqlite:///", "")
        args.config_dir = os.path.join(os.path.dirname(storage_path), "configs")

    try:
        study = optuna.load_study(
            study_name=args.study_name,
            storage=args.storage,
        )
    except KeyError as exc:
        raise SystemExit(
            f"Study '{args.study_name}' not found in storage '{args.storage}'."
        ) from exc

    has_results = print_study_summary(study)
    if not has_results:
        return

    print_best_trial(study)
    print_top_k(study, args.top_k)

    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(study, args.output_dir)

    print(f"\nExporting top-{args.top_k} configs for full retraining:")
    export_top_configs(study, args.output_dir, args.config_dir, args.top_k)

    export_results_csv(study, args.output_dir)


if __name__ == "__main__":
    main()
