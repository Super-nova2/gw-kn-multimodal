"""GW-level statistics and publication plots for mixed retrieval evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mixed_retrieval import MIXED_SCOPES, allocate_mixed_negative_counts
from plot_style import apply_mnras_style

METRIC_COLUMNS = ("recall_at_1", "recall_at_5", "recall_at_10", "mrr")
PLOT_METRICS = (
    ("recall_at_1", "R@1"),
    ("recall_at_10", "R@10"),
    ("mrr", "MRR"),
)
MODEL_STYLES = {
    "Mixed Gallery v1": {"color": "#1769AA", "marker": "o", "linestyle": "-"},
    "Default MAGIKS": {"color": "#E07A1F", "marker": "s", "linestyle": "--"},
    "Optical-only": {"color": "#73777B", "marker": "D", "linestyle": ":"},
}
FALLBACK_STYLES = (
    {"color": "#1769AA", "marker": "o", "linestyle": "-"},
    {"color": "#E07A1F", "marker": "s", "linestyle": "--"},
    {"color": "#73777B", "marker": "D", "linestyle": ":"},
)
RANDOM_STYLE = {"color": "#333333", "linestyle": ":", "linewidth": 1.2}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def random_baseline(n_candidates: int) -> dict[str, float]:
    """Return exact expected metrics for a uniformly random ranking."""
    n = int(n_candidates)
    return {
        "recall_at_1": 1.0 / n,
        "recall_at_5": min(5, n) / n,
        "recall_at_10": min(10, n) / n,
        "mrr": float(np.sum(1.0 / np.arange(1, n + 1))) / n,
    }


def collapse_trials(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Average repeated gallery trials within each GW before inference."""
    required = {
        "model",
        "condition",
        "scope",
        "gallery_size",
        "trial",
        "gw_id",
        "source",
        *METRIC_COLUMNS,
    }
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"Outcome table is missing columns: {sorted(missing)}")
    group_columns = [
        "model",
        "condition",
        "scope",
        "gallery_size",
        "gw_id",
        "source",
    ]
    for column in (
        "redshift",
        "redshift_bin_index",
        "redshift_bin_label",
    ):
        if column in outcomes.columns:
            group_columns.append(column)
    value_columns = [
        column
        for column in (*METRIC_COLUMNS, "rank", "kn_margin", "nonkn_margin")
        if column in outcomes.columns
    ]
    collapsed = (
        outcomes.groupby(group_columns, sort=True, as_index=False, dropna=False)[
            value_columns
        ]
        .mean()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    collapsed["n_trials"] = (
        outcomes.groupby(group_columns, sort=True, dropna=False)["trial"]
        .nunique()
        .to_numpy()
    )
    return collapsed


def _bootstrap_source_means(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    point: dict[str, np.ndarray] = {}
    draws: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for source_index, (source, source_frame) in enumerate(
        frame.groupby("source", sort=True)
    ):
        ordered = source_frame.sort_values("gw_id")
        if ordered["gw_id"].duplicated().any():
            raise ValueError("Bootstrap input must contain one row per GW")
        values = ordered[list(value_columns)].to_numpy(dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Bootstrap input contains empty or non-finite values")
        point[str(source)] = values.mean(axis=0)
        counts[str(source)] = int(values.shape[0])
        rng = np.random.default_rng(int(seed) + 104729 * int(source_index))
        sampled_means = np.empty(
            (int(n_bootstrap), len(value_columns)), dtype=np.float64
        )
        for start in range(0, int(n_bootstrap), 256):
            stop = min(start + 256, int(n_bootstrap))
            indices = rng.integers(
                0, values.shape[0], size=(stop - start, values.shape[0])
            )
            sampled_means[start:stop] = values[indices].mean(axis=1)
        draws[str(source)] = sampled_means
    return point, draws, counts


def _summarize_source_aggregations(
    frame: pd.DataFrame,
    *,
    value_columns: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    point, draws, counts = _bootstrap_source_means(
        frame,
        value_columns=value_columns,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    sources = sorted(point)
    total = sum(counts.values())
    pooled_point = sum(point[source] * counts[source] for source in sources) / total
    pooled_draws = sum(draws[source] * counts[source] for source in sources) / total
    macro_point = np.mean([point[source] for source in sources], axis=0)
    macro_draws = np.mean([draws[source] for source in sources], axis=0)

    variants: list[tuple[str, np.ndarray, np.ndarray, int]] = [
        ("pooled", pooled_point, pooled_draws, total),
        ("source_macro", macro_point, macro_draws, total),
    ]
    variants.extend(
        (source, point[source], draws[source], counts[source]) for source in sources
    )
    rows: list[dict[str, Any]] = []
    for aggregation, estimate, bootstrap_draws, n_unique_gw in variants:
        for index, metric in enumerate(value_columns):
            rows.append(
                {
                    "metric": str(metric),
                    "source_aggregation": aggregation,
                    "estimate": float(estimate[index]),
                    "ci_low": float(np.quantile(bootstrap_draws[:, index], 0.025)),
                    "ci_high": float(np.quantile(bootstrap_draws[:, index], 0.975)),
                    "bootstrap_unit": "gw_id_after_trial_mean",
                    "n_unique_gw": int(n_unique_gw),
                    "n_bootstrap": int(n_bootstrap),
                    "bootstrap_seed": int(seed),
                }
            )
    return rows


def bootstrap_metric_intervals(
    collapsed: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
    extra_group_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Estimate model metric intervals using source-stratified GW resampling."""
    group_columns = [
        "model",
        "condition",
        "scope",
        "gallery_size",
        *extra_group_columns,
    ]
    rows: list[dict[str, Any]] = []
    for group_key, frame in collapsed.groupby(group_columns, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        common = dict(zip(group_columns, group_key))
        summaries = _summarize_source_aggregations(
            frame,
            value_columns=METRIC_COLUMNS,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.extend({**common, **row} for row in summaries)
    return pd.DataFrame(rows)


def paired_bootstrap(
    collapsed: pd.DataFrame,
    *,
    new_name: str,
    baseline_name: str,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Compute paired new-minus-baseline intervals at the GW level."""
    key_columns = ["condition", "scope", "gallery_size", "gw_id", "source"]
    new = collapsed[collapsed["model"].eq(new_name)].set_index(key_columns)
    baseline = collapsed[collapsed["model"].eq(baseline_name)].set_index(key_columns)
    if new.index.has_duplicates or baseline.index.has_duplicates:
        raise ValueError("Paired bootstrap expects one row per model and GW")
    common = new.index.intersection(baseline.index)
    if common.empty:
        return pd.DataFrame()
    delta = (
        new.loc[common, list(METRIC_COLUMNS)].sort_index()
        - baseline.loc[common, list(METRIC_COLUMNS)].sort_index()
    ).reset_index()

    rows: list[dict[str, Any]] = []
    for group_key, frame in delta.groupby(
        ["condition", "scope", "gallery_size"], sort=True
    ):
        condition, scope, gallery_size = group_key
        summaries = _summarize_source_aggregations(
            frame,
            value_columns=METRIC_COLUMNS,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        for row in summaries:
            rows.append(
                {
                    "condition": condition,
                    "scope": scope,
                    "gallery_size": int(gallery_size),
                    "source": row["source_aggregation"],
                    "source_aggregation": row["source_aggregation"],
                    "metric": row["metric"],
                    "new_model": new_name,
                    "baseline_model": baseline_name,
                    "mean_delta": row["estimate"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "n_pairs": row["n_unique_gw"],
                    "n_unique_gw": row["n_unique_gw"],
                    "bootstrap_unit": row["bootstrap_unit"],
                    "n_bootstrap": row["n_bootstrap"],
                    "bootstrap_seed": row["bootstrap_seed"],
                }
            )
    return pd.DataFrame(rows)


def training_effect_summary(
    collapsed: pd.DataFrame,
    *,
    new_name: str,
    baseline_name: str,
    condition: str,
    scope: str,
    primary_metric: Mapping[str, Any],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate the checkpoint-selection score as one paired primary endpoint."""
    sizes = {
        int(key): float(value)
        for key, value in primary_metric["gallery_size_weights"].items()
    }
    metric_weights = {
        str(key): float(value)
        for key, value in primary_metric["metric_weights"].items()
    }
    selected = collapsed[
        collapsed["condition"].eq(condition)
        & collapsed["scope"].eq(scope)
        & collapsed["gallery_size"].isin(sizes)
        & collapsed["model"].isin([new_name, baseline_name])
    ]
    keys = ["condition", "scope", "gallery_size", "gw_id", "source"]
    new = selected[selected["model"].eq(new_name)].set_index(keys)
    baseline = selected[selected["model"].eq(baseline_name)].set_index(keys)
    common = new.index.intersection(baseline.index)
    if common.empty:
        raise ValueError("Primary endpoint models have no paired GW outcomes")
    delta = pd.DataFrame(index=common)
    delta["component"] = 0.0
    for metric, metric_weight in metric_weights.items():
        delta["component"] += metric_weight * (
            new.loc[common, metric] - baseline.loc[common, metric]
        )
    delta = delta.reset_index()
    delta["component"] *= delta["gallery_size"].map(sizes)
    counts = delta.groupby(["gw_id", "source"])["gallery_size"].nunique()
    if not counts.eq(len(sizes)).all():
        raise ValueError(
            "Primary endpoint is missing one or more weighted gallery sizes"
        )
    per_gw = (
        delta.groupby(["gw_id", "source"], as_index=False)["component"]
        .sum()
        .rename(columns={"component": str(primary_metric["name"])})
    )
    summaries = _summarize_source_aggregations(
        per_gw,
        value_columns=[str(primary_metric["name"])],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    row = next(
        item for item in summaries if item["source_aggregation"] == "source_macro"
    )
    supported = bool(row["ci_low"] > 0.0)
    return pd.DataFrame(
        [
            {
                "endpoint": str(primary_metric["name"]),
                "condition": condition,
                "scope": scope,
                "source_aggregation": "source_macro",
                "new_model": new_name,
                "baseline_model": baseline_name,
                "mean_delta": row["estimate"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "decision_rule": "supported if 95% CI lower bound > 0",
                "conclusion": "supported" if supported else "inconclusive",
                "n_unique_gw": row["n_unique_gw"],
                "bootstrap_unit": row["bootstrap_unit"],
                "n_bootstrap": row["n_bootstrap"],
                "bootstrap_seed": row["bootstrap_seed"],
                "metric_weights": json.dumps(metric_weights, sort_keys=True),
                "gallery_size_weights": json.dumps(sizes, sort_keys=True),
            }
        ]
    )


def _model_style(model: str, model_index: int) -> dict[str, Any]:
    return dict(
        MODEL_STYLES.get(model, FALLBACK_STYLES[model_index % len(FALLBACK_STYLES)])
    )


def _save_figure(fig: Any, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".png", ".pdf"):
        path = stem.with_suffix(suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Plot export failed: {path}")
        paths.append(str(path))
    return paths


def plot_retrieval_curves(
    intervals: pd.DataFrame,
    *,
    output_dir: Path,
    model_order: Sequence[str],
    kn_fraction: float,
) -> list[str]:
    """Render interval-aware retrieval curves for every condition and scope."""
    import matplotlib.pyplot as plt

    apply_mnras_style(plt, base_font_size=13)
    paths: list[str] = []
    pooled = intervals[intervals["source_aggregation"].eq("pooled")]
    for condition in sorted(pooled["condition"].unique()):
        for scope in MIXED_SCOPES:
            panel = pooled[
                pooled["condition"].eq(condition) & pooled["scope"].eq(scope)
            ]
            if panel.empty:
                continue
            fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=True)
            for metric_index, (metric, label) in enumerate(PLOT_METRICS):
                ax = axes[metric_index]
                metric_panel = panel[panel["metric"].eq(metric)]
                for model_index, model in enumerate(model_order):
                    series = metric_panel[metric_panel["model"].eq(model)].sort_values(
                        "gallery_size"
                    )
                    if series.empty:
                        continue
                    style = _model_style(model, model_index)
                    x = series["gallery_size"].to_numpy(dtype=float)
                    estimate = series["estimate"].to_numpy(dtype=float)
                    ax.plot(x, estimate, label=model, markersize=5, **style)
                    ax.fill_between(
                        x,
                        series["ci_low"].to_numpy(dtype=float),
                        series["ci_high"].to_numpy(dtype=float),
                        color=style["color"],
                        alpha=0.13,
                        linewidth=0,
                    )
                sizes = sorted(metric_panel["gallery_size"].unique())
                random_values = []
                for size in sizes:
                    n_kn, n_nonkn = allocate_mixed_negative_counts(size, kn_fraction)
                    n_candidates = {
                        "all": size,
                        "kn_only": 1 + n_kn,
                        "nonkn_only": 1 + n_nonkn,
                    }[scope]
                    random_values.append(random_baseline(n_candidates)[metric])
                ax.plot(sizes, random_values, label="Random ranking", **RANDOM_STYLE)
                ax.set_xscale("log")
                ax.set_ylim(bottom=0.0)
                ax.set_title(label)
                ax.set_xlabel("Gallery size")
                if metric_index == 0:
                    ax.set_ylabel("Retrieval performance")
                ax.grid(True, alpha=0.2)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles, labels, loc="upper center", ncol=len(labels), frameon=False
            )
            title = condition.replace("_", " ").title()
            fig.suptitle(f"{title}: {scope.replace('_', ' ')} candidates", y=0.94)
            n_unique_gw = int(panel["n_unique_gw"].max())
            fig.text(
                0.5,
                0.01,
                f"n={n_unique_gw} query GWs; shading is a source-stratified 95% GW bootstrap interval.",
                ha="center",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0.05, 1, 0.86))
            plot_dir = output_dir / "plots" / condition / scope
            paths.extend(_save_figure(fig, plot_dir / "retrieval_curves"))
            plt.close(fig)
    return [str(Path(path).relative_to(output_dir)) for path in paths]


def plot_training_deltas(
    paired: pd.DataFrame,
    primary: pd.DataFrame,
    *,
    output_dir: Path,
    condition: str,
    scope: str,
) -> list[str]:
    """Render paired Mixed-minus-Default metric deltas over gallery size."""
    import matplotlib.pyplot as plt

    apply_mnras_style(plt, base_font_size=13)
    panel = paired[
        paired["condition"].eq(condition)
        & paired["scope"].eq(scope)
        & paired["source_aggregation"].eq("source_macro")
    ]
    if panel.empty:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharex=True)
    for axis, (metric, label) in zip(axes, PLOT_METRICS):
        series = panel[panel["metric"].eq(metric)].sort_values("gallery_size")
        x = series["gallery_size"].to_numpy(dtype=float)
        y = series["mean_delta"].to_numpy(dtype=float)
        axis.axhline(0.0, color="#333333", linewidth=1.0)
        axis.plot(x, y, color="#1769AA", marker="o", linestyle="-")
        axis.fill_between(
            x,
            series["ci_low"].to_numpy(dtype=float),
            series["ci_high"].to_numpy(dtype=float),
            color="#1769AA",
            alpha=0.15,
            linewidth=0,
        )
        axis.set_xscale("log")
        axis.set_title(label)
        axis.set_xlabel("Gallery size")
        axis.grid(True, alpha=0.2)
    axes[0].set_ylabel("Mixed Gallery v1 − Default MAGIKS")
    summary = primary.iloc[0]
    fig.suptitle(
        "Paired retrieval deltas; "
        f"training score Δ={summary['mean_delta']:.4f} "
        f"[{summary['ci_low']:.4f}, {summary['ci_high']:.4f}]",
        y=0.96,
    )
    fig.text(
        0.5,
        0.01,
        "Equal-source macro; trials averaged within GW before paired bootstrap.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.88))
    paths = _save_figure(
        fig, output_dir / "plots" / condition / scope / "training_effect_delta"
    )
    plt.close(fig)
    return [str(Path(path).relative_to(output_dir)) for path in paths]


def plot_redshift_performance(
    intervals: pd.DataFrame,
    *,
    output_dir: Path,
    model_order: Sequence[str],
    condition: str,
    scope: str,
    gallery_size: int,
    bin_labels: Sequence[str],
    aggregations: Sequence[str],
    kn_fraction: float,
) -> list[str]:
    """Render pooled and source-macro redshift performance with GW intervals."""
    import matplotlib.pyplot as plt

    panel = intervals[
        intervals["condition"].eq(condition)
        & intervals["scope"].eq(scope)
        & intervals["gallery_size"].eq(gallery_size)
    ]
    if panel.empty:
        return []
    apply_mnras_style(plt, base_font_size=12)
    fig, axes = plt.subplots(
        len(aggregations), 3, figsize=(15.2, 4.3 * len(aggregations)), squeeze=False
    )
    x = np.arange(len(bin_labels), dtype=float)
    for row_index, aggregation in enumerate(aggregations):
        aggregation_panel = panel[panel["source_aggregation"].eq(aggregation)]
        count_rows = (
            aggregation_panel[aggregation_panel["metric"].eq(PLOT_METRICS[0][0])]
            .groupby("redshift_bin_index", sort=True)["n_unique_gw"]
            .first()
        )
        tick_labels = [
            f"{label}\n(n={int(count_rows.loc[index])})"
            for index, label in enumerate(bin_labels)
        ]
        for column_index, (metric, label) in enumerate(PLOT_METRICS):
            ax = axes[row_index, column_index]
            metric_panel = aggregation_panel[aggregation_panel["metric"].eq(metric)]
            for model_index, model in enumerate(model_order):
                series = metric_panel[metric_panel["model"].eq(model)].sort_values(
                    "redshift_bin_index"
                )
                if series.empty:
                    continue
                style = _model_style(model, model_index)
                estimate = series["estimate"].to_numpy(dtype=float)
                ax.plot(x, estimate, label=model, markersize=5, **style)
                ax.fill_between(
                    x,
                    series["ci_low"].to_numpy(dtype=float),
                    series["ci_high"].to_numpy(dtype=float),
                    color=style["color"],
                    alpha=0.13,
                    linewidth=0,
                )
            n_kn, n_nonkn = allocate_mixed_negative_counts(gallery_size, kn_fraction)
            n_candidates = {
                "all": gallery_size,
                "kn_only": 1 + n_kn,
                "nonkn_only": 1 + n_nonkn,
            }[scope]
            ax.axhline(
                random_baseline(n_candidates)[metric],
                label="Random ranking",
                **RANDOM_STYLE,
            )
            ax.set_ylim(bottom=0.0)
            ax.set_title(label)
            ax.set_xticks(x, tick_labels, rotation=18, ha="right")
            ax.grid(True, alpha=0.2)
            if column_index == 0:
                ax.set_ylabel(
                    "Pooled" if aggregation == "pooled" else "Equal-source macro"
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle(
        f"Retrieval by query-GW redshift (gallery size {gallery_size})", y=0.97
    )
    fig.text(
        0.5,
        0.01,
        "Bins apply to query GW events only; shading is a 95% GW bootstrap interval.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.91))
    paths = _save_figure(
        fig, output_dir / "plots" / condition / scope / "redshift_performance"
    )
    plt.close(fig)
    return [str(Path(path).relative_to(output_dir)) for path in paths]


def postprocess(
    outcomes: pd.DataFrame,
    cfg: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Create GW-level statistics, decision summary, and all static figures."""
    collapsed = collapse_trials(outcomes)
    n_bootstrap = int(cfg["bootstrap_samples"])
    seed = int(cfg["bootstrap_seed"])
    intervals = bootstrap_metric_intervals(
        collapsed, n_bootstrap=n_bootstrap, seed=seed
    )
    intervals.to_csv(output_dir / "gw_metric_intervals.csv", index=False)

    new_name = str(cfg["bootstrap_new_model"])
    baseline_name = str(cfg["bootstrap_baseline_model"])
    paired = paired_bootstrap(
        collapsed,
        new_name=new_name,
        baseline_name=baseline_name,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    paired.to_csv(output_dir / "paired_bootstrap.csv", index=False)
    primary = training_effect_summary(
        collapsed,
        new_name=new_name,
        baseline_name=baseline_name,
        condition=str(cfg["primary_condition"]),
        scope=str(cfg["primary_scope"]),
        primary_metric=cfg["primary_metric"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    primary.to_csv(output_dir / "training_effect_summary.csv", index=False)
    (output_dir / "training_effect_summary.json").write_text(
        json.dumps(primary.iloc[0].to_dict(), indent=2, default=_json_default),
        encoding="utf-8",
    )

    model_order = [str(model["name"]) for model in cfg["models"]]
    retrieval_paths = plot_retrieval_curves(
        intervals,
        output_dir=output_dir,
        model_order=model_order,
        kn_fraction=float(cfg["kn_fraction"]),
    )
    delta_paths = plot_training_deltas(
        paired,
        primary,
        output_dir=output_dir,
        condition=str(cfg["primary_condition"]),
        scope=str(cfg["primary_scope"]),
    )

    redshift_paths: list[str] = []
    redshift_intervals = pd.DataFrame()
    redshift_cfg = cfg["redshift_analysis"]
    if redshift_cfg.get("enabled", False):
        required = {
            "redshift",
            "redshift_bin_index",
            "redshift_bin_label",
        }
        if not required.issubset(collapsed.columns):
            raise ValueError(
                "Redshift analysis is enabled but outcomes lack redshift columns"
            )
        redshift_collapsed = collapsed[
            collapsed["gallery_size"].eq(redshift_cfg["primary_gallery_size"])
            & collapsed["scope"].eq(cfg["primary_scope"])
        ]
        redshift_intervals = bootstrap_metric_intervals(
            redshift_collapsed,
            n_bootstrap=n_bootstrap,
            seed=seed,
            extra_group_columns=(
                "redshift_bin_index",
                "redshift_bin_label",
            ),
        )
        redshift_intervals.to_csv(
            output_dir / "redshift_metric_intervals.csv", index=False
        )
        for condition in cfg["conditions"]:
            redshift_paths.extend(
                plot_redshift_performance(
                    redshift_intervals,
                    output_dir=output_dir,
                    model_order=model_order,
                    condition=str(condition),
                    scope=str(cfg["primary_scope"]),
                    gallery_size=int(redshift_cfg["primary_gallery_size"]),
                    bin_labels=redshift_cfg["bin_labels"],
                    aggregations=redshift_cfg["aggregations"],
                    kn_fraction=float(cfg["kn_fraction"]),
                )
            )

    primary_prefix = f"plots/{cfg['primary_condition']}/{cfg['primary_scope']}/"
    all_plots = [*retrieval_paths, *delta_paths, *redshift_paths]
    return {
        "metric_intervals": intervals,
        "paired_bootstrap": paired,
        "training_effect": primary,
        "redshift_intervals": redshift_intervals,
        "artifacts": {
            "gw_metric_intervals": "gw_metric_intervals.csv",
            "paired_bootstrap": "paired_bootstrap.csv",
            "training_effect_csv": "training_effect_summary.csv",
            "training_effect_json": "training_effect_summary.json",
            "redshift_metric_intervals": (
                "redshift_metric_intervals.csv"
                if redshift_cfg.get("enabled", False)
                else None
            ),
            "retrieval_plots": retrieval_paths,
            "training_delta_plots": delta_paths,
            "redshift_plots": redshift_paths,
            "primary_plots": [
                path for path in all_plots if path.startswith(primary_prefix)
            ],
            "supplementary_plots": [
                path for path in all_plots if not path.startswith(primary_prefix)
            ],
        },
    }
