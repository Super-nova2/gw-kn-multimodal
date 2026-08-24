#!/usr/bin/env python3
"""Plot MRR curves for the mixed KN/non-KN retrieval comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_STYLES = {
    "Physical Pairing v1": {
        "color": "#1769AA",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.4,
        "zorder": 5,
    },
    "Default MAGIKS": {
        "color": "#E07A1F",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.0,
        "zorder": 4,
    },
    "w/o Retrieval Loss": {
        "color": "#788C32",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 1.8,
        "zorder": 3,
    },
    "Optical-only": {
        "color": "#73777B",
        "marker": "D",
        "linestyle": ":",
        "linewidth": 1.8,
        "zorder": 2,
    },
}

CONDITIONS = (
    ("training_aligned", "Training-aligned time/sky"),
    ("positive_shared", "Positive-shared time/sky"),
)
SCOPES = (
    ("all", "Mixed gallery"),
    ("kn_only", "KN distractors only"),
    ("nonkn_only", "Non-KN distractors only"),
)


def plot_results(metrics_path: Path, output_prefix: Path) -> None:
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["source"].eq("all")].copy()
    expected = set(MODEL_STYLES)
    missing = expected - set(metrics["model"])
    if missing:
        raise ValueError(f"Missing models in metrics: {sorted(missing)}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True, sharey=True)

    for row, (condition, condition_label) in enumerate(CONDITIONS):
        for column, (scope, scope_label) in enumerate(SCOPES):
            axis = axes[row, column]
            panel = metrics[
                metrics["condition"].eq(condition) & metrics["scope"].eq(scope)
            ]
            for model, style in MODEL_STYLES.items():
                curve = panel[panel["model"].eq(model)].sort_values("gallery_size")
                axis.plot(
                    curve["gallery_size"],
                    curve["mrr"],
                    label=model,
                    markersize=5.0,
                    markeredgewidth=0.8,
                    markerfacecolor=(
                        "white" if model != "Physical Pairing v1" else style["color"]
                    ),
                    **style,
                )
            random_curve = panel.groupby("gallery_size", as_index=False)[
                "random_mrr"
            ].first()
            axis.plot(
                random_curve["gallery_size"],
                random_curve["random_mrr"],
                color="#222222",
                linewidth=1.2,
                linestyle=(0, (1, 2)),
                label="Random ranking",
                zorder=1,
            )
            axis.set_xscale("log")
            axis.set_ylim(0.0, 1.0)
            axis.set_xticks(sorted(panel["gallery_size"].unique()))
            axis.set_xticklabels(
                [str(value) for value in sorted(panel["gallery_size"].unique())]
            )
            axis.grid(axis="y", color="#D9DDE1", linewidth=0.7, alpha=0.8)
            axis.grid(axis="x", visible=False)
            axis.set_title(scope_label if row == 0 else "")
            if column == 0:
                axis.set_ylabel(f"{condition_label}\nMRR")
            if row == 1:
                axis.set_xlabel("Gallery size")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
        handlelength=2.8,
    )
    fig.suptitle("Mixed KN/non-KN retrieval comparison", fontsize=14, y=0.995)
    fig.text(
        0.5,
        0.965,
        "512 GW queries (256 BNS + 256 NSBH), 10 trials; KN:non-KN distractors = 1:3",
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    fig.subplots_adjust(
        left=0.085, right=0.985, bottom=0.09, top=0.875, wspace=0.16, hspace=0.16
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()
    plot_results(
        args.metrics.expanduser().resolve(), args.output_prefix.expanduser().resolve()
    )


if __name__ == "__main__":
    main()
