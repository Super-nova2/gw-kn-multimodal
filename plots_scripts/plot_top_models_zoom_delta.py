#!/usr/bin/env python3
"""Plot top-6 retrieval curves: zoomed absolute view + delta vs Full v11.

Usage:
    python plots_scripts/plot_top_models_zoom_delta.py
    python plots_scripts/plot_top_models_zoom_delta.py /path/to/result_dir

Matches the visual style of ``plot_retrieval_curves`` as closely as possible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULT_DIR = REPO_ROOT / "Model/eval_results/retrieval_comparison_v11"

PLOT_DPI = 300

GALLERY_WEIGHTS = [10, 100, 500, 1000, 2000, 5000]

# Per-metric y-limit for the absolute row (matches original full-range [0,1]
# but zoomed to the top-6 range so small gaps become visible).
ABS_YLIM: dict[str, tuple[float, float]] = {
    "R@1":  (0.18, 1.02),
    "R@10": (0.58, 1.03),
    "MRR":  (0.30, 1.02),
}


def _label(method: str) -> str:
    return method.replace("v11 ", "")


def _rank_top6(table_rows: list[dict]) -> list[str]:
    """Return top-6 method names ranked by log10(gallery)-weighted MRR."""
    scores: dict[str, float] = {}
    for row in table_rows:
        w_sum = 0.0
        g_sum = 0.0
        for g in GALLERY_WEIGHTS:
            mrr_v = row[f"gallery_{g}"]["MRR"]
            w = np.log10(g)
            w_sum += mrr_v * w
            g_sum += w
        scores[row["method"]] = w_sum / g_sum
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:6]]


def _detect_json_path(result_dir: Path) -> Path:
    """Find the JSON file — can be ``ablation_comparison.json`` or
    ``gw170817a_retrieval.json`` (170817-like eval)."""
    for candidate in ["ablation_comparison.json", "gw170817a_retrieval.json"]:
        p = result_dir / candidate
        if p.exists():
            return p
    raise FileNotFoundError(f"No JSON found in {result_dir}")


def main() -> int:
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULT_DIR
    json_path = _detect_json_path(result_dir)
    out_path = result_dir / "retrieval_curves_top_models_zoom_delta.png"

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # ── rank top 6 by log10-weighted MRR ───────────────────────────────────
    top6 = _rank_top6(data["table"]["rows"])
    print("Top 6 models by log10-weighted MRR:")
    for i, name in enumerate(top6, 1):
        print(f"  {i}. {name}")

    # ── pivot curve_rows → dict[method][metric][gallery] = value ──────────
    curve_rows = data.get("curve_rows", [])
    gallery_sizes = sorted({int(r["gallery_size_target"]) for r in curve_rows})

    by_method: dict[str, dict[str, dict[int, float]]] = {}
    for r in curve_rows:
        m = str(r["method"])
        metric = str(r["metric_name"])
        g = int(r["gallery_size_target"])
        by_method.setdefault(m, {}).setdefault(metric, {})[g] = float(r["metric_value"])

    ref = by_method["Full v11"]

    # ── build figure ──────────────────────────────────────────────────────
    metrics = ["R@1", "R@10", "MRR"]
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 9),
        gridspec_kw={"hspace": 0.30, "wspace": 0.10},
    )

    # ── draw ──────────────────────────────────────────────────────────────
    for col, metric in enumerate(metrics):
        ax_abs = axes[0, col]
        ax_delta = axes[1, col]
        g_plot = gallery_sizes

        # ── top: absolute curves ──────────────────────────────────────
        for m in top6:
            vals = [by_method[m][metric][g] for g in g_plot]
            ax_abs.plot(
                g_plot, vals,
                marker="o", linewidth=2,
                label=_label(m),
            )

        ax_abs.set_xscale("log")
        ax_abs.set_ylim(*ABS_YLIM[metric])
        ax_abs.set_ylabel(metric)
        ax_abs.set_title(f"{metric} vs Gallery Size")
        ax_abs.grid(True, alpha=0.3)
        ax_abs.set_xlabel("")

        # ── bottom: delta vs Full v11 (percentage points) ──────────────
        for m in top6:
            deltas = [(by_method[m][metric][g] - ref[metric][g]) * 100
                      for g in g_plot]
            ax_delta.plot(
                g_plot, deltas,
                marker="o", linewidth=2,
                label=_label(m),
            )

        ax_delta.axhline(0, color="gray", linewidth=1.0, linestyle="--", alpha=0.5)
        ax_delta.set_xscale("log")
        # Symmetric delta y-range, at least ±2 pp
        all_d = [
            (by_method[m][metric][g] - ref[metric][g]) * 100
            for m in top6 for g in g_plot
        ]
        d_pad = max(max(abs(min(all_d)), abs(max(all_d))) * 1.2, 2.0)
        ax_delta.set_ylim(-d_pad, d_pad)
        ax_delta.set_xlabel("Gallery Size")
        ax_delta.set_ylabel("Δ (pp)")
        ax_delta.set_title(f"Δ {metric} vs Full v11")
        ax_delta.grid(True, alpha=0.3)

    # ── legend (original style: frameon=False, top-center) ─────────────────
    handles, labels = [], []
    for ax in axes[0, :]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi); labels.append(li)
    rank_map = {_label(m): i for i, m in enumerate(top6)}
    paired = sorted(zip(handles, labels), key=lambda x: rank_map.get(x[1], 99))
    h_sorted, l_sorted = zip(*paired)
    if h_sorted:
        fig.legend(
            h_sorted, l_sorted,
            loc="upper center",
            ncol=max(1, min(3, len(l_sorted))),
            frameon=False,
        )

    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.98, hspace=0.30, wspace=0.12)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
