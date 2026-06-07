from __future__ import annotations

from typing import Iterable, Sequence


MNRAS_SERIF_FONTS = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
MNRAS_FONT_SIZE = 15


def apply_mnras_style(plt, *, base_font_size: int = MNRAS_FONT_SIZE) -> None:
    """Apply shared publication-style Matplotlib defaults for MNRAS figures."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": MNRAS_SERIF_FONTS,
            "font.size": base_font_size,
            "axes.labelsize": base_font_size,
            "axes.titlesize": base_font_size,
            "xtick.labelsize": base_font_size - 1,
            "ytick.labelsize": base_font_size - 1,
            "legend.fontsize": base_font_size - 1,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.0,
            "savefig.dpi": 300,
        }
    )


def layout_top_below_legend(
    fig,
    legend,
    *,
    gap: float = 0.012,
    min_top: float = 0.80,
    max_top: float = 0.94,
) -> float:
    """Return a tight-layout top bound just below a figure-level legend."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = legend.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())
    return max(min_top, min(max_top, float(bbox.y0) - gap))


def set_no_titles(axes) -> None:
    """Clear titles from one Axes, an iterable of Axes, or a numpy axes array."""
    if hasattr(axes, "flat"):
        axes_iter: Iterable = axes.flat
    elif isinstance(axes, (list, tuple)):
        axes_iter = axes
    else:
        axes_iter = (axes,)
    for ax in axes_iter:
        ax.set_title("")


def add_panel_labels_below(
    axes,
    *,
    labels: Sequence[str] | None = None,
    y: float = -0.34,
    fontsize: int | None = None,
) -> None:
    """Embed panel labels such as (a), (b), (c) below each subplot."""
    if hasattr(axes, "flat"):
        axes_list = list(axes.flat)
    elif isinstance(axes, (list, tuple)):
        axes_list = list(axes)
    else:
        axes_list = [axes]

    if labels is None:
        labels = [f"({chr(ord('a') + idx)})" for idx in range(len(axes_list))]

    for ax, label in zip(axes_list, labels):
        if not ax.axison:
            continue
        ax.text(
            0.5,
            y,
            str(label),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=fontsize,
            clip_on=False,
        )
