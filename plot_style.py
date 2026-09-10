from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt


BASE_FONT_SIZE = 12
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 14
LEGEND_FONT_SIZE = 11
TITLE_FONT_SIZE = 13
FIGURE_TITLE_SIZE = 14
COLORBAR_LABEL_SIZE = 13
COLORBAR_TICK_SIZE = 13


def apply_readable_plot_style() -> None:
    """Use manuscript-readable text without changing figure dimensions."""
    plt.rcParams.update(
        {
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "figure.titlesize": FIGURE_TITLE_SIZE,
            "axes.linewidth": 1.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "legend.framealpha": 0.9,
        }
    )


def _iter_axes(axes) -> Iterable:
    if axes is None:
        return ()
    if hasattr(axes, "flat"):
        return axes.flat
    if isinstance(axes, (list, tuple)):
        return axes
    return (axes,)


def style_axes(axes) -> None:
    for ax in _iter_axes(axes):
        ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
        ax.tick_params(axis="both", which="minor", labelsize=max(TICK_LABEL_SIZE - 2, 8))
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(LEGEND_FONT_SIZE)


def style_colorbar(colorbar) -> None:
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    colorbar.ax.yaxis.label.set_size(COLORBAR_LABEL_SIZE)
