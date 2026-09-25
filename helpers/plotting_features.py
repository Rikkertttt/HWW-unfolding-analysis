import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional

import awkward as ak
import numpy as np

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np


def plot_hist(
    data: ak.Array,
    filename: Optional[str] = None,
    *,
    show: bool = False,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Entries",
    show_mean_std: bool = False,
    bins: int = 50,
    density: bool = False,
):
    """Plot a clean histogram of nested or flat data."""

    values = ak.to_numpy(data)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        values,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        color="#0072B2",
        density=density,
    )

    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel="Normalized entries" if density else ylabel,
    )

    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Cleaner analysis-style axes.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", top=False, right=False)

    if show_mean_std:
        stats = (
            f"Entries: {len(values)}\n"
            f"Mean: {np.mean(values):.4f}\n"
            f"Std: {np.std(values):.3f}"
        )

        ax.text(
            0.98,
            0.98,
            stats,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.9,
            },
        )

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename + ".pdf", bbox_inches="tight")
        fig.savefig(filename + ".png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_comparison_hist(
    data: list[ak.Array],
    datanames: Optional[list[str]] = None,
    filename: Optional[str] = None,
    *,
    show: bool = False,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Entries",
    bins: int = 50,
    density: bool = False,
    ratio: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a visually clean comparison histogram."""

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    # Flatten once and use shared bin edges for every dataset.
    values_list = [ak.to_numpy(ak.ravel(dataset)) for dataset in data]

    global_min = min(values.min() for values in values_list)
    global_max = max(values.max() for values in values_list)
    bin_edges = np.linspace(global_min, global_max, bins + 1)

    ax_ratio: Optional[Axes] = None
    
    if ratio:
        fig, (ax_main, ax_ratio) = plt.subplots(
            2, 1,
            figsize=(7, 6),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.subplots_adjust(hspace=0.05)
    else:
        fig, ax_main = plt.subplots(figsize=(7, 5))

    counts_list = []

    for i, values in enumerate(values_list):
        label = datanames[i] if datanames is not None else f"Dataset {i + 1}"

        counts, _ = np.histogram(values, bins=bin_edges, density=density)
        counts_list.append(counts)

        ax_main.hist(
            values,
            bins=bin_edges,
            histtype="step",
            linewidth=1.8,
            color=colors[i % len(colors)],
            label=label,
            density=density,
        )

    ax_main.set(
        title=title,
        xlabel=xlabel,
        ylabel="Normalized entries" if density else ylabel,
    )

    ax_main.grid(axis="y", alpha=0.25, linestyle="--")
    ax_main.legend(frameon=False)
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.tick_params(direction="in", top=False, right=False)

    if ratio and ax_ratio is not None:
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ref = counts_list[0]

        # First dataset is the reference — draw it as a flat line at 1
        ax_ratio.axhline(1.0, color=colors[0], linewidth=1.5)

        for i, counts in enumerate(counts_list[1:], start=1):
            r = np.where(ref > 0, counts / ref, np.nan)
            ax_ratio.step(
                centers, r,
                where="mid",
                color=colors[i % len(colors)],
                linewidth=1.5,
            )

        ax_ratio.set(xlabel=xlabel, ylabel="Ratio", ylim=(0.5, 1.5))
        ax_ratio.grid(axis="y", alpha=0.25, linestyle="--")
        ax_ratio.spines["top"].set_visible(False)
        ax_ratio.spines["right"].set_visible(False)
        ax_ratio.tick_params(direction="in", top=False, right=False)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename + ".pdf", bbox_inches="tight")
        fig.savefig(filename + ".png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax_main