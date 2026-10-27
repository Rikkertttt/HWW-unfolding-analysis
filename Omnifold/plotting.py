import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import os
import json

from typing import List, Tuple, Optional

def open_saved_weights(
    PATH:            str,
    weight_name:     str = "weights",
    json_name:       str = "features",
    clip_percentile: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Load OmniFold weights and feature names from PATH.
    
    Returns final reco weights (step 1), final truth weights (step 2), and feature names.
    If clip_percentile is given, both weight arrays are clipped at that percentile.
    """

    weights = np.load(os.path.join(PATH, weight_name + ".npy"))

    w_reco  = weights[-1, 0, :]
    w_truth = weights[-1, 1, :]

    if clip_percentile is not None:
        final_reco_weights  = np.clip(w_reco,  0, np.percentile(w_reco,  clip_percentile))
        final_truth_weights = np.clip(w_truth, 0, np.percentile(w_truth, clip_percentile))
    else:
        final_reco_weights  = w_reco
        final_truth_weights = w_truth

    json_path = os.path.join(PATH, json_name + ".json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            feature_names = json.load(f)
    else:
        feature_names = None
        print(f"Warning: {json_path} not found, feature names unavailable.")

    return final_reco_weights, final_truth_weights, feature_names

def plot_weight_distributions(
    final_reco_weights:  np.ndarray,
    final_truth_weights: np.ndarray,
    outdir:              str,
    filename:            str = "omnifold_weights",
    title:               str = "OmniFold weight distributions",
) -> None:
    """Plot reco (step 1) and truth (step 2) weight distributions side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, w, label in zip(
        axes,
        [final_reco_weights, final_truth_weights],
        ["Reco weights (Step 1)", "Truth weights (Step 2)"],
    ):
        ax.hist(w, bins=80, histtype="stepfilled", alpha=0.7)
        ax.axvline(1.0, color="black", linestyle="--", label="Weight = 1")
        ax.set_title(label)
        ax.set_xlabel("OmniFold weight")
        ax.set_ylabel("Events")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{filename}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{filename}.png"), bbox_inches="tight")
    plt.close(fig)

def plot_feature_row(
    axes:                np.ndarray,
    reco_prior:          np.ndarray,
    truth_prior:         np.ndarray,
    data_reco:           np.ndarray,
    data_truth:          np.ndarray,
    final_reco_weights:  np.ndarray,
    final_truth_weights: np.ndarray,
    bins:                np.ndarray,
    label:               str,
) -> None:
    """Draw one row: detector-level | particle-level."""

    # Detector-level
    axes[0].hist(reco_prior, bins=bins, density=True, histtype="step", linewidth=2, label="Reco sim (HWW), unweighted")
    axes[0].hist(data_reco,  bins=bins, density=True, histtype="step", linewidth=2, label="Reco data (WWjj)")
    axes[0].hist(reco_prior, bins=bins, density=True, histtype="step", linewidth=2, label="Reco sim (HWW), Step 1", weights=final_reco_weights)
    axes[0].set_xlabel(label)
    axes[0].set_ylabel("Normalised events")
    axes[0].set_title(f"Detector-level — {label}")
    axes[0].legend(fontsize=9)

    # Particle-level
    axes[1].hist(truth_prior, bins=bins, density=True, histtype="step", linewidth=2, label="Truth prior (HWW), unweighted")
    axes[1].hist(data_truth,  bins=bins, density=True, histtype="step", linewidth=2, label="Truth data (WWjj)")
    axes[1].hist(truth_prior, bins=bins, density=True, histtype="step", linewidth=2, label="Unfolded truth, Step 2", weights=final_truth_weights)
    axes[1].set_xlabel(label)
    axes[1].set_ylabel("Normalised events")
    axes[1].set_title(f"Particle-level — {label}")
    axes[1].legend(fontsize=9)

def plot_omnifold_overview(
    feature_rows: list,   # list of dicts, each with keys matching plot_feature_row args (except axes)
    outdir:       str,
    filename:     str  = "omnifold_overview",
    title:        str  = "OmniFold stress test",
) -> None:
    """Create a figure with one row per feature and save it."""
    n = len(feature_rows)
    fig, axes = plt.subplots(n, 2, figsize=(17, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing works for single row

    for row, kwargs in enumerate(feature_rows):
        plot_feature_row(axes[row], **kwargs)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{filename}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{filename}.png"), bbox_inches="tight")
    plt.close(fig)







def plot_closure_summary_row(
    axes:                np.ndarray,   # array of 2 axes: [main, ratio]
    truth_prior:         np.ndarray,
    data_reco:           np.ndarray,
    data_truth:          np.ndarray,
    final_truth_weights: np.ndarray,
    bins:                np.ndarray,
    label:               str,
    **kwargs,
) -> None:
    """Draw one column: main distribution + ratio panel."""

    bin_centers      = 0.5 * (bins[:-1] + bins[1:])
    counts_truth,    _ = np.histogram(data_truth,   bins=bins, density=True)
    counts_unfolded, _ = np.histogram(truth_prior,  bins=bins, density=True, weights=final_truth_weights)

    # Main
    axes[0].hist(data_reco,   bins=bins, density=True, histtype="step", linewidth=2, label="Reco data (WWjj)")
    axes[0].hist(data_truth,  bins=bins, density=True, histtype="step", linewidth=2, label="Truth data (WWjj)")
    axes[0].hist(truth_prior, bins=bins, density=True, histtype="step", linewidth=2, label="Unfolded (Step 2)", weights=final_truth_weights)
    axes[0].set_ylabel("Normalised events")
    axes[0].set_title(label)
    axes[0].legend(fontsize=9)
    axes[0].set_xticks([])

    # Ratio
    ratio = np.full_like(counts_truth, np.nan)
    nonzero = counts_truth > 0
    ratio[nonzero] = counts_unfolded[nonzero] / counts_truth[nonzero]
    
    axes[1].plot(bin_centers, ratio, "o", markersize=3)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_xlabel(label)
    axes[1].set_ylabel("Unf. / Truth")


def plot_closure_summary(
    feature_rows: list,
    outdir:       str,
    filename:     str = "omnifold_closure_summary",
    title:        str = "OmniFold closure — summary",
    **kwargs,
) -> None:
    n = len(feature_rows)
    fig, axes = plt.subplots(
        2, n,
        figsize=(5 * n, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    if n == 1:
        axes = axes[:, np.newaxis]  # ensure 2D indexing for single column

    for col_idx, kwargs in enumerate(feature_rows):
        plot_closure_summary_row(axes[:, col_idx], **kwargs)

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{filename}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"{filename}.png"), bbox_inches="tight")
    plt.close(fig)