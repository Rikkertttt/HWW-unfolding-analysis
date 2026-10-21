

import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

import numpy as np
from Omnifold.plotting import open_saved_files, plot_omnifold_overview, plot_closure_summary

# Path where all the .npy files are saved
PATH = "/user/rvanrhee/projects_rik/HWW-unfolding-analysis/unfolding/newpipelinetest"

# Open all .npy files
gen_features, reco_features, data_gen_features, data_reco_features, weights, feature_names = open_saved_files(
    PATH,
    data_gen_name="gen_wwjj_features",
    data_reco_name="reco_wwjj_features",
)

# Extract weights
final_reco_weights  = weights[-1, 0, :]
final_truth_weights = weights[-1, 1, :]

# Decide which features to plot 
# (column, label, bins)
plot_cfg = [
    (0, r"Leading-jet $p_T$ [GeV]",    np.linspace(0,    500, 41)),
    (1, r"Leading-jet $\eta$",          np.linspace(-4.5, 4.5, 41)),
    (3, r"Subleading-jet $p_T$ [GeV]", np.linspace(0,    300, 41)),
    (4, r"Subleading-jet $\eta$",       np.linspace(-4.5, 4.5, 41)),
]

# Make rows of the chosen features
rows = [
    dict(
        reco_prior=reco_features[:, col],
        truth_prior=gen_features[:, col],
        data_reco=data_reco_features[:, col],
        data_truth=data_gen_features[:, col],
        final_reco_weights=final_reco_weights,
        final_truth_weights=final_truth_weights,
        bins=bins,
        label=label,
    )
    for col, label, bins in plot_cfg
]

plot_omnifold_overview(
    feature_rows=rows,
    outdir=PATH,
    filename="omnifold_overview_wwjj",
    title="OmniFold overview — WWjj stress test",
)

plot_closure_summary(
    feature_rows=rows,
    outdir=PATH,
    filename="omnifold_closure_wwjj",
    title="Omnifold closure summary - WWjj stress test",
)