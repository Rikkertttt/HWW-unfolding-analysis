import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

import numpy as np
import awkward as ak
from Omnifold.plotting import open_saved_weights, plot_omnifold_overview, plot_closure_summary, plot_weight_distributions
from Omnifold.features import build_feature_rows
from helpers.read_data import load_events

# Path where weights and matched parquet files are saved
PATH = "/user/rvanrhee/projects_rik/HWW-unfolding-analysis/unfolding/closure_test"

# Load pre-matched events (no reload/rematch needed)
prior_gen_events       = load_events(os.path.join(PATH, "prior_gen_matched.parquet"))
prior_reco_events      = load_events(os.path.join(PATH, "prior_reco_matched.parquet"))
data_gen_events  = load_events(os.path.join(PATH, "data_gen_matched.parquet"))
data_reco_events = load_events(os.path.join(PATH, "data_reco_matched.parquet"))

# Load weights
final_reco_weights, final_truth_weights, feature_names =  open_saved_weights(PATH=PATH, clip_percentile=99.9)
assert feature_names is not None

# Compute observables directly from EventObjects
rows = build_feature_rows(features=feature_names + ["higgs_mass"], 
                          prior_gen_events=prior_gen_events,
                          prior_reco_events=prior_reco_events,
                          data_gen_events=data_gen_events,
                          data_reco_events=data_reco_events,
                          final_reco_weights=final_reco_weights,
                          final_truth_weights=final_truth_weights,)

print("Starting plotting")

plot_weight_distributions(
    final_reco_weights=final_reco_weights,
    final_truth_weights=final_truth_weights,
    outdir=PATH,
    filename="omnifold_weights_closure",
)

print("Plotted weights")

plot_omnifold_overview(
    feature_rows=rows,
    outdir=PATH,
    filename="omnifold_overview_closure",
    title="OmniFold overview — closure test",
)

print("Plotted overview")

plot_closure_summary(
    feature_rows=rows,
    outdir=PATH,
    filename="omnifold_closure_closure",
    title="OmniFold closure summary — closure test",
)

print("Plotted summary")