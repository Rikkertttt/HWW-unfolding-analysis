import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

import numpy as np
import awkward as ak
from Omnifold.plotting import open_saved_weights, plot_omnifold_overview, plot_closure_summary, plot_weight_distributions
from Omnifold.features import build_feature_rows
from helpers.read_data import load_events

# Path where weights and matched parquet files are saved
PATH = "/user/rvanrhee/projects_rik/HWW-unfolding-analysis/unfolding/hww_closure_ydf_rotated_set_to_j1/"

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

overview_dir = os.path.join(PATH, "individual_plots", "overview")
closure_dir = os.path.join(PATH, "individual_plots", "closure")
os.makedirs(overview_dir, exist_ok=True)
os.makedirs(closure_dir, exist_ok=True)

plot_weight_distributions(
    final_reco_weights=final_reco_weights,
    final_truth_weights=final_truth_weights,
    outdir=PATH,
    filename="omnifold_weights_hww_closure_ydf_rotated",
    bins=150,
)

print("Plotted weights")

plot_omnifold_overview(
    feature_rows=rows,
    outdir=PATH,
    filename="omnifold_overview_hww_closure_ydf_rotated",
    title="OmniFold overview — hww_closure_ydf_rotated",
    prior_data_name=["HWW prior", "HWW data"],
)

for row in rows:
    plot_omnifold_overview(
        feature_rows=[row],
        outdir=overview_dir,
        filename=f"omnifold_overview_{row['feature']}_hww_closure_ydf_rotated",
        title=f"OmniFold overview — {row['feature']} (hww_closure_ydf_rotated)",
        prior_data_name=["HWW prior", "HWW data"],
    )

print("Plotted overview")

plot_closure_summary(
    feature_rows=rows,
    outdir=PATH,
    filename="omnifold_closure_hww_closure_ydf_rotated",
    title="OmniFold closure summary — hww_closure_ydf_rotated",
    data_name="HWW data",
)

for row in rows:
    plot_closure_summary(
        feature_rows=[row],
        outdir=closure_dir,
        filename=f"omnifold_closure_{row['feature']}_hww_closure_ydf_rotated",
        title=f"OmniFold closure summary — {row['feature']} (hww_closure_ydf_rotated)",
        data_name="HWW data",
    )

print("Plotted summary")