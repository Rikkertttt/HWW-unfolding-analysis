import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

import numpy as np
import awkward as ak
from Omnifold.plotting import open_saved_weights, plot_omnifold_overview, plot_closure_summary, plot_weight_distributions
from Omnifold.features import build_feature_rows
from helpers.read_data import load_events

# ==============================================================================
# Configuration
# ==============================================================================
# Directory where weights and matched parquet files are saved
PATH = "/user/rvanrhee/projects_rik/HWW-unfolding-analysis/unfolding/runs/ydf_datawwjj_priorhwwttbar"

# Run label used in filenames and titles (defaults to the folder name in PATH)
RUN_NAME = os.path.basename(os.path.normpath(PATH))

# Display names for legends and titles
PRIOR_NAME = "HWW + ttbar prior"   # Name of simulation / prior
DATA_NAME  = "WWjj data"             # Name of data / pseudo-data

# Optional settings
CLIP_PERCENTILE = 99.9
EXTRA_FEATURES  = ["higgs_mass"]
WEIGHT_BINS     = 150
# ==============================================================================

# Combined list for functions expecting [prior_name, data_name]
PRIOR_DATA_NAMES = [PRIOR_NAME, DATA_NAME]

# Load pre-matched events (no reload/rematch needed)
prior_gen_events  = load_events(os.path.join(PATH, "prior_gen_matched.parquet"))
prior_reco_events = load_events(os.path.join(PATH, "prior_reco_matched.parquet"))
data_gen_events   = load_events(os.path.join(PATH, "data_gen_matched.parquet"))
data_reco_events  = load_events(os.path.join(PATH, "data_reco_matched.parquet"))

# Load weights
final_reco_weights, final_truth_weights, feature_names = open_saved_weights(
    PATH=PATH,
    clip_percentile=CLIP_PERCENTILE
)

assert feature_names is not None, f"Could not find features.json in {PATH}"

# Compute observables directly from EventObjects
all_features = feature_names + [f for f in EXTRA_FEATURES if f not in feature_names]
rows = build_feature_rows(
    features=all_features,
    prior_gen_events=prior_gen_events,
    prior_reco_events=prior_reco_events,
    data_gen_events=data_gen_events,
    data_reco_events=data_reco_events,
    final_reco_weights=final_reco_weights,
    final_truth_weights=final_truth_weights,
)

print(f"Starting plotting for run: {RUN_NAME}")

overview_dir = os.path.join(PATH, "individual_plots", "overview")
closure_dir  = os.path.join(PATH, "individual_plots", "closure")
os.makedirs(overview_dir, exist_ok=True)
os.makedirs(closure_dir, exist_ok=True)

# 1. Weight distributions
plot_weight_distributions(
    final_reco_weights=final_reco_weights,
    final_truth_weights=final_truth_weights,
    outdir=PATH,
    filename=f"omnifold_weights_{RUN_NAME}",
    title=f"OmniFold weights — {RUN_NAME}",
    bins=WEIGHT_BINS,
)
print("Plotted weights")

# 2. Overview plots (all features grid + individual features)
plot_omnifold_overview(
    feature_rows=rows,
    outdir=PATH,
    filename=f"omnifold_overview_{RUN_NAME}",
    title=f"OmniFold overview — {RUN_NAME}",
    prior_data_name=PRIOR_DATA_NAMES,
)

for row in rows:
    feat = row["feature"]
    plot_omnifold_overview(
        feature_rows=[row],
        outdir=overview_dir,
        filename=f"omnifold_overview_{feat}_{RUN_NAME}",
        title=f"OmniFold overview — {feat} ({RUN_NAME})",
        prior_data_name=PRIOR_DATA_NAMES,
    )
print("Plotted overview")

# 3. Closure summary plots (all features grid + individual features)
plot_closure_summary(
    feature_rows=rows,
    outdir=PATH,
    filename=f"omnifold_closure_{RUN_NAME}",
    title=f"OmniFold closure summary — {RUN_NAME}",
    data_name=DATA_NAME,
)

for row in rows:
    feat = row["feature"]
    plot_closure_summary(
        feature_rows=[row],
        outdir=closure_dir,
        filename=f"omnifold_closure_{feat}_{RUN_NAME}",
        title=f"OmniFold closure summary — {feat} ({RUN_NAME})",
        data_name=DATA_NAME,
    )
print("Plotted summary")
