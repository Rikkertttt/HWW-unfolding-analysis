import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/LGATrslim_HWW_unfolding")

import numpy as np
import json

from Omnifold.omnifold_lgatr import omnifold_lgatr
from Omnifold.features import make_features, make_cfg, make_theta_and_scale
from helpers.read_data import save_events
from helpers.data_loading import get_all_data_omnifold

import torch

prior_gen_DIR  = "/data/atlas/users/rvanrhee/hww_METisNeutrino/full_selection/gen/"
prior_reco_DIR = "/data/atlas/users/rvanrhee/hww_METisNeutrino/full_selection/reco/"

OUTDIR = "/user/rvanrhee/projects_rik/LGATrslim_HWW_unfolding/unfolding/hww_closure_lgatr/"
os.makedirs(OUTDIR, exist_ok=True)

ITERATIONS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

prior_gen_events, prior_reco_events, data_gen_events, data_reco_events = get_all_data_omnifold(
    prior_gen_events_path=prior_gen_DIR,
    prior_reco_events_path=prior_reco_DIR,
    closure_test=True,
    verbose=True,
)

assert data_gen_events is not None


# ---------------------------------------------------------------------------
# No rotation needed — L-GATr-slim is azimuthally equivariant by construction
# ---------------------------------------------------------------------------

print(f"Prior reco events: {len(prior_reco_events)}")
print(f"Prior gen  events: {len(prior_gen_events)}")
print(f"Data  reco events: {len(data_reco_events)}")


# ---------------------------------------------------------------------------
# Build theta0 / theta_unknown_S
# (used only to determine array sizes; L-GATr reads four-momenta directly)
# ---------------------------------------------------------------------------

features = [
    "j1_pt", "j1_eta", "j1_phi",
    "j2_pt", "j2_eta", "j2_phi",
    "electron_pt", "electron_eta", "electron_phi",
    "muon_pt",     "muon_eta",     "muon_phi",
]
feature_cfg = make_cfg(features)

prior_reco_features = make_features(prior_reco_events, feature_cfg)
prior_gen_features  = make_features(prior_gen_events,  feature_cfg)
data_reco_features  = make_features(data_reco_events,  feature_cfg)
data_gen_features   = make_features(data_gen_events,   feature_cfg)

theta0, theta_unknown_S, _ = make_theta_and_scale(
    prior_gen_features, prior_reco_features, data_reco_features
)


# ---------------------------------------------------------------------------
# Run OmniFold with L-GATr-slim
# ---------------------------------------------------------------------------

weights = omnifold_lgatr(
    theta0=theta0,
    theta_unknown_S=theta_unknown_S,
    iterations=ITERATIONS,
    prior_gen_events=prior_gen_events,
    prior_reco_events=prior_reco_events,
    data_reco_events=data_reco_events,
    verbose=True,
    save_report_path=OUTDIR,
    # --- model architecture ---
    num_blocks=4,
    hidden_v_channels=16,
    hidden_s_channels=32,
    num_heads=4,
    # --- training ---
    n_epochs=5,
    batch_size=1024,
    lr=1e-3,
    device=DEVICE,
)


# ---------------------------------------------------------------------------
# Save weights and events
# ---------------------------------------------------------------------------

np.save(os.path.join(OUTDIR, "weights.npy"), weights)
save_events(prior_gen_events,  os.path.join(OUTDIR, "prior_gen_matched.parquet"))
save_events(prior_reco_events, os.path.join(OUTDIR, "prior_reco_matched.parquet"))
save_events(data_gen_events,   os.path.join(OUTDIR, "data_gen_matched.parquet"))
save_events(data_reco_events,  os.path.join(OUTDIR, "data_reco_matched.parquet"))

with open(os.path.join(OUTDIR, "features.json"), "w") as f:
    json.dump(features, f, indent=2)


# ---------------------------------------------------------------------------
# Inspect final weights
# ---------------------------------------------------------------------------

final_reco_weights  = np.asarray(weights[-1, 0, :])
final_truth_weights = np.asarray(weights[-1, 1, :])

assert np.isfinite(final_reco_weights).all(),  "Non-finite reco weights!"
assert np.isfinite(final_truth_weights).all(), "Non-finite truth weights!"

print(
    "Final reco weights:  "
    f"mean={final_reco_weights.mean():.3f}, "
    f"std={final_reco_weights.std():.3f}, "
    f"min={final_reco_weights.min():.3f}, "
    f"max={final_reco_weights.max():.3f}"
)
print(
    "Final truth weights: "
    f"mean={final_truth_weights.mean():.3f}, "
    f"std={final_truth_weights.std():.3f}, "
    f"min={final_truth_weights.min():.3f}, "
    f"max={final_truth_weights.max():.3f}"
)
