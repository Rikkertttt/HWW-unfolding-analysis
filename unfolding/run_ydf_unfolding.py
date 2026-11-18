import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

import numpy as np
import json

from Omnifold.omnifold_ydf import omnifold_ydf
from Omnifold.features import make_features, make_cfg, make_theta_and_scale
from helpers.read_data import save_events
from helpers.data_loading import get_all_data_omnifold

prior_gen_DIR = "/data/atlas/users/rvanrhee/hww_METisNeutrino/full_selection/gen/"
prior_reco_DIR = "/data/atlas/users/rvanrhee/hww_METisNeutrino/full_selection/reco/"
data_gen_DIR  = "/data/atlas/users/rvanrhee/wwjj_METisNeutrino/full_selection/gen/"
data_reco_DIR = "/data/atlas/users/rvanrhee/wwjj_METisNeutrino/full_selection/reco/"


OUTDIR = "/user/rvanrhee/projects_rik/HWW-unfolding-analysis/unfolding/runs/wwjj_ydf_new_inputs/"
os.makedirs(OUTDIR, exist_ok=True)


prior_gen_events, prior_reco_events, data_gen_events, data_reco_events = get_all_data_omnifold(
    prior_gen_events_path=prior_gen_DIR,
    prior_reco_events_path=prior_reco_DIR,
    data_gen_events_path=data_gen_DIR,
    data_reco_events_path=data_reco_DIR,
    # closure_test=True,
    verbose=True,
)

assert data_gen_events is not None

features = [
    # Individual kinematics
    "j1_pt", "j1_eta",
    "j2_pt", "j2_eta",
    "electron_pt", "electron_eta",
    "muon_pt", "muon_eta",
    "met_pt",

    # Dijet system
    "m_jj", "jj_delta_eta", "jj_delta_phi", "jj_delta_R",

    # Dilepton system
    "ll_mass", "ll_pt", "ll_delta_phi", "ll_delta_eta", "ll_delta_R",

    # Jet-lepton system
    "j1_lep_delta_eta", "j2_lep_delta_eta",
    "j1_lep_delta_R", "j2_lep_delta_R",
]

feature_cfg = make_cfg(features)

# Augment the prior events by rotating and inverting them
print("Augmenting prior events by rotating and inverting...")
prior_reco_events = prior_reco_events.rotate_to_j1()
prior_gen_events = prior_gen_events.rotate_to_j1()
data_reco_events = data_reco_events.rotate_to_j1()
data_gen_events = data_gen_events.rotate_to_j1()
print(f"Augmented prior reco events: {len(prior_reco_events)}")
print(f"Augmented prior gen events: {len(prior_gen_events)}")

rng = np.random.default_rng(42)
idx = rng.choice(len(prior_reco_events), size=len(data_reco_events), replace=False)
prior_reco_events = prior_reco_events[idx]
prior_gen_events  = prior_gen_events[idx]

prior_reco_features = make_features(prior_reco_events, feature_cfg)
prior_gen_features  = make_features(prior_gen_events, feature_cfg)
data_reco_features = make_features(data_reco_events, feature_cfg)
data_gen_features = make_features(data_gen_events, feature_cfg)


theta0, theta_unknown_S, _ = make_theta_and_scale(prior_gen_features, prior_reco_features, data_reco_features)

ITERATIONS = 4


# Run Omnifold with YDF
weights = omnifold_ydf(
    theta0=theta0,
    theta_unknown_S=theta_unknown_S,
    iterations=ITERATIONS,
    feature_names=features,
    verbose=True,
    save_report_path=OUTDIR,
    num_trees=400,
    max_depth=6,
)


np.save(os.path.join(OUTDIR, "weights.npy"), weights)
save_events(prior_gen_events,       os.path.join(OUTDIR, "prior_gen_matched.parquet"))
save_events(prior_reco_events,      os.path.join(OUTDIR, "prior_reco_matched.parquet"))
save_events(data_gen_events,  os.path.join(OUTDIR, "data_gen_matched.parquet"))
save_events(data_reco_events, os.path.join(OUTDIR, "data_reco_matched.parquet"))

with open(os.path.join(OUTDIR, "features.json"), "w") as f:
    json.dump(features, f, indent=2)

final_reco_weights  = np.asarray(weights[-1, 0, :])
final_truth_weights = np.asarray(weights[-1, 1, :])

assert np.isfinite(final_reco_weights).all()
assert np.isfinite(final_truth_weights).all()

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
