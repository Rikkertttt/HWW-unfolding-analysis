import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

import numpy as np
import tensorflow as tf
import json

import Omnifold.wandb_helpers as wb
from Omnifold.omnifold import omnifold
from Omnifold.models import build_omnifold_classifier
from Omnifold.features import make_features, make_cfg, make_theta_and_scale
from helpers.read_data import save_events
from helpers.data_loading import get_all_data_omnifold

prior_gen_DIR  = "/data/atlas/users/rvanrhee/hww_METisNeutrino/full_selection/gen/"
prior_reco_DIR = "/data/atlas/users/rvanrhee/hww_METisNeutrino/full_selection/reco/"
data_gen_DIR   = "/data/atlas/users/rvanrhee/wwjj_parquet/full_selection/gen/"
data_reco_DIR  = "/data/atlas/users/rvanrhee/wwjj_parquet/full_selection/reco/"

wbname = "omnifold_hww_wwjj_rotated"

OUTDIR = "/user/rvanrhee/projects_rik/HWW-unfolding-analysis/unfolding/wwjj_mlp_rotated_v2_set_to_j1/"
os.makedirs(OUTDIR, exist_ok=True)


prior_gen_events, prior_reco_events, data_gen_events, data_reco_events = get_all_data_omnifold(
    prior_gen_events_path=prior_gen_DIR,
    prior_reco_events_path=prior_reco_DIR,
    data_gen_events_path=data_gen_DIR,
    data_reco_events_path=data_reco_DIR,
    verbose=True,
)

assert data_gen_events is not None

features = [
    "j1_pt", "j1_eta", "j1_phi",
    "j2_pt", "j2_eta", "j2_phi",
    "electron_pt", "electron_eta", "electron_phi",
    "muon_pt", "muon_eta", "muon_phi",
]
feature_cfg = make_cfg(features)

# Augment the prior events by rotating and inverting
print("Augmenting prior events by rotating and inverting...")
# prior_reco_events = prior_reco_events + prior_reco_events.rotate(2) + prior_reco_events.rotate(np.pi / 2) + prior_reco_events.invert()
# prior_gen_events  = prior_gen_events  + prior_gen_events.rotate(2)  + prior_gen_events.rotate(np.pi / 2)  + prior_gen_events.invert()
prior_reco_events = prior_reco_events.rotate_to_j1()
prior_gen_events = prior_gen_events.rotate_to_j1()
data_reco_events = data_reco_events.rotate_to_j1()
data_gen_events = data_gen_events.rotate_to_j1()
print(f"Augmented prior reco events: {len(prior_reco_events)}")
print(f"Augmented prior gen events:  {len(prior_gen_events)}")

# Subsample prior to match data size (avoid class imbalance in classifier)
rng = np.random.default_rng(42)
idx = rng.choice(len(prior_reco_events), size=len(data_reco_events), replace=False)
prior_reco_events_balanced = prior_reco_events[idx]
prior_gen_events_balanced  = prior_gen_events[idx]

prior_reco_features = make_features(prior_reco_events_balanced, feature_cfg)
prior_gen_features  = make_features(prior_gen_events_balanced,  feature_cfg)
data_reco_features  = make_features(data_reco_events,           feature_cfg)
data_gen_features   = make_features(data_gen_events,            feature_cfg)

theta0, theta_unknown_S, _ = make_theta_and_scale(prior_gen_features, prior_reco_features, data_reco_features)

tf.keras.utils.set_random_seed(42)
ITERATIONS = 2

model = build_omnifold_classifier(n_features=theta0.shape[-1])
model.summary()

# 1. Init W&B run
wb.init_omnifold_run(
    iterations=ITERATIONS,
    theta0=theta0,
    theta_unknown_S=theta_unknown_S,
    run_name=wbname,
    extra_config={"data": "wwjj", "closure_test": True, "augmented": True},
)

# 2. Callback factory
def make_callbacks(iteration):
    def make_es():
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )
    return [
        [wb.WandbOmniFoldCallback(iteration, step=1), make_es()],
        [wb.WandbOmniFoldCallback(iteration, step=2), make_es()],
    ]

# 3. Run OmniFold
weights = omnifold(
    theta0,
    theta_unknown_S,
    iterations=ITERATIONS,
    model=model,
    callbacks_fn=make_callbacks,
    verbose=True,
)

# 4. Log weights per iteration
for i in range(ITERATIONS):
    wb.log_weights(i, weights[i, 0, :], weights[i, 1, :])

wb.wandb.finish()

np.save(os.path.join(OUTDIR, "weights.npy"), weights)
save_events(prior_gen_events_balanced,  os.path.join(OUTDIR, "prior_gen_matched.parquet"))
save_events(prior_reco_events_balanced, os.path.join(OUTDIR, "prior_reco_matched.parquet"))
save_events(data_gen_events,            os.path.join(OUTDIR, "data_gen_matched.parquet"))
save_events(data_reco_events,           os.path.join(OUTDIR, "data_reco_matched.parquet"))

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
