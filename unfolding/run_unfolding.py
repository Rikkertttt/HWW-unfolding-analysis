import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

import numpy as np
import awkward as ak
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

import Omnifold.wandb_helpers as wb
from Omnifold.omnifold import omnifold
from helpers.read_data import load_multiple_events, match_events, EventObjects

gen_DIR = "/data/atlas/users/rvanrhee/hww_parquet/full_selection_2/gen/"
reco_DIR = "/data/atlas/users/rvanrhee/hww_parquet/full_selection_2/reco/"

print("Loading \t(0/2)")
gen_events = load_multiple_events(DIR=gen_DIR)
print("Gen loaded \t(1/2)")
reco_events = load_multiple_events(DIR=reco_DIR)
print("Reco loaded \t(2/2)")

gen_mask, reco_mask = match_events(gen_events, reco_events)

gen_events = gen_events[gen_mask]
extra_reco = reco_events[~reco_mask]
reco_events = reco_events[reco_mask]


def make_features(events: EventObjects) -> np.ndarray:
    """Create fixed-size event features from leading/subleading jets, muon, and electron."""

    leading_jet    = events.jets[:, 0]
    subleading_jet = events.jets[:, 1]
    muon           = events.muons[:, 0]
    electron       = events.electrons[:, 0]

    features = [
        # Leading jet
        ak.to_numpy(leading_jet.pt),
        ak.to_numpy(leading_jet.eta),
        ak.to_numpy(np.sin(leading_jet.phi)),
        ak.to_numpy(np.cos(leading_jet.phi)),

        # Subleading jet
        ak.to_numpy(subleading_jet.pt),
        ak.to_numpy(subleading_jet.eta),
        ak.to_numpy(np.sin(subleading_jet.phi)),
        ak.to_numpy(np.cos(subleading_jet.phi)),

        # Muon
        ak.to_numpy(muon.pt),
        ak.to_numpy(muon.eta),
        ak.to_numpy(np.sin(muon.phi)),
        ak.to_numpy(np.cos(muon.phi)),

        # Electron
        ak.to_numpy(electron.pt),
        ak.to_numpy(electron.eta),
        ak.to_numpy(np.sin(electron.phi)),
        ak.to_numpy(np.cos(electron.phi)),
    ]

    return np.column_stack(features)




reco_features = make_features(reco_events)
gen_features = make_features(gen_events)

theta0 = np.stack([gen_features, reco_features], axis=1)
theta_unknown_S = make_features(extra_reco[:300000])

scaler = StandardScaler()
scaler.fit(np.concatenate([gen_features, reco_features], axis=0))

theta0[:, 0, :] = scaler.transform(theta0[:, 0, :])
theta0[:, 1, :] = scaler.transform(theta0[:, 1, :])
theta_unknown_S = scaler.transform(theta_unknown_S)

tf.keras.utils.set_random_seed(42)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(theta0.shape[-1],), name="features"),

        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),

        # OmniFold reweight() uses f / (1 - f), so this must be a probability.
        tf.keras.layers.Dense(1, activation="sigmoid", name="classification"),
    ],
    name="omnifold_classifier",
)

model.summary()


# 1. Init run
wb.init_omnifold_run(iterations=2, 
                     theta0=theta0, 
                     theta_unknown_S=theta_unknown_S,
                     run_name=None)

# 2. Define callback factory
def make_callbacks(iteration):
    return [
        [wb.WandbOmniFoldCallback(iteration, step=1)],
        [wb.WandbOmniFoldCallback(iteration, step=2)],
    ]

# 3. Run OmniFold
weights = omnifold(theta0, theta_unknown_S, 
                   iterations=3, 
                   model=model, 
                   callbacks_fn=make_callbacks,
                   verbose=True)

# 4. Log final weights
for i in range(3):
    wb.log_weights(i, weights[i, 0, :], weights[i, 1, :])

wb.wandb.finish()

import matplotlib.pyplot as plt

final_reco_weights = np.asarray(weights[-1, 0, :])
final_truth_weights = np.asarray(weights[-1, 1, :])

assert np.isfinite(final_reco_weights).all()
assert np.isfinite(final_truth_weights).all()

reco_prior_eta = ak.to_numpy(reco_events.jets[:, 0].eta)
data_reco_eta   = ak.to_numpy(extra_reco.jets[:, 0].eta)
truth_prior_eta = ak.to_numpy(gen_events.jets[:, 0].eta)

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

bins = np.linspace(-4.5, 4.5, 41)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

# Detector-level: Step 1 should reweight reco prior towards data.
axes[0].hist(reco_prior_eta, bins=bins, density=True, histtype="step", linewidth=2, label="Reco prior, unweighted")
axes[0].hist(data_reco_eta,  bins=bins, density=True, histtype="step", linewidth=2, label="Reco data")
axes[0].hist(reco_prior_eta, bins=bins, density=True, histtype="step", linewidth=2, label="Reco prior, OmniFold Step 1", weights=final_reco_weights)
axes[0].set_title("Detector-level validation")
axes[0].set_xlabel(r"Leading-jet $\eta$")
axes[0].set_ylabel("Normalised events")
axes[0].legend(fontsize=9)

# Particle-level: Step 2 weights applied to truth prior.
axes[1].hist(truth_prior_eta, bins=bins, density=True, histtype="step", linewidth=2, label="Truth prior, unweighted")
axes[1].hist(truth_prior_eta, bins=bins, density=True, histtype="step", linewidth=2, label="Unfolded truth, OmniFold Step 2", weights=final_truth_weights)
axes[1].set_title("Particle-level unfolding")
axes[1].set_xlabel(r"Leading-jet $\eta$")
axes[1].set_ylabel("Normalised events")
axes[1].legend(fontsize=9)

# Weight distribution.
axes[2].hist(final_truth_weights, bins=80, histtype="stepfilled", alpha=0.7)
axes[2].axvline(1.0, color="black", linestyle="--", label="Weight = 1")
axes[2].set_title("Final particle-level weights")
axes[2].set_xlabel("OmniFold weight")
axes[2].set_ylabel("Prior events")
axes[2].legend()

fig.suptitle("OmniFold unfolding — leading jet $\\eta$", y=1.03)
fig.tight_layout()
fig.savefig("omnifold_unfolding_leading_jet_eta.png", dpi=200, bbox_inches="tight")
plt.show()