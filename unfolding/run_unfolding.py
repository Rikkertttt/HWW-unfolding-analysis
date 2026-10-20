import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

import numpy as np
import tensorflow as tf
import json

import Omnifold.wandb_helpers as wb
from Omnifold.omnifold import omnifold
from helpers.read_data import load_multiple_events, match_events
from Omnifold.models import build_omnifold_classifier
from Omnifold.features import make_features, make_cfg, make_theta_and_scale

gen_DIR  = "/data/atlas/users/rvanrhee/hww_parquet/full_selection/gen/"
reco_DIR = "/data/atlas/users/rvanrhee/hww_parquet/full_selection/reco/"
gen_wwjj_DIR = "/data/atlas/users/rvanrhee/wwjj_parquet/full_selection/gen/"
reco_wwjj_DIR = "/data/atlas/users/rvanrhee/wwjj_parquet/full_selection/reco/"

OUTDIR = "/user/rvanrhee/projects_rik/HWW-unfolding-analysis/unfolding/newpipelinetest/"
os.makedirs(OUTDIR, exist_ok=True)

print("Loading \t(0/4)")
gen_events  = load_multiple_events(DIR=gen_DIR)
print("Gen loaded \t(1/4)")
reco_events = load_multiple_events(DIR=reco_DIR)
print("Reco loaded \t(2/4)")
gen_wwjj_events = load_multiple_events(DIR=gen_wwjj_DIR, max_files=300)
print("Gen WWJJ loaded \t(3/4)")
reco_wwjj_events = load_multiple_events(DIR=reco_wwjj_DIR, max_files=300)
print("Reco WWJJ loaded \t(4/4)")

print(f"Gen events:  {len(gen_events)}")
print(f"Reco events: {len(reco_events)}")
print(f"Gen WWJJ events:  {len(gen_wwjj_events)}")
print(f"Reco WWJJ events: {len(reco_wwjj_events)}")

gen_mask, reco_mask = match_events(gen_events, reco_events)
gen_wwjj_mask, reco_wwjj_mask = match_events(gen_wwjj_events, reco_wwjj_events)

# Keep only matched events
gen_events  = gen_events[gen_mask]
reco_events = reco_events[reco_mask]
gen_wwjj_events  = gen_wwjj_events[gen_wwjj_mask]
reco_wwjj_events = reco_wwjj_events[reco_wwjj_mask]

print(f"Matched events: {len(gen_events)}")
print(f"Matched WWJJ events: {len(gen_wwjj_events)}")


features = ["j1_pt", "j1_eta", "j1_phi", "j2_pt", "j2_eta", "j2_phi"]
feature_cfg = make_cfg(features)

reco_features = make_features(reco_events, feature_cfg)
gen_features  = make_features(gen_events, feature_cfg)
reco_wwjj_features = make_features(reco_wwjj_events, feature_cfg)
gen_wwjj_features = make_features(gen_wwjj_events, feature_cfg)


theta0, theta_unknown_S, _ = make_theta_and_scale(gen_features, reco_features, reco_wwjj_features)

tf.keras.utils.set_random_seed(42)

ITERATIONS = 2

model = build_omnifold_classifier(n_features=theta0.shape[-1])

model.summary()

# 1. Init W&B run
wb.init_omnifold_run(
    iterations=ITERATIONS,
    theta0=theta0,
    theta_unknown_S=theta_unknown_S,
    run_name="wwjj_new_eval_test",
    extra_config={"data": "wwjj", "closure_test": False},
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
np.save(os.path.join(OUTDIR, "gen_features.npy"),       gen_features)
np.save(os.path.join(OUTDIR, "reco_features.npy"),      reco_features)
np.save(os.path.join(OUTDIR, "gen_wwjj_features.npy"),  gen_wwjj_features)
np.save(os.path.join(OUTDIR, "reco_wwjj_features.npy"), reco_wwjj_features)

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



# # Define features to plot: (column index, label, bins)
# plot_features = [
#     (0, r"Leading-jet $p_T$ [GeV]",      np.linspace(0,    500, 41)),
#     (1, r"Leading-jet $\eta$",            np.linspace(-4.5, 4.5, 41)),
#     (3, r"Subleading-jet $p_T$ [GeV]",   np.linspace(0,    300, 41)),
#     (4, r"Subleading-jet $\eta$",         np.linspace(-4.5, 4.5, 41)),
# ]

# n_features = len(plot_features)
# fig, axes = plt.subplots(n_features, 3, figsize=(17, 4.5 * n_features))

# for row, (col, label, bins) in enumerate(plot_features):
#     reco_prior  = reco_features[:,  col]
#     truth_prior = gen_features [:,  col]
#     data_reco   = reco_wwjj_features[:, col]
#     data_truth  = gen_wwjj_features[:, col]

#     # Step 1: detector-level
#     axes[row, 0].hist(reco_prior, bins=bins, density=True, histtype="step", linewidth=2, label="Reco sim (HWW), unweighted")
#     axes[row, 0].hist(data_reco,  bins=bins, density=True, histtype="step", linewidth=2, label="Reco data (wwjj)")
#     axes[row, 0].hist(reco_prior, bins=bins, density=True, histtype="step", linewidth=2, label="Reco sim (HWW), Step 1", weights=final_reco_weights)
#     axes[row, 0].set_xlabel(label)
#     axes[row, 0].set_ylabel("Normalised events")
#     axes[row, 0].set_title(f"Detector-level — {label}")
#     axes[row, 0].legend(fontsize=9)

#     # Step 2: particle-level
#     axes[row, 1].hist(truth_prior, bins=bins, density=True, histtype="step", linewidth=2, label="Truth prior (HWW), unweighted")
#     axes[row, 1].hist(data_truth,  bins=bins, density=True, histtype="step", linewidth=2, label="Truth data (wwjj)")
#     axes[row, 1].hist(truth_prior, bins=bins, density=True, histtype="step", linewidth=2, label="Unfolded truth, Step 2", weights=final_truth_weights)
#     axes[row, 1].set_xlabel(label)
#     axes[row, 1].set_ylabel("Normalised events")
#     axes[row, 1].set_title(f"Particle-level — {label}")
#     axes[row, 1].legend(fontsize=9)

#     # Weight distribution (same for all rows, but useful to repeat)
#     axes[row, 2].hist(final_truth_weights, bins=80, histtype="stepfilled", alpha=0.7)
#     axes[row, 2].axvline(1.0, color="black", linestyle="--", label="Weight = 1")
#     axes[row, 2].set_title("Final particle-level weights")
#     axes[row, 2].set_xlabel("OmniFold weight")
#     axes[row, 2].set_ylabel("Events")
#     axes[row, 2].legend()

# fig.suptitle("OmniFold closure test", y=1.01)
# fig.tight_layout()
# fig.savefig(f"{OUTDIR}/omnifold_closure_test.pdf", bbox_inches="tight")
# fig.savefig(f"{OUTDIR}/omnifold_closure_test.png", bbox_inches="tight")
# plt.show()

# fig2, axes2 = plt.subplots(
#     2, n_features,
#     figsize=(5 * n_features, 6),
#     gridspec_kw={"height_ratios": [3, 1]},
# )

# for col_idx, (col, label, bins) in enumerate(plot_features):
#     data_reco   = reco_wwjj_features[:, col]
#     data_truth  = gen_wwjj_features [:, col]
#     truth_prior = gen_features [:,  col]

#     bin_centers = 0.5 * (bins[:-1] + bins[1:])
#     counts_truth,    _ = np.histogram(data_truth,  bins=bins, density=True)
#     counts_unfolded, _ = np.histogram(truth_prior, bins=bins, density=True, weights=final_truth_weights)

#     # Main plot
#     ax = axes2[0, col_idx]
#     ax.hist(data_reco,   bins=bins, density=True, histtype="step", linewidth=2, label="Reco data (wwjj)")
#     ax.hist(data_truth,  bins=bins, density=True, histtype="step", linewidth=2, label="Truth data (wwjj)")
#     ax.hist(truth_prior, bins=bins, density=True, histtype="step", linewidth=2, label="Unfolded (Step 2)", weights=final_truth_weights)
#     ax.set_ylabel("Normalised events")
#     ax.set_title(label)
#     ax.legend(fontsize=9)
#     ax.set_xticks([])  # hide x-ticks on main, shared with ratio

#     # Ratio plot
#     ax_ratio = axes2[1, col_idx]
#     ratio = np.where(counts_truth > 0, counts_unfolded / counts_truth, np.nan)
#     ax_ratio.plot(bin_centers, ratio, "o", markersize=3)
#     ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1)
#     ax_ratio.set_ylim(0.5, 1.5)
#     ax_ratio.set_xlabel(label)
#     ax_ratio.set_ylabel("Unf. / Truth")

# fig2.suptitle("OmniFold closure — summary", y=1.01)
# fig2.tight_layout()
# fig2.savefig(f"{OUTDIR}/omnifold_closure_summary.pdf", bbox_inches="tight")
# fig2.savefig(f"{OUTDIR}/omnifold_closure_summary.png", bbox_inches="tight")
# plt.show()