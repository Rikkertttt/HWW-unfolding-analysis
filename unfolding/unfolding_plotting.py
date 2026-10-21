import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

import numpy as np
import awkward as ak
from Omnifold.plotting import open_saved_weights, plot_omnifold_overview, plot_closure_summary
from Omnifold.features import higgs_mass
from helpers.read_data import load_events

# Path where weights and matched parquet files are saved
PATH = "/user/rvanrhee/projects_rik/HWW-unfolding-analysis/unfolding/newpipelinetest"

# Load pre-matched events (no reload/rematch needed)
gen_events       = load_events(os.path.join(PATH, "prior_gen.parquet"))
reco_events      = load_events(os.path.join(PATH, "prior_reco.parquet"))
gen_wwjj_events  = load_events(os.path.join(PATH, "data_gen.parquet"))
reco_wwjj_events = load_events(os.path.join(PATH, "data_reco.parquet"))

# Load weights
weights, _ = open_saved_weights(PATH=PATH)

final_reco_weights  = weights[-1, 0, :]
final_truth_weights = weights[-1, 1, :]

# Compute observables directly from EventObjects
rows = [
    dict(
        truth_prior        = ak.to_numpy(gen_events.jets[:, 0].pt),
        reco_prior         = ak.to_numpy(reco_events.jets[:, 0].pt),
        data_truth         = ak.to_numpy(gen_wwjj_events.jets[:, 0].pt),
        data_reco          = ak.to_numpy(reco_wwjj_events.jets[:, 0].pt),
        final_reco_weights = final_reco_weights,
        final_truth_weights= final_truth_weights,
        bins               = np.linspace(0, 500, 41),
        label              = r"Leading-jet $p_T$ [GeV]",
    ),
    dict(
        truth_prior        = ak.to_numpy(gen_events.jets[:, 0].eta),
        reco_prior         = ak.to_numpy(reco_events.jets[:, 0].eta),
        data_truth         = ak.to_numpy(gen_wwjj_events.jets[:, 0].eta),
        data_reco          = ak.to_numpy(reco_wwjj_events.jets[:, 0].eta),
        final_reco_weights = final_reco_weights,
        final_truth_weights= final_truth_weights,
        bins               = np.linspace(-4.5, 4.5, 41),
        label              = r"Leading-jet $\eta$",
    ),
    dict(
        truth_prior        = ak.to_numpy(gen_events.jets[:, 1].pt),
        reco_prior         = ak.to_numpy(reco_events.jets[:, 1].pt),
        data_truth         = ak.to_numpy(gen_wwjj_events.jets[:, 1].pt),
        data_reco          = ak.to_numpy(reco_wwjj_events.jets[:, 1].pt),
        final_reco_weights = final_reco_weights,
        final_truth_weights= final_truth_weights,
        bins               = np.linspace(0, 300, 41),
        label              = r"Subleading-jet $p_T$ [GeV]",
    ),
    dict(
        truth_prior        = ak.to_numpy(gen_events.jets[:, 1].eta),
        reco_prior         = ak.to_numpy(reco_events.jets[:, 1].eta),
        data_truth         = ak.to_numpy(gen_wwjj_events.jets[:, 1].eta),
        data_reco          = ak.to_numpy(reco_wwjj_events.jets[:, 1].eta),
        final_reco_weights = final_reco_weights,
        final_truth_weights= final_truth_weights,
        bins               = np.linspace(-4.5, 4.5, 41),
        label              = r"Subleading-jet $\eta$",
    ),
    dict(
        truth_prior        = ak.to_numpy(higgs_mass(gen_events)),
        reco_prior         = ak.to_numpy(higgs_mass(reco_events)),
        data_truth         = ak.to_numpy(higgs_mass(gen_wwjj_events)),
        data_reco          = ak.to_numpy(higgs_mass(reco_wwjj_events)),
        final_reco_weights = final_reco_weights,
        final_truth_weights= final_truth_weights,
        bins               = np.linspace(0, 200, 41),
        label              = r"Higgs mass $m_H$ [GeV]",
    ),
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
    title="OmniFold closure summary — WWjj stress test",
)