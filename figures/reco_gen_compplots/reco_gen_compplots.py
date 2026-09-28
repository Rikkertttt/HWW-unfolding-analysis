import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

from typing import Optional
import helpers.read_data as rd
import helpers.plotting_features as plotting
import numpy as np

import time

gen_DIR = "/user/rvanrhee/data_rik/hww_parquet/gen/"
reco_DIR = "/user/rvanrhee/data_rik/hww_parquet/reco/"
gen_files = sorted([f for f in os.listdir(gen_DIR) if f.endswith(".parquet")])
reco_files = sorted([f for f in os.listdir(reco_DIR) if f.endswith(".parquet")])

gen_events: Optional[rd.EventObjects] = None
reco_events: Optional[rd.EventObjects] = None

t0 = time.time()

for file in gen_files:
    loaded = rd.load_events(path=gen_DIR + file)
    if gen_events is None:
        gen_events = loaded
    else:
        gen_events += loaded

t1 = time.time()
print(f"Gen loaded in {t1 - t0:.2f}s")

for file in reco_files:
    loaded = rd.load_events(path=reco_DIR + file)
    if reco_events is None:
        reco_events = loaded
    else:
        reco_events += loaded

t2 = time.time()
print(f"Reco loaded in {t2 - t1:.2f}s")

assert gen_events  is not None
assert reco_events is not None

# --- Higgs proxy (unmatched) ---
gen_higgs  = gen_events.muons[:, 0]  + gen_events.electrons[:, 0]  + gen_events.met
reco_higgs = reco_events.muons[:, 0] + reco_events.electrons[:, 0] + reco_events.met

# --- Matching masks ---
gen_mask, reco_mask = rd.match_gen_reco(gen_events=gen_events, reco_events=reco_events)

# ── Jets ──────────────────────────────────────────────────────────────────────

plotting.plot_comparison_hist(
    data=[reco_events.jets[:, :2].pt, gen_events.jets[:, :2].pt],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/jet_pt_unmatched",
    title="(Sub)Leading Jets pT (unmatched)",
    xlabel="pT [GeV]",
    bins=150,
    density=True,
    ratio=True,
)
plotting.plot_comparison_hist(
    data=[reco_events.jets[:, :2][reco_mask].pt, gen_events.jets[:, :2][gen_mask].pt],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/jet_pt_matched",
    title="(Sub)Leading Jets pT (matched)",
    xlabel="pT [GeV]",
    bins=150,
    density=True,
    ratio=True,
)

plotting.plot_comparison_hist(
    data=[reco_events.jets[:, :2].phi, gen_events.jets[:, :2].phi],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/jet_phi_unmatched",
    title="(Sub)Leading Jets φ (unmatched)",
    xlabel="φ [rad]",
    bins=150,
    density=True,
    ratio=True,
)
plotting.plot_comparison_hist(
    data=[reco_events.jets[:, :2][reco_mask].phi, gen_events.jets[:, :2][gen_mask].phi],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/jet_phi_matched",
    title="(Sub)Leading Jets φ (matched)",
    xlabel="φ [rad]",
    bins=150,
    density=True,
    ratio=True,
)

plotting.plot_comparison_hist(
    data=[reco_events.jets[:, :2].eta, gen_events.jets[:, :2].eta],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/jet_eta_unmatched",
    title="(Sub)Leading Jets η (unmatched)",
    xlabel="η",
    bins=150,
    density=True,
    ratio=True,
)
plotting.plot_comparison_hist(
    data=[reco_events.jets[:, :2][reco_mask].eta, gen_events.jets[:, :2][gen_mask].eta],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/jet_eta_matched",
    title="(Sub)Leading Jets η (matched)",
    xlabel="η",
    bins=150,
    density=True,
    ratio=True,
)

plotting.plot_comparison_hist(
    data=[reco_events.jets[:, :2].mass, gen_events.jets[:, :2].mass],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/jet_mass_unmatched",
    title="(Sub)Leading Jets mass (unmatched)",
    xlabel="Mass [GeV]",
    bins=150,
    density=True,
    ratio=True,
)
plotting.plot_comparison_hist(
    data=[reco_events.jets[:, :2][reco_mask].mass, gen_events.jets[:, :2][gen_mask].mass],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/jet_mass_matched",
    title="(Sub)Leading Jets mass (matched)",
    xlabel="Mass [GeV]",
    bins=150,
    density=True,
    ratio=True,
)

# ── Muons ─────────────────────────────────────────────────────────────────────

plotting.plot_comparison_hist(
    data=[reco_events.muons[:, 0].pt, gen_events.muons[:, 0].pt],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/muon_pt_unmatched",
    title="Leading muon pT (unmatched)",
    xlabel="pT [GeV]",
    bins=150,
    density=True,
    ratio=True,
)
plotting.plot_comparison_hist(
    data=[reco_events.muons[:, 0][reco_mask].pt, gen_events.muons[:, 0][gen_mask].pt],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/muon_pt_matched",
    title="Leading muon pT (matched)",
    xlabel="pT [GeV]",
    bins=150,
    density=True,
    ratio=True,
)

plotting.plot_comparison_hist(
    data=[reco_events.muons[:, 0].eta, gen_events.muons[:, 0].eta],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/muon_eta_unmatched",
    title="Leading muon η (unmatched)",
    xlabel="η",
    bins=150,
    density=True,
    ratio=True,
)
plotting.plot_comparison_hist(
    data=[reco_events.muons[:, 0][reco_mask].eta, gen_events.muons[:, 0][gen_mask].eta],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/muon_eta_matched",
    title="Leading muon η (matched)",
    xlabel="η",
    bins=150,
    density=True,
    ratio=True,
)

# ── Higgs proxy mass ──────────────────────────────────────────────────────────

plotting.plot_comparison_hist(
    data=[reco_higgs.mass, gen_higgs.mass],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/higgs_mass_unmatched",
    title="Higgs proxy mass (unmatched)",
    xlabel="Mass [GeV]",
    bins=150,
    density=True,
    ratio=True,
)
plotting.plot_comparison_hist(
    data=[reco_higgs[reco_mask].mass, gen_higgs[gen_mask].mass],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/higgs_mass_matched",
    title="Higgs proxy mass (matched)",
    xlabel="Mass [GeV]",
    bins=150,
    density=True,
    ratio=True,
)

t3 = time.time()
print(f"Plots saved in {t3 - t2:.2f}s")