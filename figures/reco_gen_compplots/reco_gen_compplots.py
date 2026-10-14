import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

from typing import Optional
import helpers.read_data as rd
import helpers.plotting_features as plotting
import numpy as np

import time

gen_DIR = "/data/atlas/users/rvanrhee/hww_parquet/full_selection/gen/"
reco_DIR = "/data/atlas/users/rvanrhee/hww_parquet/full_selection/reco/"
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


# --- Matching masks ---
gen_mask, reco_mask = rd.match_events(gen_events, reco_events)

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
    xlim=(None, 450),
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
    xlim=(None, 450),
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
    xlim=(None, 75)
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
    xlim=(None, 75)
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
    xlim=(None, 150),
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
    xlim=(None, 150),
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


def higgs_mass(event, m_is0 = True):

    dilepton = event.muons[:, 0] + event.electrons[:, 0]

    if m_is0:
        m_miss = 0
    else:
        m_miss = dilepton.mass

    E_Tll = np.sqrt(dilepton.pt**2 + dilepton.mass**2)
    E_Tmiss = np.sqrt(event.met.pt**2 + m_miss**2)

    return np.sqrt(dilepton.mass**2 + m_miss**2 + 2*(E_Tll*E_Tmiss - dilepton.pt*event.met.pt*np.cos(dilepton.phi - event.met.phi)))

    


plotting.plot_comparison_hist(
    data=[higgs_mass(reco_events), higgs_mass(gen_events)],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/higgs_mass_mis0_unmatched",
    title=r"Higgs proxy mass (unmatched) ($m_{miss} = 0$)",
    xlabel="Mass [GeV]",
    bins=300,
    density=True,
    ratio=True,
    xlim=(0, 500),
)
plotting.plot_comparison_hist(
    data=[higgs_mass(reco_events)[reco_mask], higgs_mass(gen_events)[gen_mask]],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/higgs_mass_mis0_matched",
    title=r"Higgs proxy mass (matched) ($m_{miss} = 0$)",
    xlabel="Mass [GeV]",
    bins=300,
    density=True,
    ratio=True,
    xlim=(0, 500),
)

plotting.plot_comparison_hist(
    data=[higgs_mass(reco_events, m_is0=False), higgs_mass(gen_events, m_is0=False)],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/higgs_mass_mnot0_unmatched",
    title=r"Higgs proxy mass (unmatched) ($m_{miss} = m_{\ell\ell}$)",
    xlabel="Mass [GeV]",
    bins=300,
    density=True,
    ratio=True,
    xlim=(0, 500),
)
plotting.plot_comparison_hist(
    data=[higgs_mass(reco_events, m_is0=False)[reco_mask], higgs_mass(gen_events, m_is0=False)[gen_mask]],
    datanames=["Reco", "Gen"],
    filename="/user/rvanrhee/projects_rik/HWW-unfolding-analysis/figures/reco_gen_compplots/higgs_mass_mnot0_matched",
    title=r"Higgs proxy mass (matched) ($m_{miss} = m_{\ell\ell}$)",
    xlabel="Mass [GeV]",
    bins=300,
    density=True,
    ratio=True,
    xlim=(0, 500),
)

t3 = time.time()
print(f"Plots saved in {t3 - t2:.2f}s")