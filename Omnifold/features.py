from typing import List, Callable, Tuple
import numpy as np
import awkward as ak
from sklearn.preprocessing import StandardScaler
from helpers.read_data import EventObjects

def higgs_mass(event, m_is0 = True):
    """
    Compute the Higgs mass from an event. Choose either m_met = 0 or m_met = m_ll.
    """
    dilepton = event.muons[:, 0] + event.electrons[:, 0]
    
    if m_is0:
        m_miss = 0
    else:
        m_miss = dilepton.mass

    E_Tll = np.sqrt(dilepton.pt**2 + dilepton.mass**2)
    E_Tmiss = np.sqrt(event.met.pt**2 + m_miss**2)

    return np.sqrt(dilepton.mass**2 + m_miss**2 + 2*(E_Tll*E_Tmiss - dilepton.pt*event.met.pt*np.cos(dilepton.phi - event.met.phi)))


def make_features(events: EventObjects, feature_cfg: List[Callable]) -> np.ndarray:
    """
    Create a feature matrix from the given events using the specified feature configuration.
    feature_cfg is a list of functions that take an EventObjects instance and return a feature array.
    """
    if not feature_cfg:
        raise ValueError("Feature configuration is empty. Please provide a valid feature configuration.")
    return np.column_stack([ak.to_numpy(fn(events)) for fn in feature_cfg])

def make_cfg(features: List[str]) -> List[Callable]:
    """
    Create a default feature configuration for the Omnifold classifier.
    This function returns a list of functions that extract features from EventObjects.
    """

    cfg = []
    for feature in features:
        if feature == "j1_pt":
            cfg.append(lambda e: e.jets[:, 0].pt)
        elif feature == "j1_eta":
            cfg.append(lambda e: e.jets[:, 0].eta)
        elif feature == "j1_phi":
            cfg.append(lambda e: e.jets[:, 0].phi)
        elif feature == "j2_pt":
            cfg.append(lambda e: e.jets[:, 1].pt)
        elif feature == "j2_eta":
            cfg.append(lambda e: e.jets[:, 1].eta)
        elif feature == "j2_phi":
            cfg.append(lambda e: e.jets[:, 1].phi)
        elif feature == "muon_pt":
            cfg.append(lambda e: e.muons[:, 0].pt)
        elif feature == "muon_eta":
            cfg.append(lambda e: e.muons[:, 0].eta)
        elif feature == "muon_phi":
            cfg.append(lambda e: e.muons[:, 0].phi)
        elif feature == "electron_pt":
            cfg.append(lambda e: e.electrons[:, 0].pt)
        elif feature == "electron_eta":
            cfg.append(lambda e: e.electrons[:, 0].eta)
        elif feature == "electron_phi":
            cfg.append(lambda e: e.electrons[:, 0].phi)
        elif feature == "met_pt":
            cfg.append(lambda e: e.met.pt)
        elif (feature == "higgs_mass"):
            cfg.append(lambda e: higgs_mass(e))
        else:
            print(f"Unknown feature: {feature}, skipping.")
    return cfg

def make_theta_and_scale(
        gen: np.ndarray,
        reco: np.ndarray,
        data_reco: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        Create the theta0 and theta_unknown_S arrays for the Omnifold algorithm, and fit a StandardScaler to the combined gen and reco data.
        """

        scaler = StandardScaler()
        scaler.fit(np.concatenate([gen, reco], axis=0))

        theta0 = np.stack([
            scaler.transform(gen),
            scaler.transform(reco),
        ], axis=1)  # shape (N, 2, n_features)

        theta_unknown_S = scaler.transform(data_reco)

        return theta0, theta_unknown_S, scaler


# Lookup table: feature name -> (extractor, bins, label)
FEATURE_META = {
    "j1_pt":       (lambda e: ak.to_numpy(e.jets[:, 0].pt),         np.linspace(0,    500, 41), r"Leading-jet $p_T$ [GeV]"),
    "j1_eta":      (lambda e: ak.to_numpy(e.jets[:, 0].eta),        np.linspace(-4.5, 4.5, 41), r"Leading-jet $\eta$"),
    "j1_phi":      (lambda e: ak.to_numpy(e.jets[:, 0].phi),        np.linspace(-np.pi, np.pi, 41), r"Leading-jet $\phi$"),
    "j2_pt":       (lambda e: ak.to_numpy(e.jets[:, 1].pt),         np.linspace(0,    300, 41), r"Subleading-jet $p_T$ [GeV]"),
    "j2_eta":      (lambda e: ak.to_numpy(e.jets[:, 1].eta),        np.linspace(-4.5, 4.5, 41), r"Subleading-jet $\eta$"),
    "j2_phi":      (lambda e: ak.to_numpy(e.jets[:, 1].phi),        np.linspace(-np.pi, np.pi, 41), r"Subleading-jet $\phi$"),
    "electron_pt": (lambda e: ak.to_numpy(e.electrons[:, 0].pt),    np.linspace(0,    300, 41), r"Electron $p_T$ [GeV]"),
    "electron_eta":(lambda e: ak.to_numpy(e.electrons[:, 0].eta),   np.linspace(-4.5, 4.5, 41), r"Electron $\eta$"),
    "electron_phi":(lambda e: ak.to_numpy(e.electrons[:, 0].phi),   np.linspace(-np.pi, np.pi, 41), r"Electron $\phi$"),
    "muon_pt":     (lambda e: ak.to_numpy(e.muons[:, 0].pt),        np.linspace(0,    300, 41), r"Muon $p_T$ [GeV]"),
    "muon_eta":    (lambda e: ak.to_numpy(e.muons[:, 0].eta),       np.linspace(-4.5, 4.5, 41), r"Muon $\eta$"),
    "muon_phi":    (lambda e: ak.to_numpy(e.muons[:, 0].phi),       np.linspace(-np.pi, np.pi, 41), r"Muon $\phi$"),
    "higgs_mass":  (lambda e: ak.to_numpy(higgs_mass(e)),           np.linspace(0,    200, 41), r"Higgs mass $m_H$ [GeV]"),
}


def build_feature_rows(
    features:            List[str],
    prior_gen_events,
    prior_reco_events,
    data_gen_events,
    data_reco_events,
    final_reco_weights:  np.ndarray,
    final_truth_weights: np.ndarray,
) -> list:
    rows = []
    for feature in features:
        if feature not in FEATURE_META:
            print(f"Warning: '{feature}' not in FEATURE_META, skipping.")
            continue
        extract, bins, label = FEATURE_META[feature]
        rows.append(dict(
            truth_prior         = extract(prior_gen_events),
            reco_prior          = extract(prior_reco_events),
            data_truth          = extract(data_gen_events),
            data_reco           = extract(data_reco_events),
            final_reco_weights  = final_reco_weights,
            final_truth_weights = final_truth_weights,
            bins                = bins,
            label               = label,
        ))
    return rows
