from typing import List, Callable
import numpy as np
import awkward as ak
from sklearn.preprocessing import StandardScaler
from helpers.read_data import EventObjects

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
    ) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
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
