from typing import List, Callable
import numpy as np
import awkward as ak
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
        else:
            print(f"Unknown feature: {feature}, skipping.")
    return cfg