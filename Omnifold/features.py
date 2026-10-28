"""
Omnifold/features.py
====================
Feature extraction and input preparation for the OmniFold unfolding algorithm.

This module provides three groups of utilities:

1. Feature extraction
   - ``higgs_mass``      : compute the transverse Higgs mass from an event.
   - ``make_cfg``        : build a list of extractor functions from a list of feature names.
   - ``make_features``   : apply a feature config to an EventObjects instance → numpy array.

2. OmniFold input preparation
   - ``make_theta_and_scale`` : scale gen/reco/data arrays and pack them into the
                                theta0 / theta_unknown_S format expected by omnifold().

3. Plotting support
   - ``FEATURE_META``    : lookup table mapping feature names to (extractor, bins, label).
   - ``build_feature_rows`` : build the list-of-dicts consumed by the plotting functions
                              in Omnifold/plotting.py.
"""

from typing import List, Callable, Tuple
import numpy as np
import awkward as ak
from sklearn.preprocessing import StandardScaler
from helpers.read_data import EventObjects

# ---------------------------------------------------------------------------
# 1.  Physics observables
# ---------------------------------------------------------------------------

def higgs_mass(event: EventObjects, m_is0: bool = True) -> ak.Array:
    """Compute the transverse Higgs mass for each event.

    Uses the collinear approximation for the missing transverse energy, with
    two choices for the neutrino mass:

    - ``m_is0=True``  (default): treat the MET as massless (m_miss = 0).
    - ``m_is0=False``           : set m_miss = m_ll (dilepton mass).

    Parameters
    ----------
    event : EventObjects
        Must contain ``muons``, ``electrons``, and ``met`` fields with
        four-momentum components ``pt``, ``eta``, ``phi``, ``mass``.
    m_is0 : bool
        If True, assume massless MET; otherwise use the dilepton mass.

    Returns
    -------
    ak.Array
        Transverse Higgs mass in GeV, one entry per event.
    """
    dilepton = event.muons[:, 0] + event.electrons[:, 0]
    
    if m_is0:
        m_miss = 0
    else:
        m_miss = dilepton.mass

    E_Tll = np.sqrt(dilepton.pt**2 + dilepton.mass**2)
    E_Tmiss = np.sqrt(event.met.pt**2 + m_miss**2)

    return np.sqrt(dilepton.mass**2 + m_miss**2 + 2*(E_Tll*E_Tmiss - dilepton.pt*event.met.pt*np.cos(dilepton.phi - event.met.phi)))

# ---------------------------------------------------------------------------
# 2.  Feature extraction
# ---------------------------------------------------------------------------

def make_cfg(features: List[str]) -> List[Callable]:
    """Build a list of extractor functions from a list of feature name strings.

    Each returned function takes an ``EventObjects`` instance and returns the
    corresponding awkward array (not yet converted to numpy).  The list is
    intended to be passed to ``make_features``.

    Supported feature names:
    ``j1_pt``, ``j1_eta``, ``j1_phi``,
    ``j2_pt``, ``j2_eta``, ``j2_phi``,
    ``electron_pt``, ``electron_eta``, ``electron_phi``,
    ``muon_pt``, ``muon_eta``, ``muon_phi``,
    ``met_pt``, ``higgs_mass``.

    Unknown names are skipped with a warning.
    """
    _map = {
        "j1_pt":        lambda e: e.jets[:, 0].pt,
        "j1_eta":       lambda e: e.jets[:, 0].eta,
        "j1_phi":       lambda e: e.jets[:, 0].phi,
        "j2_pt":        lambda e: e.jets[:, 1].pt,
        "j2_eta":       lambda e: e.jets[:, 1].eta,
        "j2_phi":       lambda e: e.jets[:, 1].phi,
        "muon_pt":      lambda e: e.muons[:, 0].pt,
        "muon_eta":     lambda e: e.muons[:, 0].eta,
        "muon_phi":     lambda e: e.muons[:, 0].phi,
        "electron_pt":  lambda e: e.electrons[:, 0].pt,
        "electron_eta": lambda e: e.electrons[:, 0].eta,
        "electron_phi": lambda e: e.electrons[:, 0].phi,
        "met_pt":       lambda e: e.met.pt,
        "higgs_mass":   lambda e: higgs_mass(e),
    }

    cfg = []
    for name in features:
        if name in _map:
            cfg.append(_map[name])
        else:
            print(f"Warning: unknown feature '{name}', skipping.")
    return cfg


def make_features(events: EventObjects, feature_cfg: List[Callable]) -> np.ndarray:
    """Apply a feature config to an EventObjects instance and return a numpy matrix.

    Parameters
    ----------
    events : EventObjects
        The event collection to extract features from.
    feature_cfg : List[Callable]
        A list of extractor functions, as returned by ``make_cfg``.

    Returns
    -------
    np.ndarray
        Shape ``(N, n_features)``.

    Raises
    ------
    ValueError
        If ``feature_cfg`` is empty.
    """
    if not feature_cfg:
        raise ValueError("feature_cfg is empty — pass at least one extractor function.")
    return np.column_stack([ak.to_numpy(fn(events)) for fn in feature_cfg])

# ---------------------------------------------------------------------------
# 3.  OmniFold input preparation
# ---------------------------------------------------------------------------

def make_theta_and_scale(
    gen:      np.ndarray,
    reco:     np.ndarray,
    data_reco: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale features and pack them into the OmniFold input format.

    Fits a ``StandardScaler`` on the combined gen+reco simulation sample,
    then transforms all three arrays.  The scaler is returned so it can be
    reused for inference or saved alongside the weights.

    Parameters
    ----------
    gen : np.ndarray, shape (N, n_features)
        Generator-level simulation features.
    reco : np.ndarray, shape (N, n_features)
        Matched detector-level simulation features.
    data_reco : np.ndarray, shape (M, n_features)
        Detector-level data (or pseudo-data) features.

    Returns
    -------
    theta0 : np.ndarray, shape (N, 2, n_features)
        Stacked ``[gen, reco]`` array expected by ``omnifold()``.
    theta_unknown_S : np.ndarray, shape (M, n_features)
        Scaled data array expected by ``omnifold()``.
    scaler : StandardScaler
        The fitted scaler.
    """
    scaler = StandardScaler()
    scaler.fit(np.concatenate([gen, reco], axis=0))

    theta0 = np.stack([
        scaler.transform(gen),
        scaler.transform(reco),
    ], axis=1)

    theta_unknown_S = scaler.transform(data_reco)

    return theta0, theta_unknown_S, scaler

# ---------------------------------------------------------------------------
# 4.  Plotting support
# ---------------------------------------------------------------------------

# Maps each feature name to a tuple of:
#   (extractor: EventObjects -> np.ndarray, bins: np.ndarray, axis label: str)
# Used by build_feature_rows() to avoid repeating bin/label definitions at
# every call site.
FEATURE_META = {
    "j1_pt":       (lambda e: ak.to_numpy(e.jets[:, 0].pt),        np.linspace(0,      500, 41), r"Leading-jet $p_T$ [GeV]"),
    "j1_eta":      (lambda e: ak.to_numpy(e.jets[:, 0].eta),       np.linspace(-4.5,   4.5, 41), r"Leading-jet $\eta$"),
    "j1_phi":      (lambda e: ak.to_numpy(e.jets[:, 0].phi),       np.linspace(-np.pi, np.pi, 41), r"Leading-jet $\phi$"),
    "j2_pt":       (lambda e: ak.to_numpy(e.jets[:, 1].pt),        np.linspace(0,      300, 41), r"Subleading-jet $p_T$ [GeV]"),
    "j2_eta":      (lambda e: ak.to_numpy(e.jets[:, 1].eta),       np.linspace(-4.5,   4.5, 41), r"Subleading-jet $\eta$"),
    "j2_phi":      (lambda e: ak.to_numpy(e.jets[:, 1].phi),       np.linspace(-np.pi, np.pi, 41), r"Subleading-jet $\phi$"),
    "electron_pt": (lambda e: ak.to_numpy(e.electrons[:, 0].pt),   np.linspace(0,      300, 41), r"Electron $p_T$ [GeV]"),
    "electron_eta":(lambda e: ak.to_numpy(e.electrons[:, 0].eta),  np.linspace(-4.5,   4.5, 41), r"Electron $\eta$"),
    "electron_phi":(lambda e: ak.to_numpy(e.electrons[:, 0].phi),  np.linspace(-np.pi, np.pi, 41), r"Electron $\phi$"),
    "muon_pt":     (lambda e: ak.to_numpy(e.muons[:, 0].pt),       np.linspace(0,      300, 41), r"Muon $p_T$ [GeV]"),
    "muon_eta":    (lambda e: ak.to_numpy(e.muons[:, 0].eta),      np.linspace(-4.5,   4.5, 41), r"Muon $\eta$"),
    "muon_phi":    (lambda e: ak.to_numpy(e.muons[:, 0].phi),      np.linspace(-np.pi, np.pi, 41), r"Muon $\phi$"),
    "higgs_mass":  (lambda e: ak.to_numpy(higgs_mass(e)),          np.linspace(0,      200, 41), r"Higgs mass $m_H$ [GeV]"),
}


def build_feature_rows(
    features:            List[str],
    prior_gen_events:    EventObjects,
    prior_reco_events:   EventObjects,
    data_gen_events:     EventObjects,
    data_reco_events:    EventObjects,
    final_reco_weights:  np.ndarray,
    final_truth_weights: np.ndarray,
) -> list:
    """Build the list of row-dicts consumed by the plotting functions.

    Each dict in the returned list corresponds to one feature and contains
    all arrays and metadata needed by ``plot_feature_row`` and
    ``plot_closure_summary_row`` in ``Omnifold/plotting.py``.

    Features not present in ``FEATURE_META`` are skipped with a warning.

    Parameters
    ----------
    features : List[str]
        Ordered list of feature names to include; must be keys of FEATURE_META.
    prior_gen_events, prior_reco_events : EventObjects
        Matched simulation events (prior half in a closure test).
    data_gen_events, data_reco_events : EventObjects
        Data events (pseudo-data half in a closure test, or real data).
    final_reco_weights : np.ndarray, shape (N,)
        OmniFold Step-1 weights (detector-level).
    final_truth_weights : np.ndarray, shape (N,)
        OmniFold Step-2 weights (particle-level).

    Returns
    -------
    list of dict
        One dict per feature, ready to be unpacked into the plot functions.
    """
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
