from helpers.read_data import load_multiple_events, match_events
from typing import List, Optional

def get_all_data_omnifold(
    prior_reco_events_path: str,
    prior_gen_events_path:  str,
    data_reco_events_path:  str,
    data_gen_events_path:   Optional[str] = None,
    max_files:              List[Optional[int]] = [None, None, None, None],
    verbose:                bool = False,
):
    """
    Load and match prior (sim) and data events for OmniFold.

    Prior events (HWW sim) are matched gen to reco by event ID.
    Data events (e.g. WWjj) are matched the same way if gen is provided.
    """

    # --- Load prior (simulation) ---
    if verbose: print("Loading prior gen  \t(1/4)")
    prior_gen_events  = load_multiple_events(DIR=prior_gen_events_path,  max_files=max_files[0])

    if verbose: print("Loading prior reco \t(2/4)")
    prior_reco_events = load_multiple_events(DIR=prior_reco_events_path, max_files=max_files[1])

    # --- Load data ---
    if verbose: print("Loading data reco  \t(3/4)")
    data_reco_events = load_multiple_events(DIR=data_reco_events_path, max_files=max_files[2])

    if data_gen_events_path is not None:
        if verbose: print("Loading data gen   \t(4/4)")
        data_gen_events = load_multiple_events(DIR=data_gen_events_path, max_files=max_files[3])
    else:
        if verbose: print("No data gen path provided, skipping \t(4/4)")
        data_gen_events = None

    if verbose:
        print(f"\nLoaded events (before matching):")
        print(f"  Prior gen:  {len(prior_gen_events)}")
        print(f"  Prior reco: {len(prior_reco_events)}")
        print(f"  Data reco:  {len(data_reco_events)}")
        if data_gen_events is not None:
            print(f"  Data gen:   {len(data_gen_events)}")

    # --- Match prior gen <-> reco ---
    prior_gen_mask, prior_reco_mask = match_events(prior_gen_events, prior_reco_events)
    prior_gen_events  = prior_gen_events[prior_gen_mask]
    prior_reco_events = prior_reco_events[prior_reco_mask]

    # --- Match data gen <-> reco (if gen is available) ---
    if data_gen_events is not None:
        data_gen_mask, data_reco_mask = match_events(data_gen_events, data_reco_events)
        data_gen_events  = data_gen_events[data_gen_mask]
        data_reco_events = data_reco_events[data_reco_mask]

    if verbose:
        print(f"\nMatched events (after matching):")
        print(f"  Prior: {len(prior_gen_events)}")
        if data_gen_events is not None:
            print(f"  Data:  {len(data_gen_events)}")
        else:
            print(f"  Data reco (unmatched): {len(data_reco_events)}")

    return prior_gen_events, prior_reco_events, data_gen_events, data_reco_events