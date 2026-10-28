from helpers.read_data import load_multiple_events, match_events
from typing import List, Optional

def get_all_data_omnifold(
    prior_reco_events_path: str,
    prior_gen_events_path:  str,
    data_reco_events_path:  Optional[str] = None,
    data_gen_events_path:   Optional[str] = None,
    max_files:              List[Optional[int]] = [None, None, None, None],
    closure_test:           bool = False,
    verbose:                bool = False,
):
    """
    Load and match prior (sim) and data events for OmniFold.

    Prior events (HWW sim) are matched gen↔reco by event ID.

    If closure_test=True, the matched prior reco events are split in half:
    one half is used as the prior, the other as pseudo-data. No external
    data paths are needed in this mode.

    Otherwise, data_reco_events_path must be provided. data_gen_events_path
    is optional and only used for stress tests where truth-level data is available.
    """

    # --- Load prior ---
    if verbose: print("Loading prior gen  \t(1/4)")
    prior_gen_events  = load_multiple_events(DIR=prior_gen_events_path,  max_files=max_files[0])

    if verbose: print("Loading prior reco \t(2/4)")
    prior_reco_events = load_multiple_events(DIR=prior_reco_events_path, max_files=max_files[1])

    # --- Match prior gen <-> reco ---
    prior_gen_mask, prior_reco_mask = match_events(prior_gen_events, prior_reco_events)
    prior_gen_events  = prior_gen_events[prior_gen_mask]
    prior_reco_events = prior_reco_events[prior_reco_mask]

    if verbose:
        print(f"\nMatched prior events: {len(prior_gen_events)}")

    # --- Closure test: split matched prior reco in half ---
    if closure_test:
        if verbose: print("\nClosure test mode: splitting prior reco into prior / pseudo-data")
        n = len(prior_reco_events)
        half = n // 2

        data_gen_events   = prior_gen_events[half:]
        data_reco_events  = prior_reco_events[half:]
        prior_gen_events  = prior_gen_events[:half]
        prior_reco_events = prior_reco_events[:half]

        return prior_gen_events, prior_reco_events, data_gen_events, data_reco_events

    # --- Load external data ---
    if data_reco_events_path is None:
        raise ValueError("data_reco_events_path must be provided when closure_test=False")

    if verbose: print("Loading data reco  \t(3/4)")
    data_reco_events = load_multiple_events(DIR=data_reco_events_path, max_files=max_files[2])

    if data_gen_events_path is not None:
        if verbose: print("Loading data gen   \t(4/4)")
        data_gen_events = load_multiple_events(DIR=data_gen_events_path, max_files=max_files[3])
    else:
        if verbose: print("No data gen path provided, skipping \t(4/4)")
        data_gen_events = None

    if verbose:
        print(f"  Data reco:  {len(data_reco_events)}")
        if data_gen_events is not None:
            print(f"  Data gen:   {len(data_gen_events)}")

    # --- Match data gen <-> reco (if gen is available) ---
    if data_gen_events is not None:
        data_gen_mask, data_reco_mask = match_events(data_gen_events, data_reco_events)
        data_gen_events  = data_gen_events[data_gen_mask]
        data_reco_events = data_reco_events[data_reco_mask]

        if verbose:
            print(f"\nMatched data events: {len(data_gen_events)}")

    return prior_gen_events, prior_reco_events, data_gen_events, data_reco_events