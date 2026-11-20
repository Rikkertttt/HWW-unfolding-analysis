from helpers.read_data import load_multiple_events, match_events, EventObjects
from typing import List, Optional, Union


def _load_process_events(
    reco_path: str,
    gen_path: Optional[str] = None,
    max_reco_files: Optional[int] = None,
    max_gen_files: Optional[int] = None,
    verbose: bool = False,
    label: str = "",
) -> tuple[Optional[EventObjects], EventObjects]:
    """Load reco (and optional gen) for a single process directory and match them."""
    if verbose:
        print(f"Loading {label} reco: {reco_path}")
    reco = load_multiple_events(DIR=reco_path, max_files=max_reco_files)

    gen = None
    if gen_path is not None:
        if verbose:
            print(f"Loading {label} gen:  {gen_path}")
        gen = load_multiple_events(DIR=gen_path, max_files=max_gen_files)

        # Match gen <-> reco within this process
        gen_mask, reco_mask = match_events(gen, reco)
        gen = gen[gen_mask]
        reco = reco[reco_mask]
        if verbose:
            print(f"  Matched {label} events: {len(reco)}")

    return gen, reco


def get_all_data_omnifold(
    prior_reco_events_path: Union[str, List[str]],
    prior_gen_events_path:  Union[str, List[str]],
    data_reco_events_path:  Optional[Union[str, List[str]]] = None,
    data_gen_events_path:   Optional[Union[str, List[str]]] = None,
    max_files:              Optional[List[Optional[int]]] = None,
    closure_test:           bool = False,
    verbose:                bool = False,
) -> tuple[EventObjects, EventObjects, Optional[EventObjects], EventObjects]:
    """
    Load and match prior (sim) and data events for OmniFold.

    Prior events are matched gen<->reco by (file_id, event_id).

    Supports either a single directory path (str) or a list of directory paths (List[str])
    for combining multiple processes (e.g. HWW + ttbar).

    If closure_test=True, the matched prior reco events are split in half:
    one half is used as the prior, the other as pseudo-data. No external
    data paths are needed in this mode.

    Otherwise, data_reco_events_path must be provided. data_gen_events_path
    is optional and only used for stress tests where truth-level data is available.
    """

    if max_files is None:
        max_files = [None, None, None, None]

    # Normalize prior paths to lists
    prior_reco_paths = [prior_reco_events_path] if isinstance(prior_reco_events_path, str) else prior_reco_events_path
    prior_gen_paths  = [prior_gen_events_path]  if isinstance(prior_gen_events_path, str)  else prior_gen_events_path

    if len(prior_reco_paths) != len(prior_gen_paths):
        raise ValueError("Number of prior reco folders must match number of prior gen folders")

    # --- Load & match prior (loop over folders and combine with +) ---
    prior_gen_events: Optional[EventObjects] = None
    prior_reco_events: Optional[EventObjects] = None

    for i, (r_dir, g_dir) in enumerate(zip(prior_reco_paths, prior_gen_paths)):
        g, r = _load_process_events(
            reco_path=r_dir,
            gen_path=g_dir,
            max_reco_files=max_files[1],
            max_gen_files=max_files[0],
            verbose=verbose,
            label=f"Prior [{i+1}/{len(prior_reco_paths)}]",
        )
        if g is None:
            raise ValueError(f"Failed to load prior gen events from {g_dir}")

        prior_gen_events  = g if prior_gen_events is None else (prior_gen_events + g)
        prior_reco_events = r if prior_reco_events is None else (prior_reco_events + r)

    if prior_gen_events is None or prior_reco_events is None:
        raise ValueError("No prior events found. Please check prior_gen_events_path and prior_reco_events_path.")

    if verbose and len(prior_reco_paths) > 1:
        print(f"\nTotal combined prior events: {len(prior_reco_events)}")

    # --- Closure test: split matched prior reco in half ---
    if closure_test:
        if verbose:
            print("\nClosure test mode: splitting prior reco into prior / pseudo-data")
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

    data_reco_paths = [data_reco_events_path] if isinstance(data_reco_events_path, str) else data_reco_events_path
    data_gen_paths  = None
    if data_gen_events_path is not None:
        data_gen_paths = [data_gen_events_path] if isinstance(data_gen_events_path, str) else data_gen_events_path
        if len(data_reco_paths) != len(data_gen_paths):
            raise ValueError("Number of data reco folders must match number of data gen folders")

    data_reco_events: Optional[EventObjects] = None
    data_gen_events: Optional[EventObjects] = None

    for i, r_dir in enumerate(data_reco_paths):
        g_dir = data_gen_paths[i] if data_gen_paths is not None else None
        g, r = _load_process_events(
            reco_path=r_dir,
            gen_path=g_dir,
            max_reco_files=max_files[2],
            max_gen_files=max_files[3],
            verbose=verbose,
            label=f"Data [{i+1}/{len(data_reco_paths)}]",
        )
        data_reco_events = r if data_reco_events is None else (data_reco_events + r)
        if g is not None:
            data_gen_events = g if data_gen_events is None else (data_gen_events + g)

    if data_reco_events is None:
        raise ValueError("No data reco events found. Please check data_reco_events_path.")

    if verbose and len(data_reco_paths) > 1:
        print(f"\nTotal combined data events: {len(data_reco_events)}")

    return prior_gen_events, prior_reco_events, data_gen_events, data_reco_events
