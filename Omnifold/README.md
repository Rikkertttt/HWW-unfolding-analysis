# HWW-unfolding-analysis

Unbinned unfolding of VBF H→WW*→eνμν jet events at particle level using [OmniFold](https://arxiv.org/abs/1911.09107) on Delphes-simulated data.

The core algorithm (`Omnifold/omnifold.py`) is adapted from [hep-lbdl/OmniFold](https://github.com/hep-lbdl/OmniFold), the implementation accompanying:

> Andreassen et al., *OmniFold: A Method to Simultaneously Unfold All Observables*, Phys. Rev. Lett. 124, 182001 (2020)
> https://doi.org/10.1103/PhysRevLett.124.182001

---

## Structure

```
helpers/
    read_data.py          EventObjects class; load/save/match events from parquet and ROOT

Omnifold/
    omnifold.py           Core OmniFold algorithm (adapted from hep-lbdl/OmniFold)
    models.py             Keras classifiers used in both OmniFold steps
    features.py           Feature extraction (make_cfg, make_features, make_theta_and_scale), and higgs_mass
    plotting.py           Histogram plots (overview + closure) and weight loading
    wandb_helpers.py      W&B initialisation, callbacks, and weight logging

unfolding/
    run_unfolding.py      Load data, run OmniFold, save weights + matched parquet files
    unfolding_plotting.py Load saved outputs, compute observables, produce plots
```

## Modifications to `omnifold.py`

- Added a `callbacks_fn` argument to pass per-iteration Keras callbacks (e.g. W&B logging, early stopping) from outside the core algorithm
- Added a `weighted_accuracy` metric to correctly handle the weight-hiding trick in `y_true`, where event weights are stacked alongside labels before being passed to the loss

---

## Workflow

**1. Run unfolding** (`unfolding/run_unfolding.py`)

Loads and matches prior (HWW) and data events, trains the OmniFold classifier, and saves:
- `weights.npy` — shape `(iterations, 2, N)`, final particle-level weights at `weights[-1, 1, :]`
- `prior_gen/reco_matched.parquet`, `data_gen/reco_matched.parquet` — matched events
- `features.json` — list of features used in training

**2. Plot results** (`unfolding/unfolding_plotting.py`)

Loads the saved parquets and weights, computes any observable directly from `EventObjects`, and applies weights at histogram time. Observables do not need to be training features.
