# OmniFold

This folder contains the OmniFold implementation used to unfold HWW jet events at
particle level.

The core algorithm (`omnifold.py`) is adapted from
[hep-lbdl/OmniFold](https://github.com/hep-lbdl/OmniFold), the implementation
accompanying the paper:

> Andreassen et al., *OmniFold: A Method to Simultaneously Unfold All Observables*,
> Phys. Rev. Lett. 124, 182001 (2020)
> https://doi.org/10.1103/PhysRevLett.124.182001

## Contents

- `omnifold.py` — core OmniFold algorithm (adapted from hep-lbdl/OmniFold)
- `wandb_helpers.py` — W&B initialisation, callbacks, and weight logging

## Modifications to omnifold.py

- Added a `callbacks_fn` argument to pass per-iteration W&B callbacks from `wandb_helpers.py`
- Added a `weighted_accuracy` metric to correctly handle the weight-hiding trick in `y_true`
  (see [this issue](https://github.com/hep-lbdl/OmniFold/issues) for details)
