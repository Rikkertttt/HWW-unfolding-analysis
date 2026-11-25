# HWW-unfolding-analysis

Unbinned unfolding of VBF H→WW\*→eνμν jet events at particle level using [OmniFold](https://arxiv.org/abs/1911.09107) on Delphes-simulated data.

The core neural-network algorithm (`Omnifold/omnifold.py`) is adapted from [hep-lbdl/OmniFold](https://github.com/hep-lbdl/OmniFold), the implementation accompanying:

> Andreassen et al., *OmniFold: A Method to Simultaneously Unfold All Observables*, Phys. Rev. Lett. 124, 182001 (2020)
> [https://doi.org/10.1103/PhysRevLett.124.182001](https://doi.org/10.1103/PhysRevLett.124.182001)

In addition, `Omnifold/omnifold_ydf.py` provides a Yggdrasil Decision Forest (YDF) gradient-boosted trees implementation (`omnifold_ydf`) as an alternative to the Keras classifier.

***

## Structure

```
helpers/
    read_data.py          EventObjects class; load/save/match events from parquet and ROOT
    data_loading.py       Load and match prior and data events (get_all_data_omnifold),
                          supporting single/multiple processes and closure tests

Omnifold/
    omnifold.py           Core OmniFold algorithm (adapted from hep-lbdl/OmniFold)
    omnifold_ydf.py       YDF Gradient Boosted Trees implementation of OmniFold
                          (omnifold_ydf, reweight_ydf, to_ydf_ds)
    models.py             Keras classifier: build_omnifold_classifier(n_features)
    features.py           Feature extraction (make_cfg, make_features, make_theta_and_scale,
                          higgs_mass), FEATURE_META lookup table, and build_feature_rows
    plotting.py           Weight loading (open_saved_weights), weight distribution plots,
                          per-feature overview plots, and closure summary plots with ratio panels
    wandb_helpers.py      W&B run initialisation (init_omnifold_run), per-epoch callback
                          (WandbOmniFoldCallback), and per-iteration weight logging (log_weights)

unfolding/
    run_unfolding.py      Load data, run Keras OmniFold, save weights + matched parquet files
    run_ydf_unfolding.py  Load data, run YDF OmniFold, save weights + matched parquet files + reports
    unfolding_plotting.py Load saved outputs, compute observables, produce plots
```

## Modifications to `omnifold.py`

* Added a `callbacks_fn` argument to pass per-iteration Keras callbacks from outside the core algorithm. It must be a callable `callbacks_fn(i)` that returns `[callbacks_step1, callbacks_step2]` — a list of two callback lists, one per step. If `None`, both steps run with no callbacks.
* Added a `weighted_accuracy` metric to correctly handle the weight-hiding trick in `y_true`, where event weights are stacked alongside labels before being passed to the loss.

## YDF Implementation (`omnifold_ydf.py`)

* Replaces the Keras classifier with `ydf.GradientBoostedTreesLearner` (`discretize_numerical_columns=True`, configurable `num_trees=300` and `max_depth=6`), passing event weights natively via `weights="weight"` instead of stacking them into `y_true`.
* `to_ydf_ds` packs feature arrays, string labels, and weights into the dictionary format expected by YDF, and `reweight_ydf` computes `probs / (1. - probs)` (with `np.nan_to_num`).
* When `save_report_path` is set, saves per-iteration, per-step HTML model summaries (`model_iter{i}_step{1,2}.html`) and train/validation loss and accuracy curves (`loss_iter{i}_step{1,2}.png`) under `<save_report_path>/ydf_reports/`.

***

## Workflow

**1. Run unfolding** (`unfolding/run_unfolding.py` or `unfolding/run_ydf_unfolding.py`)

Loads and matches prior and data events via `get_all_data_omnifold`, trains the OmniFold classifier (Keras MLP or YDF GBDT), and saves:

* `weights.npy` — shape `(iterations, 2, N)`, final reco weights at `weights[-1, 0, :]` and final particle-level weights at `weights[-1, 1, :]`
* `prior_gen_matched.parquet`, `prior_reco_matched.parquet`, `data_gen_matched.parquet`, `data_reco_matched.parquet` — matched events
* `features.json` — list of features used in training
* `ydf_reports/` — model description HTMLs and loss/accuracy plots (when running `omnifold_ydf` with `save_report_path`)

**2. Plot results** (`unfolding/unfolding_plotting.py`)

Loads the saved parquets and weights, computes any observable directly from `EventObjects`, and applies weights at histogram time. Observables do not need to be training features.
