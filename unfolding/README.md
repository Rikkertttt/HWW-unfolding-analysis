# unfolding/

Scripts for running and plotting the OmniFold unfolding on HWW jet events.

---

## Scripts

### `run_unfolding.py`

Loads matched prior and data events, trains the OmniFold classifier, and saves the results.

**Inputs** (hardcoded paths — update before running):
- `prior_gen_DIR` / `prior_reco_DIR` — directories containing matched gen/reco parquet files
- `OUTDIR` — output directory for weights and matched parquets

**What it does:**
1. Loads and matches events via `get_all_data_omnifold` (`closure_test=True` splits the prior in two halves)
2. Extracts features using `make_cfg` / `make_features` and scales them with `make_theta_and_scale`
3. Initialises a W&B run under project `hww-omnifold`
4. Runs `omnifold()` with early stopping and W&B logging callbacks
5. Logs per-iteration weight statistics to W&B

**Outputs** (written to `OUTDIR`):
- `weights.npy` — shape `(iterations, 2, N)`; final particle-level weights at `weights[-1, 1, :]`
- `prior_gen_matched.parquet`, `prior_reco_matched.parquet`
- `data_gen_matched.parquet`, `data_reco_matched.parquet`
- `features.json` — ordered list of feature names used in training

---

### `unfolding_plotting.py`

Loads the saved outputs from `run_unfolding.py` and produces diagnostic plots.

**Inputs** (hardcoded `PATH` — update before running):
- The output directory written by `run_unfolding.py`

**What it does:**
1. Loads matched parquets and weights (with optional percentile clipping)
2. Builds feature rows for all training features plus `higgs_mass` (not a training feature)
3. Produces three types of plots:
   - **Weight distributions** — Step 1 and Step 2 weight histograms
   - **Overview** — detector-level and particle-level distributions per feature
   - **Closure summary** — particle-level unfolded vs truth with ratio panel, both as a combined grid and as individual per-feature plots in `individual_plots/`

> **Note:** both scripts contain hardcoded paths for the Nikhef Stoomboot environment.
> Update `sys.path.append`, the input directories, and `OUTDIR` / `PATH` before running elsewhere.
