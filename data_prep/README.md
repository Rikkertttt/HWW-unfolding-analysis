# data_prep

This folder contains the pipeline for converting Delphes ROOT files into Parquet files with event selection applied. It comes in two flavours — **sequential** (one process, all files in a loop) and **parallel** (one HTCondor job per file). The parallel version is strongly preferred for large datasets.

The event-selection logic itself lives in `event_selection_filters.py`, which is shared by both.

---

## Folder structure

```
data_prep/
├── README.md
├── event_selection_filters.py   # All cut functions + full_preselection / full_selection
├── logs/                        # HTCondor log/err/out directory (must exist before submitting)
├── sequential/
│   ├── root_to_parquet_seq.py   # Main script — loops over all ROOT files
│   ├── root_to_parquet_seq.sub  # HTCondor submit file (single job)
│   └── run_root_to_parquet_seq.sh
└── parallel/
    ├── root_to_parquet_parallel.py   # Main script — processes one ROOT file (takes CLI args)
    ├── root_to_parquet_parallel.sub  # HTCondor submit file (one job per ROOT file)
    ├── run_root_to_parquet_parallel.sh
    └── aggregate_counts.py           # Post-processing: sum per-file cut counts into totals
```

> [!IMPORTANT]
> The `logs/` directory is ignored by git (`.gitignore`). Ensure `data_prep/logs/` exists (`mkdir -p data_prep/logs`) before submitting HTCondor jobs, otherwise jobs will fail to write logs.

---

## What the pipeline does

Each ROOT file goes through the following steps for both **reco** (reconstructed) and **gen** (generator-level) events:

1. **Read** the Delphes `TTree` with `helpers.read_data`.
2. **Save raw** events to Parquet (no selection applied).
3. **Apply preselection** (`full_preselection`) and save to Parquet.
4. **Apply full selection** (`full_selection`) on top of preselection and save to Parquet.
5. **Write per-file cut counts** to a text file.

### Generator-level processing
- **Leptons:** A $p_\text{T} > 15\text{ GeV}$ cut is applied to electrons and muons before the selection chain to match the particle-level fiducial definition.
- **Missing ET (MET):** 
  - By default, generator MET is taken from the Delphes `GenMissingET` branch.
  - When `--met-is-neutrino` is passed (parallel mode), generator MET is instead calculated directly as the vector sum of prompt status-1 neutrinos ($|\text{PID}| \in \{12, 14\}$).

### Output directory layout

```
<base_dir>/
├── raw/
│   ├── reco/   reco_<stem>_raw.parquet
│   └── gen/    gen_<stem>_raw.parquet
├── preselection/
│   ├── reco/   reco_<stem>_pre.parquet
│   └── gen/    gen_<stem>_pre.parquet
└── full_selection/
    ├── reco/
    │   ├── reco_<stem>_full.parquet
    │   └── counts/
    │       ├── counts_<stem>.txt       (parallel per-file counts)
    │       ├── counts_total.txt        (parallel aggregated counts)
    │       └── counts.txt              (sequential counts)
    └── gen/
        ├── gen_<stem>_full.parquet
        └── counts/
            ├── counts_<stem>.txt       (parallel per-file counts)
            ├── counts_total.txt        (parallel aggregated counts)
            └── counts.txt              (sequential counts)
```

---

## Event selection

Cuts are implemented in `event_selection_filters.py` and follow the VBF $H\rightarrow WW^* \rightarrow e\nu_e \mu\nu_\mu$ selection from [Phys. Rev. D 108, 072003](http://dx.doi.org/10.1103/PhysRevD.108.072003).

### Preselection (`full_preselection`)

| Step | Cut | Count label |
|---|---|---|
| Initial | Initial event count | `Start` |
| Jet cleaning | $p_\text{T} > 20\text{ GeV}$, $|\eta| < 4.5$ | *(cleans jet collection; no event cut)* |
| Lepton multiplicity | $\ge 1\ \mu$ | `1_muon` |
| Lepton multiplicity | $\ge 1\text{ e}$ | `1_electron` |
| Opposite charge | Leading $\mu$ and $\text{e}$ must have opposite charge | `opp_charge` |
| Lepton $p_\text{T}$ | Leading lepton $\ge 22\text{ GeV}$, subleading $\ge 15\text{ GeV}$ | `lepton_pt` |
| Lepton $\eta$ | $|\eta| < 2.5$ for both leading leptons | `lepton_eta` |
| Lepton $\Delta R$ | Leading $\mu\text{--e}$ separation $\Delta R > 0.1$ | `lepton_dR` |
| Lepton pair mass | $m(\text{e}\mu) > 10\text{ GeV}$ | `lepton_mass` |
| Jet–lepton overlap | Remove jets within $\Delta R < 0.4$ of either leading lepton | `lep_jet_dR_temp` *(cleans jets; event count unchanged)* |
| Jet multiplicity | $\ge 2$ remaining jets | `2_jets` |
| b-tag veto | Reject events with a b-tagged jet ($p_\text{T} > 20\text{ GeV}$, $|\eta| < 2.5$) | `btag` |

### Full selection (`full_selection`)

Applied on top of preselection:

| Step | Cut | Count label |
|---|---|---|
| Central jet veto (CJV) | No additional jet with $p_\text{T} > 20\text{ GeV}$ between the two leading jets in rapidity | `CJV` |
| Outside lepton veto (OLV) | Both selected leptons must lie within the rapidity interval of the two leading jets | `OLV` |
| Dijet mass | $m(j_1, j_2) > 450\text{ GeV}$ | `jets_mass` |
| Dijet rapidity gap | $|\Delta y(j_1, j_2)| \ge 2.1$ | `jet_rapidity` |
| Lepton $\Delta\phi$ | $\Delta\phi(\text{e}, \mu) \le 1.4\text{ rad}$ | `lepton_phi` |

---

## Sequential mode

Processes all ROOT files one by one in a single Python process. Use this for quick tests on a small dataset or when HTCondor is not available.

### How to run locally

Before running, edit the `DIR` and `save_DIR_*` paths at the top of `root_to_parquet_seq.py` to point to your input ROOT files and desired output location.

```bash
cd data_prep/sequential
source /path/to/your/venv/bin/activate
python3 root_to_parquet_seq.py
```

### How to submit to HTCondor (single job)

Ensure `data_prep/logs/` exists, and update the `executable` path in `root_to_parquet_seq.sub` to point to `run_root_to_parquet_seq.sh` in your checkout:

```bash
cd data_prep/sequential
condor_submit root_to_parquet_seq.sub
```

#### HTCondor resource request (sequential job)

| Resource | Value |
|---|---|
| CPUs | 4 |
| Memory | 32 GB |
| OS | el9 |
| Job category | medium |

---

## Parallel mode

Processes **one ROOT file per HTCondor job**. This is the recommended approach for large datasets. After all jobs finish, run `aggregate_counts.py` to sum the per-file cut counts.

### Workflow

1. **Ensure logs directory exists:**
   ```bash
   mkdir -p data_prep/logs
   ```

2. **Edit `root_to_parquet_parallel.sub`:**
   - Set the glob pattern in `queue file matching files ...` to your ROOT file directory.
   - Set the `arguments` line:
     ```condor
     arguments = $(file) <base_output_directory> [--met-is-neutrino]
     ```
     > [!NOTE]
     > Keep the literal HTCondor macro `$(file)` as the first argument — HTCondor substitutes each matched ROOT file into this variable automatically.
     > Add the optional `--met-is-neutrino` flag if you want generator MET computed from prompt neutrinos instead of the `GenMissingET` branch.
   - Update `executable`, `error`, and `log` paths to point to your checkout if needed.

3. **Submit all jobs:**
   ```bash
   cd data_prep/parallel
   condor_submit root_to_parquet_parallel.sub
   ```

4. **Monitor jobs:**
   ```bash
   condor_q
   ```

5. **Aggregate cut counts:**
   After all jobs complete, edit `save_DIR_full` in `aggregate_counts.py` to point to your `full_selection/` directory, then run:
   ```bash
   cd data_prep/parallel
   python3 aggregate_counts.py
   ```
   This will print the summary tables and write `counts_total.txt` into both `reco/counts/` and `gen/counts/`.

### What each file does

| File | Role |
|---|---|
| `root_to_parquet_parallel.py` | Processes a single ROOT file. CLI args: `--file <path>` (required), `--savedir <path>` (required), and `--met-is-neutrino` (optional flag). |
| `run_root_to_parquet_parallel.sh` | Wrapper script: activates the virtual environment and calls `root_to_parquet_parallel.py`, forwarding all arguments (`${@:3}`). |
| `root_to_parquet_parallel.sub` | HTCondor submit file. Spawns one job per ROOT file via `queue file matching files`. |
| `aggregate_counts.py` | Reads all per-file `counts_<stem>.txt` files from `full_selection/{reco,gen}/counts/` and sums them into `counts_total.txt`. |

### HTCondor resource request (per parallel job)

| Resource | Value |
|---|---|
| CPUs | 2 |
| Memory | 16 GB |
| OS | el9 |
| Job category | express |

Adjust these values in `root_to_parquet_parallel.sub` as needed for your cluster.

### Configuring paths

The following locations are hardcoded and must be updated before use:

- **`root_to_parquet_parallel.sub`** — `executable`: path to `run_root_to_parquet_parallel.sh`; `arguments`: `$(file) <base_output_directory> [--met-is-neutrino]`; `error` and `log`: path to `data_prep/logs/`; `queue file matching files`: glob pattern for your input ROOT files.
- **`root_to_parquet_seq.sub`** — `executable`: path to `run_root_to_parquet_seq.sh`; `log`, `output`, and `error`: path to `data_prep/logs/`.
- **`aggregate_counts.py`** — `save_DIR_full`: path to the `full_selection/` output directory.
- **`run_root_to_parquet_parallel.sh`** and **`run_root_to_parquet_seq.sh`** — update the `source` line to activate your Python virtual environment.

---

## Dependencies

- [`uproot`](https://uproot.readthedocs.io/) + [`awkward`](https://awkward-array.org/) + [`vector`](https://vector.readthedocs.io/) — ROOT I/O and four-momentum objects
- [`pyarrow`](https://arrow.apache.org/docs/python/) (or `fastparquet`) — Parquet I/O engine required by `ak.to_parquet`
- `numpy`
- `tqdm` (sequential only)
- `helpers.read_data` — internal module for reading Delphes trees and saving/loading `EventObjects`

All dependencies should be installed in a Python virtual environment. Activate it before running any script, and update the `source` line in the shell scripts to point to your venv.
