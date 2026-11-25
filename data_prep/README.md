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

| Step | Cut |
|---|---|
| Jet cleaning | pT > 20 GeV, \|η\| < 4.5 |
| Lepton multiplicity | ≥ 1 μ and ≥ 1 e |
| Opposite charge | leading μ and e must have opposite charge |
| Lepton pT | leading lepton ≥ 22 GeV, subleading ≥ 15 GeV |
| Lepton η | \|η\| < 2.5 for both |
| Lepton ΔR | leading μ–e separation ΔR > 0.1 |
| Lepton pair mass | m(eμ) > 10 GeV |
| Jet–lepton overlap | remove jets within ΔR < 0.4 of either leading lepton |
| Jet multiplicity | ≥ 2 remaining jets |
| b-tag veto | reject events with a b-tagged jet with pT > 20 GeV, \|η\| < 2.5 |

### Full selection (`full_selection`)

Applied on top of preselection:

| Step | Cut |
|---|---|
| Central jet veto (CJV) | no additional jet with pT > 20 GeV between the two leading jets in rapidity |
| Outside lepton veto (OLV) | both selected leptons must lie within the rapidity interval of the two leading jets |
| Dijet mass | m(j1, j2) > 450 GeV |
| Dijet rapidity gap | \|Δy(j1, j2)\| ≥ 2.1 |
| Lepton Δφ | Δφ(e, μ) ≤ 1.4 rad |

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
