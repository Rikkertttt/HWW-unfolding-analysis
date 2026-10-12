# data_prep

This folder contains the pipeline for converting Delphes ROOT files into Parquet files with event selection applied. It comes in two flavours — **sequential** (one process, all files in a loop) and **parallel** (one HTCondor job per file). Both produce identical output; the parallel version is strongly preferred for large datasets.

The event-selection logic itself lives in `event_selection_filters.py`, which is shared by both.

---

## Folder structure

```
data_prep/
├── event_selection_filters.py   # All cut functions + full_preselection / full_selection
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

For generator-level events, a pt > 15 GeV cut is applied to electrons and muons before the selection chain, to match the particle-level fiducial definition.

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
    │   └── counts/counts_<stem>.txt   (parallel) / counts.txt (sequential)
    └── gen/
        ├── gen_<stem>_full.parquet
        └── counts/counts_<stem>.txt   (parallel) / counts.txt (sequential)
```

---

## Event selection

Cuts are implemented in `event_selection_filters.py` and follow the VBF H→WW* eμ selection from [Phys. Rev. D 108, 072003](http://dx.doi.org/10.1103/PhysRevD.108.072003).

### Preselection (`full_preselection`)

| Step | Cut |
|---|---|
| Jet cleaning | pt > 20 GeV, \|η\| < 4.5 |
| Lepton multiplicity | ≥ 1 muon and ≥ 1 electron |
| Opposite charge | leading μ and e must have opposite charge |
| Lepton pt | leading lepton ≥ 22 GeV, subleading ≥ 15 GeV |
| Lepton η | \|η\| < 2.5 for both |
| Lepton ΔR | leading μ–e separation > 0.1 |
| Lepton pair mass | m(eμ) > 10 GeV |
| Jet–lepton overlap | remove jets within ΔR < 0.4 of either leading lepton |
| Jet multiplicity | ≥ 2 remaining jets |
| b-tag veto | reject events with a b-tagged jet (pt > 20 GeV, \|η\| < 2.5) |

### Full selection (`full_selection`)

Applied on top of preselection:

| Step | Cut |
|---|---|
| Central jet veto (CJV) | no additional jet with pt > 20 GeV between the two leading jets in rapidity |
| Outside lepton veto (OLV) | both selected leptons must lie within the rapidity interval of the two leading jets |
| Dijet mass | m(j1, j2) > 450 GeV |
| Dijet rapidity gap | \|Δy(j1, j2)\| ≥ 2.1 |
| Lepton Δφ | Δφ(e, μ) ≤ 1.4 rad |

---

## Sequential mode

Processes all ROOT files one by one in a single Python process. Use this for quick tests on a small dataset or when HTCondor is not available.

### How to run locally

```bash
cd /user/rvanrhee/projects_rik/HWW-unfolding-analysis/data_prep/sequential
source /user/rvanrhee/projects_rik/.venv/bin/activate
python3 root_to_parquet_seq.py
```

Edit the `DIR` and `save_DIR_*` paths at the top of `root_to_parquet_seq.py` before running.

### How to submit to HTCondor (single job)

```bash
# On a Stoomboot login node:
condor_submit root_to_parquet_seq.sub
```

The submit file requests **4 CPUs and 32 GB RAM** for the single job.  
Logs are written to `data_prep/logs/`.

---

## Parallel mode

Processes **one ROOT file per HTCondor job**. This is the recommended approach for large datasets. After all jobs finish, run `aggregate_counts.py` to sum the per-file cut counts.

### Workflow

```
1.  Edit root_to_parquet_parallel.sub
       - Set the glob pattern in `queue file matching files ...` to your ROOT file directory.
       - Set the `arguments` line: <ROOT file> <base output directory>

2.  Submit all jobs:
        condor_submit root_to_parquet_parallel.sub

3.  Monitor:
        condor_q

4.  After all jobs complete, aggregate cut counts:
        python3 aggregate_counts.py
```

### What each file does

| File | Role |
|---|---|
| `root_to_parquet_parallel.py` | Processes a single ROOT file. Takes `--file` and `--savedir` as CLI arguments. |
| `run_root_to_parquet_parallel.sh` | Wrapper script: activates the venv and calls the Python script. This is what HTCondor executes. |
| `root_to_parquet_parallel.sub` | HTCondor submit file. Uses `queue file matching files` to spawn one job per ROOT file. |
| `aggregate_counts.py` | Reads all per-file `counts_<stem>.txt` files and sums them into `counts_total.txt`. |

### HTCondor resource request (per job)

| Resource | Value |
|---|---|
| CPUs | 2 |
| Memory | 16 GB |
| OS | el9 |
| Job category | medium |

### Configuring paths

Before submitting, check the following hardcoded paths and update them if needed:

- **`root_to_parquet_parallel.sub`** — `arguments` line: change the output base directory; `queue file matching files` line: change the input ROOT file glob.
- **`aggregate_counts.py`** — `save_DIR_full`: change to the `full_selection/` directory of your output.
- **`run_root_to_parquet_parallel.sh`** — venv path: `source /user/rvanrhee/projects_rik/.venv/bin/activate`.

---

## Dependencies

- [`uproot`](https://uproot.readthedocs.io/) + [`awkward`](https://awkward-array.org/) + [`vector`](https://vector.readthedocs.io/) — ROOT I/O and four-momentum objects
- `numpy`
- `tqdm` (sequential only)
- `helpers.read_data` — internal module for reading Delphes trees and saving/loading `EventObjects`

All dependencies are expected to be installed in the project virtualenv at  
`/user/rvanrhee/projects_rik/.venv`.
