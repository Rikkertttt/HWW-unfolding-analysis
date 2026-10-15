# This code is used to convert ROOT files to Parquet files, while applying event selection filters.
# This code is meant to be used in parallel, if access to HTCondor is not available, it is recommended to use the sequential version of this code: root_to_parquet_seq.py

import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis/data_prep")

import argparse
import helpers.read_data as rd
import event_selection_filters as filter
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--file",    required=True, help="Full path to ROOT file")
parser.add_argument("--savedir", required=True, help="Base save directory for parquet output")
args = parser.parse_args()

total_reco_counts = defaultdict(int)
total_gen_counts = defaultdict(int)

BASE_DIR = args.savedir.rstrip("/") + "/"

save_DIR_raw  = BASE_DIR + "raw/"
save_DIR_pre  = BASE_DIR + "preselection/"
save_DIR_full = BASE_DIR + "full_selection/"

os.makedirs(save_DIR_raw + "reco/counts", exist_ok=True)
os.makedirs(save_DIR_raw + "gen/counts",  exist_ok=True)
os.makedirs(save_DIR_pre + "reco/counts", exist_ok=True)
os.makedirs(save_DIR_pre + "gen/counts",  exist_ok=True)
os.makedirs(save_DIR_full + "reco/counts", exist_ok=True)
os.makedirs(save_DIR_full + "gen/counts",  exist_ok=True)


file = args.file
stem = os.path.splitext(os.path.basename(file))[0]

PATH = file + ":Delphes"
tree = rd.open_root_file(PATH, show_keys=False, verbose=False)

reco_event = rd.load_reco_objects(tree, verbose=False)
gen_event  = rd.load_gen_objects(tree, verbose=False, MET_is_neutrino=True)

# Save raw data
rd.save_events(reco_event, f"{save_DIR_raw}reco/reco_{stem}_raw.parquet")
rd.save_events(gen_event,  f"{save_DIR_raw}gen/gen_{stem}_raw.parquet")

# Preselection reco + save
reco_event, counts = filter.full_preselection(reco_event)
rd.save_events(reco_event, f"{save_DIR_pre}reco/reco_{stem}_pre.parquet")
# Selection reco + save
reco_event, counts = filter.full_selection(reco_event, counts=counts)
rd.save_events(reco_event, f"{save_DIR_full}reco/reco_{stem}_full.parquet")

# Get number of events after each cut
for label, n in counts.items():
    total_reco_counts[label] += n

# Filter only high PT generator leptons
gen_event = rd.EventObjects(
    event_id  = gen_event.event_id,
    jets      = gen_event.jets,
    electrons = gen_event.electrons[gen_event.electrons.pt > 15],
    muons     = gen_event.muons[gen_event.muons.pt > 15],
    met       = gen_event.met,
)

# Preselection gen + save
gen_event, counts = filter.full_preselection(gen_event)
rd.save_events(gen_event, f"{save_DIR_pre}gen/gen_{stem}_pre.parquet")
# Selection gen + save
gen_event, counts = filter.full_selection(gen_event, counts=counts)
rd.save_events(gen_event,  f"{save_DIR_full}gen/gen_{stem}_full.parquet")

# Get number of events after each cut
for label, n in counts.items():
    total_gen_counts[label] += n


def write_counts(counts: dict, path: str) -> None:
    with open(path, "w") as f:
        for label, n in counts.items():
            f.write(f"{label}:\t{n}\n")

write_counts(total_reco_counts, f"{save_DIR_full}reco/counts/counts_{stem}.txt")
write_counts(total_gen_counts,  f"{save_DIR_full}gen/counts/counts_{stem}.txt")

print("\nReco -----------------------")
for label, n in total_reco_counts.items():
    print(f"N after {label}:\t\t{n}")

print("\nGen -----------------------")
for label, n in total_gen_counts.items():
    print(f"N after {label}:\t\t{n}")