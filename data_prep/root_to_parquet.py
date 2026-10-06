import os
import sys
sys.path.append("/user/rvanrhee/projects_rik/HWW-unfolding-analysis")

import time
import helpers.read_data as rd
import event_selection_filters as filter
from collections import defaultdict
from tqdm import tqdm

total_reco_counts = defaultdict(int)
total_gen_counts = defaultdict(int)

DIR = "/dcache/atlas/llehmann/Eventgeneration/standalone/hww/"
files = sorted([f for f in os.listdir(DIR) if f.endswith(".root")])

save_DIR_raw = "/data/atlas/users/rvanrhee/hww_parquet/raw_new/"
save_DIR_pre = "/data/atlas/users/rvanrhee/hww_parquet/preselection_new/"
save_DIR_full = "/data/atlas/users/rvanrhee/hww_parquet/full_selection_new/"
os.makedirs(save_DIR_raw + "reco/", exist_ok=True)
os.makedirs(save_DIR_raw + "gen/",  exist_ok=True)
os.makedirs(save_DIR_pre + "reco/", exist_ok=True)
os.makedirs(save_DIR_pre + "gen/",  exist_ok=True)
os.makedirs(save_DIR_full + "reco/", exist_ok=True)
os.makedirs(save_DIR_full + "gen/",  exist_ok=True)

for file in tqdm(files, desc="Processing files", unit="file"):
    t0 = time.time()
    stem = os.path.splitext(file)[0]

    PATH = DIR + file + ":Delphes"
    tree = rd.open_root_file(PATH, show_keys=False, verbose=False)

    reco_event = rd.load_reco_objects(tree, verbose=False)
    gen_event  = rd.load_gen_objects(tree, verbose=False)

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

    tqdm.write(f"  {file} done in {time.time() - t0:.1f}s")

def write_counts(counts: dict, path: str) -> None:
    with open(path, "w") as f:
        for label, n in counts.items():
            f.write(f"{label}:\t{n}\n")

write_counts(total_reco_counts, f"{save_DIR_full}reco/counts.txt")
write_counts(total_gen_counts,  f"{save_DIR_full}gen/counts.txt")

print("\nReco -----------------------")
for label, n in total_reco_counts.items():
    print(f"N after {label}:\t\t{n}")

print("\nGen -----------------------")
for label, n in total_gen_counts.items():
    print(f"N after {label}:\t\t{n}")