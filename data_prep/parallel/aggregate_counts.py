# This script aggregates the counts from all the parallel jobs and writes out a total count file for each of reco and gen.
# Run this after all the parallel jobs have finished to get the total counts for each cut.

import os
from collections import defaultdict

save_DIR_full = "/data/atlas/users/rvanrhee/eft_chwtil5p0_parquet/full_selection/"

for label, subdir in [("Reco", "reco/counts/"), ("Gen", "gen/counts/")]:
    totals = defaultdict(int)
    count_files = [f for f in os.listdir(save_DIR_full + subdir) if f.startswith("counts_")]
    for fname in count_files:
        with open(save_DIR_full + subdir + "/" + fname) as f:
            for line in f:
                key, n = line.strip().split(":\t")
                totals[key] += int(n)
    print(f"\n{label} -----------------------")
    for key, n in totals.items():
        print(f"N after {key}:\t\t{n}")
    with open(save_DIR_full + subdir + "/counts_total.txt", "w") as f:
        for key, n in totals.items():
            f.write(f"{key}:\t{n}\n")