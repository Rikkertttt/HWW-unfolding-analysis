#!/bin/bash
set -euo pipefail

cd /user/rvanrhee/projects_rik/HWW-unfolding-analysis/data_prep

source /user/rvanrhee/projects_rik/.venv/bin/activate

echo "Host: $(hostname -f)"
echo "Python: $(which python3)"
python3 --version

exec python3 root_to_parquet.py
