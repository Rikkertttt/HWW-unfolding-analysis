#!/bin/bash
set -euo pipefail

cd /user/rvanrhee/projects_rik/HWW-unfolding-analysis/data_prep/parallel

source /user/rvanrhee/projects_rik/.venv/bin/activate

echo "Host: $(hostname -f)"
echo "Python: $(which python3)"
echo "File: $1"
python3 --version

exec python3 root_to_parquet_parallel.py --file "$1" --savedir "$2"