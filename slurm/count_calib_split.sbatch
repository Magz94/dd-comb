#!/bin/bash
#SBATCH --job-name=count-calib
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/magz94/logs/slurm/sb_%j.out
#SBATCH --error=/home/magz94/logs/slurm/sb_%j.err
#SBATCH --container-mounts='/datasets'
#SBATCH --container-workdir='/home/magz94/ddcomb'
#SBATCH --container-mount-home

set -Eeuo pipefail
cd /home/magz94/ddcomb
export PYTHONPATH=/home/magz94/ddcomb/src:${PYTHONPATH:-}

# POINT 2: exact size (and per-class breakdown) of the temperature-scaling
# calibration split. It is the official DeepWeeds subset0 val split (background
# class 8 dropped) -- a fixed labelled split, NOT the k-shot support set.
# Standalone: reads the val CSV directly (Filename,Label), no package imports.
python - <<'PY'
import csv, os
from collections import Counter
labels_dir = "/datasets/DeepWeeds/labels"
cands = [f"{labels_dir}/val_subset0.csv", f"{labels_dir}/subset0/val.csv"]
path = next((c for c in cands if os.path.exists(c)), None)
assert path, f"No DeepWeeds val CSV found. Checked: {cands}"
labs = [int(r["Label"]) for r in csv.DictReader(open(path)) if int(r["Label"]) != 8]
print("CSV:", path)
print("DeepWeeds subset0 val -> TS calibration split size (excl. background):", len(labs))
print("per-class:", dict(sorted(Counter(labs).items())))
PY
echo "Done $(date)"
