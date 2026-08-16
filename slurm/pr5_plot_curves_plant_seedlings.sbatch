#!/bin/bash
#SBATCH --job-name=pr5-plot-seedlings
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/magz94/logs/slurm/sb_%j.out
#SBATCH --error=/home/magz94/logs/slurm/sb_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user='mohamed.amine.guezguez@mni.thm.de'
#SBATCH --container-mounts='/datasets'
#SBATCH --container-workdir='/home/magz94/ddcomb'
#SBATCH --container-mount-home

set -Eeuo pipefail
mkdir -p /home/magz94/logs/slurm

cd /home/magz94/ddcomb
export PYTHONPATH=/home/magz94/ddcomb/src:${PYTHONPATH:-}
export MPLCONFIGDIR=/tmp/matplotlib
python -m pip install -r environment/requirements.txt

python scripts/plot_pr_curves_kshot5_macro.py \
  --shot 5 \
  --dataset-keys plant_seedlings_v2_split \
  --output pr_curves_kshot5_macro_plant_seedlings_v2.png \
  --metrics-csv pr_curves_kshot5_macro_plant_seedlings_v2_metrics.csv
