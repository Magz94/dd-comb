#!/bin/bash
#SBATCH --job-name=dw-longk-manifests
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

python -m pip install -r environment/requirements.txt

# Keep the professor-requested long-k manifests separate from the standard
# kshot/manifests/deepweeds_no_bg_mergeval 1/5/10/20 manifests.
MANIFEST_ROOT="kshot/manifests_deepweeds_longk"

python scripts/build_kshot_manifests.py \
  dataset_key=deepweeds_no_bg_mergeval \
  dataset_type=deepweeds_csv \
  deepweeds_labels_dir=/datasets/DeepWeeds/labels \
  deepweeds_subset=subset0 \
  deepweeds_drop_label8=true \
  deepweeds_merge_val_into_train=true \
  manifest_root="${MANIFEST_ROOT}" \
  'shots=[50,100,500]' \
  repeats=20 \
  base_seed=42 \
  force_resample=false

echo "Long-k DeepWeeds manifests:"
find "${MANIFEST_ROOT}/deepweeds_no_bg_mergeval" -maxdepth 1 -type f | sort | wc -l
find "${MANIFEST_ROOT}/deepweeds_no_bg_mergeval" -maxdepth 1 -type f | sort | head -12
