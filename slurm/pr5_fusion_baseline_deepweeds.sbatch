#!/bin/bash
#SBATCH --job-name=pr5-base-deepweeds
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --output=/home/magz94/logs/slurm/sb_%j.out
#SBATCH --error=/home/magz94/logs/slurm/sb_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user='mohamed.amine.guezguez@mni.thm.de'
#SBATCH --container-mounts='/datasets'
#SBATCH --container-workdir='/home/magz94/ddcomb'
#SBATCH --container-mount-home

set -Eeuo pipefail
mkdir -p /home/magz94/logs/slurm

export WANDB_API_KEY="$(cat ~/.secrets/wandb_api_key)"
export WANDB_ENTITY="kihug"

cd /home/magz94/ddcomb
export PYTHONPATH=/home/magz94/ddcomb/src:${PYTHONPATH:-}
python -m pip install -r environment/requirements.txt

python scripts/sweep_kshot.py -m \
  stage_a_source=img100_stagea_softmax_linear_resnet_order_v1 \
  dataset=deepweeds_no_bg_mergeval \
  kshot_dataset_key=deepweeds_no_bg_mergeval \
  kshot_shot=5 \
  'kshot_run_idx=range(1,21)' \
  export_kshot_predictions=true
