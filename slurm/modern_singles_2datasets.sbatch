#!/bin/bash
#SBATCH --job-name=modern-singles
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/home/magz94/logs/slurm/sb_%A_%a.out
#SBATCH --error=/home/magz94/logs/slurm/sb_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user='mohamed.amine.guezguez@mni.thm.de'
#SBATCH --container-mounts='/datasets'
#SBATCH --container-workdir='/home/magz94/ddcomb'
#SBATCH --container-mount-home

set -Eeuo pipefail
mkdir -p /home/magz94/logs/slurm
cd /home/magz94/ddcomb
export PYTHONPATH=/home/magz94/ddcomb/src:${PYTHONPATH:-}
export HYDRA_FULL_ERROR=1

python -m pip install -r environment/requirements.txt

# Modern backbones, each ALONE (single-model baseline), with the linear_ln head
# = the standard linear-probe protocol for frozen foundation features. DINOv2 /
# OpenCLIP / ConvNeXt-V2 with their own preprocessing, via the clean
# self-contained pipeline. 2 datasets x 3 backbones = 6 array tasks.
DATASETS=(deepweeds_no_bg_mergeval plant_seedlings_v2_split)
BACKBONES=(dinov2_vitb14 openclip_vitb16 convnextv2_tiny)

TID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${DATASETS[$(( TID / 3 ))]}"
BB="${BACKBONES[$(( TID % 3 ))]}"

echo "Modern single | ${DATASET} | ${BB} | head=linear_ln | $(date)"
python scripts/run_modern_backbone_experiment.py \
  --dataset "${DATASET}" \
  --backbones "${BB}" \
  --combine softmax_ensemble --width common --head linear_ln \
  --shots 1,5,10,20 --runs 1-20 \
  --results-root "kshot/results_modern_singles/linear_ln/${BB}"
echo "Done ${DATASET}/${BB} $(date)"
