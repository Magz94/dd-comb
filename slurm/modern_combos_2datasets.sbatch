#!/bin/bash
#SBATCH --job-name=modern-combos
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-7%3
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

# DD-CoBE (Dempster) over modern-backbone COMBINATIONS to pick the best 2-3.
# All pairs + the triple, common width. 2 datasets x 4 combos = 8 array tasks.
DATASETS=(deepweeds_no_bg_mergeval plant_seedlings_v2_split)
COMBOS=(
  "dinov2_vitb14,openclip_vitb16"
  "dinov2_vitb14,convnextv2_tiny"
  "openclip_vitb16,convnextv2_tiny"
  "dinov2_vitb14,openclip_vitb16,convnextv2_tiny"
)
LABELS=(dino_clip dino_cnx clip_cnx dino_clip_cnx)

TID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${DATASETS[$(( TID / 4 ))]}"
CIDX=$(( TID % 4 ))
COMBO="${COMBOS[$CIDX]}"
LABEL="${LABELS[$CIDX]}"

echo "Modern combo | ${DATASET} | ${LABEL} (${COMBO}) | $(date)"
python scripts/run_modern_backbone_experiment.py \
  --dataset "${DATASET}" \
  --backbones "${COMBO}" \
  --combine dempster --kl 0.1 --width common --head linear_ln \
  --shots 1,5,10,20 --runs 1-20 \
  --results-root "kshot/results_modern_combos/${LABEL}_dempster_linear_ln"
echo "Done ${DATASET}/${LABEL} $(date)"
