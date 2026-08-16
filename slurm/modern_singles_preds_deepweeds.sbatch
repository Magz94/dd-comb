#!/bin/bash
#SBATCH --job-name=modern-singles-preds-dw
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-4%3
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

# Single-backbone predictions for the 5 MODERN foundation backbones on DeepWeeds.
# K=1 (one backbone per run) with the linear_ln probe head, softmax_ensemble (plain,
# non-evidential single head). --export-preds saves per-(shot,run) test predictions
# (head_pred, y_true) so the modern backbones can be included alongside the 5 original
# ones in the disagreement matrix and oracle-coverage analysis of the backbone study.
BACKBONES=(dinov2_vitb14 openclip_vitb16 convnextv2_tiny siglip2_base mae_vitb16)
BB="${BACKBONES[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "Modern single-backbone preds | ${BB} | $(date)"
python scripts/run_modern_backbone_experiment.py \
  --dataset deepweeds_no_bg_mergeval \
  --backbones "${BB}" \
  --combine softmax_ensemble \
  --head linear_ln \
  --width native \
  --shots 1,5,10,20 \
  --runs 1-20 \
  --expected-runs 20 \
  --export-preds \
  --results-root kshot/results_modern_singles_preds/${BB} \
  --cache-root embedding_cache_modern \
  --deepweeds-root /datasets/DeepWeeds
echo "Done ${BB} $(date)"
