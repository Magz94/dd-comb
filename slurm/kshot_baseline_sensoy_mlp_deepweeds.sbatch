#!/bin/bash
#SBATCH --job-name=sensoy-mlp-dw
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-1
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

# Sensoy EDL on the NATIVE-WIDTH MLP head (hidden=768 = embedding dim of both
# dinov2_vitb14 and convnextv2_tiny), to head-match the fusion's per-backbone MLP
# heads and MC-Dropout. The standard linear-head (hidden=0) Sensoy runs already
# exist under kshot/results_calib/baselines and are kept as the secondary recipe;
# these MLP runs go to a FRESH dir so nothing is overwritten. --export-logits saves
# per-(shot,run) TEST + CALIB(val) logits + alpha for temperature scaling / vacuity.
ARGS="--method sensoy --hidden 768 --dataset deepweeds_no_bg_mergeval \
      --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
      --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds \
      --export-logits --results-root kshot/results_calib/baselines_edl_mlp"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) python scripts/run_modern_baselines.py --backbone dinov2_vitb14   ${ARGS} ;;
  1) python scripts/run_modern_baselines.py --backbone convnextv2_tiny ${ARGS} ;;
esac
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} $(date)"
