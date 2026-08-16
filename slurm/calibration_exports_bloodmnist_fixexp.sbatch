#!/bin/bash
#SBATCH --job-name=calib-fixexp-blood
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-5
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

# ---------------------------------------------------------------------------
# CLEAN RE-RUN (BloodMNIST) — all 8 models, EXPORT ON, PER-TASK PRIVATE cache.
# BloodMNIST was in the same concurrent array as the corrupted plant/ssl jobs and
# its DINOv2 numbers looked suspicious, so we re-run it under the confirmed-safe
# recipe (private per-task cache => no shared-cache race; exports safe). Lands in
# results_calib_fix_exp alongside plant/ssl, giving one clean logits-included source.
# ---------------------------------------------------------------------------
DS="bloodmnist_224_imagefolder_mergeval"
IMROOT="data/bloodmnist_224_imagefolder"
CACHE="embedding_cache_fix/tb${SLURM_ARRAY_TASK_ID:-0}"     # private per task (tb* = blood)
RR="kshot/results_calib_fix_exp"
FUS="--dataset ${DS} --imagefolder-root ${IMROOT} --head mlp --width native \
     --shots 1,5,10,20 --runs 1-20 --expected-runs 20 --cache-root ${CACHE} --export-logits"
BASE="--dataset ${DS} --imagefolder-root ${IMROOT} --shots 1,5,10,20 --runs 1-20 \
      --expected-runs 20 --cache-root ${CACHE} --export-logits"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14 \
        --combine softmax_ensemble --results-root ${RR}/dinov2_standard ;;
  1) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones convnextv2_tiny \
        --combine softmax_ensemble --results-root ${RR}/convnextv2_standard ;;
  2) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14,convnextv2_tiny \
        --combine dempster --results-root ${RR}/dino_cnx_dempster ;;
  3) python scripts/run_vanilla_deep_ensemble.py ${BASE} --backbones dinov2_vitb14,convnextv2_tiny \
        --results-root ${RR}/vanilla_ens ;;
  4) python scripts/run_modern_baselines.py ${BASE} --backbone dinov2_vitb14 --method both \
        --results-root ${RR}/baselines ;;
  5) python scripts/run_modern_baselines.py ${BASE} --backbone convnextv2_tiny --method both \
        --results-root ${RR}/baselines ;;
esac
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} $(date)"
