#!/bin/bash
#SBATCH --job-name=sensoy-mlp-mds
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
# MLP-head Sensoy EDL (--hidden 768) for the multidataset study, matching the
# variant already run on DeepWeeds (baselines_edl_mlp). The standard LINEAR Sensoy
# collapses to a near-vacuous, below-random solution on weak features (e.g. DINOv2
# on BloodMNIST); the MLP head gives it fair capacity. EXPORT ON + PER-TASK PRIVATE
# cache (confirmed-safe recipe). One task per (dataset x backbone).
# Results -> kshot/results_calib_fix_exp/baselines_edl_mlp/sensoy/<backbone>/<dataset>/
# ---------------------------------------------------------------------------
RR="kshot/results_calib_fix_exp/baselines_edl_mlp"
CACHE="embedding_cache_fix_edlmlp/t${SLURM_ARRAY_TASK_ID:-0}"   # private per task

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) DS="plant_seedlings_v2_split";            IMROOT="data/plant_seedlings_v2_split"; BB="dinov2_vitb14"  ;;
  1) DS="plant_seedlings_v2_split";            IMROOT="data/plant_seedlings_v2_split"; BB="convnextv2_tiny";;
  2) DS="ssl_patches";                         IMROOT="data/ssl_patches";              BB="dinov2_vitb14"  ;;
  3) DS="ssl_patches";                         IMROOT="data/ssl_patches";              BB="convnextv2_tiny";;
  4) DS="bloodmnist_224_imagefolder_mergeval"; IMROOT="data/bloodmnist_224_imagefolder"; BB="dinov2_vitb14"  ;;
  5) DS="bloodmnist_224_imagefolder_mergeval"; IMROOT="data/bloodmnist_224_imagefolder"; BB="convnextv2_tiny";;
esac

python scripts/run_modern_baselines.py --method sensoy --hidden 768 \
  --dataset "${DS}" --imagefolder-root "${IMROOT}" --backbone "${BB}" \
  --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
  --cache-root "${CACHE}" --export-logits --results-root "${RR}"
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} ${DS} ${BB} $(date)"
