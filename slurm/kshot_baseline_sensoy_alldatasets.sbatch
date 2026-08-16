#!/bin/bash
#SBATCH --job-name=kshot-sensoy
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-19%4
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

# Standard Sensoy EDL baseline (Sensoy et al., NeurIPS 2018): ReLU evidence ->
# Dirichlet, Brier (sum-of-squares) Bayes-risk loss + annealed KL-to-uniform.
# Single linear head on each frozen backbone embedding (baseline_hidden_dim=0),
# matching the single-backbone CE baseline so only the evidential loss differs.
# One array task per (dataset x backbone); 4 datasets x 5 backbones = 20 tasks.

DATASETS=(
  deepweeds_no_bg_mergeval
  bloodmnist_224_imagefolder_mergeval
  plant_seedlings_v2_split
  ssl_patches
)
BACKBONES=(ResNet18 VGG16 DenseNet121 EfficientNetB0 ViT-B16)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DS_IDX=$(( TASK_ID / 5 ))
BB_IDX=$(( TASK_ID % 5 ))
DATASET="${DATASETS[$DS_IDX]}"
BACKBONE="${BACKBONES[$BB_IDX]}"

MANIFEST="kshot/manifests/${DATASET}/1_shot_run_1.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "Missing k-shot manifest: ${MANIFEST}" >&2
  exit 2
fi

echo "========================================"
echo "Sensoy EDL baseline | dataset=${DATASET} | backbone=${BACKBONE} | task=${TASK_ID}"
echo "Started: $(date)"

python scripts/sweep_kshot_baselines.py -m \
  stage_a_source=img100_dirichlet_imagenet_1_acc85 \
  dataset="${DATASET}" \
  kshot_dataset_key="${DATASET}" \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_expected_runs=20 \
  kshot_results_root="kshot/results_baseline_sensoy/${BACKBONE}" \
  skip_existing_kshot_results=true \
  pipeline.baseline_method=sensoy \
  pipeline.baseline_backbone="${BACKBONE}" \
  pipeline.baseline_hidden_dim=0 \
  pipeline.sensoy_evidence=relu \
  pipeline.sensoy_loss=mse \
  pipeline.ddcobe_kl_warmup_epochs=10 \
  pipeline.use_wandb=false

echo "Finished ${DATASET} / ${BACKBONE} at: $(date)"
