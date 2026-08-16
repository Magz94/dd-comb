#!/bin/bash
#SBATCH --job-name=kshot-ddcobe-clean-multi
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-8%3
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

# Clean, attention-free DD-CoBE ensemble (models/ddcobe_ensemble.py) extended to
# three more datasets, to reproduce the DeepWeeds attribution on broader data.
# The clean-architecture ladder (all share the same MLP head + joint training):
#   softmax_ensemble     - non-evidential floor (avg softmax),   KL=0.0
#   mean_evidence_kl0p1  - mean-evidence control (avg evidence),  KL=0.1
#   dempster_kl0p1       - DD-CoBE (reduced Dempster rule),       KL=0.1
# No attention, no router, no fused view, no head-div. Same loss structure as
# results_ddcobe_clean_deepweeds. (Vanilla deep ensemble is a separate single-
# backbone export pipeline and is reported on the primary dataset, DeepWeeds.)

DATASETS=(
  bloodmnist_224_imagefolder_mergeval
  plant_seedlings_v2_split
  ssl_patches
)
VARIANTS=(softmax_ensemble mean_evidence_kl0p1 dempster_kl0p1)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DS_IDX=$(( TASK_ID / 3 ))
VAR_IDX=$(( TASK_ID % 3 ))
DATASET="${DATASETS[$DS_IDX]}"
VARIANT="${VARIANTS[$VAR_IDX]}"

case "${VARIANT}" in
  softmax_ensemble)    combine=softmax_ensemble; kl=0.0 ;;
  mean_evidence_kl0p1) combine=mean_evidence;    kl=0.1 ;;
  dempster_kl0p1)      combine=dempster;         kl=0.1 ;;
  *) echo "Unknown variant: ${VARIANT}" >&2; exit 1 ;;
esac

MANIFEST="kshot/manifests/${DATASET}/1_shot_run_1.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "Missing k-shot manifest: ${MANIFEST}" >&2
  exit 2
fi

echo "========================================"
echo "Clean DD-CoBE (multidataset)"
echo "Dataset: ${DATASET}"
echo "Variant: ${VARIANT} (combine=${combine}, kl=${kl})"
echo "Task:    ${TASK_ID}"
echo "Started: $(date)"

python scripts/sweep_kshot_ddcobe.py -m \
  stage_a_source=img100_dirichlet_imagenet_1_acc85 \
  dataset="${DATASET}" \
  kshot_dataset_key="${DATASET}" \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_expected_runs=20 \
  kshot_results_root="kshot/results_ddcobe_clean_multidataset/${VARIANT}" \
  skip_existing_kshot_results=true \
  pipeline.ddcobe_combine=${combine} \
  pipeline.ddcobe_kl_lambda=${kl} \
  pipeline.ddcobe_kl_warmup_epochs=10 \
  pipeline.use_wandb=false

echo "Finished ${DATASET} / ${VARIANT} at: $(date)"
