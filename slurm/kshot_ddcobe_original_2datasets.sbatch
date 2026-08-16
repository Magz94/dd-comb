#!/bin/bash
#SBATCH --job-name=kshot-ddcobe-orig
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-3%2
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

# DD-CoBE (Dempster) on the ORIGINAL ImageNet-1k pretrained backbones, ONLY the
# main model, comparing the two heads:
#   mlp_common - MLP adapter + LN, common 512 width (the default "as always").
#                Reuses the dempster_kl0p1 folder so finished runs are skipped.
#   linear_ln  - LayerNorm + Linear probe (no hidden adapter).
# 2 datasets x 2 heads = 4 array tasks.
DATASETS=(deepweeds_no_bg_mergeval plant_seedlings_v2_split)
HEADS=(mlp linear_ln)
LABELS=(dempster_kl0p1 dempster_linear_ln)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${DATASETS[$(( TASK_ID / 2 ))]}"
HIDX=$(( TASK_ID % 2 ))
HEAD="${HEADS[$HIDX]}"
LABEL="${LABELS[$HIDX]}"

MANIFEST="kshot/manifests/${DATASET}/1_shot_run_1.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "Missing k-shot manifest: ${MANIFEST}" >&2
  exit 2
fi

echo "========================================"
echo "DD-CoBE (ORIGINAL) Dempster | dataset=${DATASET} | head=${HEAD} (${LABEL})"
echo "Started: $(date)"

python scripts/sweep_kshot_ddcobe.py -m \
  stage_a_source=imagenet1k_original_5bb \
  dataset="${DATASET}" \
  kshot_dataset_key="${DATASET}" \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_expected_runs=20 \
  kshot_results_root="kshot/results_ddcobe_original_5bb/${LABEL}" \
  skip_existing_kshot_results=true \
  pipeline.ddcobe_combine=dempster \
  pipeline.ddcobe_kl_lambda=0.1 \
  pipeline.ddcobe_kl_warmup_epochs=10 \
  pipeline.ddcobe_head_type=${HEAD} \
  pipeline.use_wandb=false

echo "Finished ${DATASET} / ${HEAD} at: $(date)"
