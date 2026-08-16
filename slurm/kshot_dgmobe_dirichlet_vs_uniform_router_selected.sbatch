#!/bin/bash
#SBATCH --job-name=kshot-router-du
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-2%3
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

# Uniform-router control for datasets where Dirichlet-router results already exist.
DATASETS=(
  bloodmnist_224_imagefolder_mergeval
  plant_seedlings_v2_split
  ssl_patches
)

ROUTERS=(
  uniform
  uniform
  uniform
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DATASET="${DATASETS[$TASK_ID]}"
ROUTER="${ROUTERS[$TASK_ID]}"

MANIFEST="kshot/manifests/${DATASET}/1_shot_run_1.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "Missing k-shot manifest: ${MANIFEST}"
  echo "Build manifests for ${DATASET} before running this task."
  exit 2
fi

echo "========================================"
echo "DG-MoBE router comparison"
echo "Dataset: ${DATASET}"
echo "Router:  ${ROUTER}"
echo "Task:    ${TASK_ID}"
echo "Started: $(date)"

python scripts/sweep_kshot.py -m \
  stage_a_source=img100_dirichlet_imagenet_1_acc85 \
  dataset="${DATASET}" \
  kshot_dataset_key="${DATASET}" \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_expected_runs=20 \
  kshot_results_root="kshot/results_dgmobe_router_dirichlet_vs_uniform/${ROUTER}" \
  skip_existing_kshot_results=true \
  pipeline.emb_cache_root="embedding_cache_router_dirichlet_vs_uniform/${DATASET}/${ROUTER}" \
  pipeline.added_postfix="dgmobe_${ROUTER}_router_headdiv_g0p5_aux0p2_resp0p1_tau0p5_routercmp" \
  pipeline.use_dirichlet_logit_moe=true \
  pipeline.dirichlet_logit_moe_router="${ROUTER}" \
  pipeline.dirichlet_logit_moe_combine_mode=residual \
  pipeline.dirichlet_logit_moe_gamma=0.5 \
  pipeline.dirichlet_logit_moe_aux_lambda=0.2 \
  pipeline.dirichlet_logit_moe_resp_lambda=0.1 \
  pipeline.dirichlet_logit_moe_tau=0.5 \
  pipeline.dirichlet_logit_moe_warmup_epochs=3 \
  pipeline.attention_loss_mode=head_diversity \
  pipeline.attention_loss_warmup_epochs=3 \
  pipeline.attention_head_diversity_lambda=0.05 \
  pipeline.enable_uncertainty_metrics=true \
  pipeline.enable_diagnostic_plots=false \
  pipeline.use_wandb=false \
  export_kshot_predictions=false

echo "Finished at: $(date)"
