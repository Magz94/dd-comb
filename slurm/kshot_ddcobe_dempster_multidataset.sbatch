#!/bin/bash
#SBATCH --job-name=kshot-ddcobe-multi
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

# DD-CoBE (Dempster-Dirichlet Combination of Backbone Experts) ablation across
# three datasets. The fixed KL=0.1 attribution triad from DeepWeeds:
#   pure_kl0p1_fv     - DD-CoBE champion (Dempster rule, fused head as a view)
#   pure_kl0p1_nofv   - same, experts-only (isolates the fused view's value)
#   mean_ev_kl0p1_fv  - uniform control: identical opinions+losses, evidence
#                       averaging instead of the Dempster rule
# Baselines for the narrative (uniform-router & Dirichlet-router DG-MoBE) come
# from the existing router-comparison / headdiv result folders.

DATASETS=(
  bloodmnist_224_imagefolder_mergeval
  plant_seedlings_v2_split
  ssl_patches
)
ARMS=(pure_kl0p1_fv pure_kl0p1_nofv mean_ev_kl0p1_fv)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
DS_IDX=$(( TASK_ID / 3 ))
ARM_IDX=$(( TASK_ID % 3 ))
DATASET="${DATASETS[$DS_IDX]}"
ARM="${ARMS[$ARM_IDX]}"

case "${ARM}" in
  pure_kl0p1_fv)    rule=dempster;      loss_mode=pure; kl=0.1; edl=1.0; fv=true  ;;
  pure_kl0p1_nofv)  rule=dempster;      loss_mode=pure; kl=0.1; edl=1.0; fv=false ;;
  mean_ev_kl0p1_fv) rule=mean_evidence; loss_mode=pure; kl=0.1; edl=1.0; fv=true  ;;
  *) echo "Unknown arm: ${ARM}" >&2; exit 1 ;;
esac

MANIFEST="kshot/manifests/${DATASET}/1_shot_run_1.json"
if [ ! -f "${MANIFEST}" ]; then
  echo "Missing k-shot manifest: ${MANIFEST}" >&2
  exit 2
fi

echo "========================================"
echo "DD-CoBE Dempster fusion"
echo "Dataset: ${DATASET}"
echo "Arm:     ${ARM} (rule=${rule}, loss=${loss_mode}, kl=${kl}, edl=${edl}, fused_view=${fv})"
echo "Task:    ${TASK_ID}"
echo "Started: $(date)"

python scripts/sweep_kshot.py -m \
  stage_a_source=img100_dirichlet_imagenet_1_acc85 \
  dataset="${DATASET}" \
  kshot_dataset_key="${DATASET}" \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_expected_runs=20 \
  kshot_results_root="kshot/results_ddcobe_dempster_multidataset/${ARM}" \
  skip_existing_kshot_results=true \
  pipeline.emb_cache_root="embedding_cache_ddcobe/${DATASET}" \
  pipeline.added_postfix="ddcobe_dempster_${ARM}_headdiv" \
  pipeline.use_dirichlet_logit_moe=true \
  pipeline.dirichlet_logit_moe_combine_mode=dempster \
  pipeline.dirichlet_logit_moe_router=uniform \
  pipeline.dirichlet_logit_moe_dempster_rule="${rule}" \
  pipeline.dirichlet_logit_moe_dempster_loss_mode="${loss_mode}" \
  pipeline.dirichlet_logit_moe_dempster_kl_lambda="${kl}" \
  pipeline.dirichlet_logit_moe_dempster_edl_lambda="${edl}" \
  pipeline.dirichlet_logit_moe_dempster_fused_view="${fv}" \
  pipeline.dirichlet_logit_moe_warmup_epochs=10 \
  pipeline.attention_loss_mode=head_diversity \
  pipeline.attention_loss_warmup_epochs=3 \
  pipeline.attention_head_diversity_lambda=0.05 \
  pipeline.enable_uncertainty_metrics=true \
  pipeline.enable_diagnostic_plots=false \
  pipeline.use_wandb=false \
  export_kshot_predictions=false

echo "Finished ${DATASET} / ${ARM} at: $(date)"
