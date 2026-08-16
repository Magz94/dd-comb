#!/bin/bash
#SBATCH --job-name=dw-full-dgmobe
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-3
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

python -m pip install -r environment/requirements.txt

DATASET="deepweeds_no_bg_mergeval"
RESULT_ROOT_BASE="kshot/results_deepweeds_longk_professor_full"
EMB_CACHE_BASE="embedding_cache_deepweeds_full_professor"
FULL_RUNS="${FULL_RUNS:-1}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_ID}" in
  0)
    VARIANT="attention_fusion_headdiv"
    ADDED_POSTFIX="attention_fusion_headdiv_full"
    ;;
  1)
    VARIANT="dgmobe_dirichlet_router"
    ADDED_POSTFIX="dgmobe_dirichlet_router_headdiv_g0p5_aux0p2_resp0p1_tau0p5_full"
    ;;
  2)
    VARIANT="dgmobe_uniform_router"
    ADDED_POSTFIX="dgmobe_uniform_router_headdiv_g0p5_aux0p2_resp0p1_tau0p5_full"
    ;;
  3)
    VARIANT="dgmobe_experts_only_uniform"
    ADDED_POSTFIX="dgmobe_experts_only_uniform_headdiv_aux0p2_resp0p1_tau0p5_full"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0..3" >&2
    exit 2
    ;;
esac

echo "========================================"
echo "DeepWeeds FULL professor ablation"
echo "Variant: ${VARIANT}"
echo "Full runs/seeds: ${FULL_RUNS}"
echo "Result root: ${RESULT_ROOT_BASE}/${VARIANT}"
echo "Started: $(date)"

COMMON_OVERRIDES=(
  mode=stage_d
  stage_a_source=img100_dirichlet_imagenet_1_acc85
  dataset="${DATASET}"
  skip_train=false
  return_full_metrics=true
  pipeline.kshot_manifest_path=null
  pipeline.added_postfix="${ADDED_POSTFIX}"
  pipeline.attention_loss_mode=head_diversity
  pipeline.attention_loss_warmup_epochs=3
  pipeline.attention_head_diversity_lambda=0.05
  pipeline.dirichlet_kl_regularization=false
  pipeline.class_evidential_loss_mode=none
  pipeline.use_pre_fusion_uncertainty=false
  pipeline.use_evidential_fusion=false
  pipeline.use_mean_preserving_evidential_moe=false
  pipeline.enable_uncertainty_metrics=true
  pipeline.enable_diagnostic_plots=false
  pipeline.use_wandb=false
)

run_one() {
  local run_idx="$1"
  local seed=$((42 + run_idx))
  export FULL_VARIANT="${VARIANT}"
  export FULL_DATASET_KEY="${DATASET}"
  export FULL_RUN_IDX="${run_idx}"
  export FULL_EXPECTED_RUNS="${FULL_RUNS}"
  export FULL_RESULTS_ROOT="${RESULT_ROOT_BASE}"

  echo "=== Full run ${run_idx}/${FULL_RUNS}: ${VARIANT}, seed=${seed}, $(date) ==="

  case "${VARIANT}" in
    attention_fusion_headdiv)
      python scripts/train_fusion_full_to_json.py \
        "${COMMON_OVERRIDES[@]}" \
        pipeline.run_tag="deepweeds_full_${VARIANT}_run${run_idx}" \
        pipeline.seed="${seed}" \
        pipeline.emb_cache_root="${EMB_CACHE_BASE}/${VARIANT}/run${run_idx}" \
        pipeline.use_dirichlet_logit_moe=false
      ;;
    dgmobe_dirichlet_router)
      python scripts/train_fusion_full_to_json.py \
        "${COMMON_OVERRIDES[@]}" \
        pipeline.run_tag="deepweeds_full_${VARIANT}_run${run_idx}" \
        pipeline.seed="${seed}" \
        pipeline.emb_cache_root="${EMB_CACHE_BASE}/${VARIANT}/run${run_idx}" \
        pipeline.use_dirichlet_logit_moe=true \
        pipeline.dirichlet_logit_moe_router=dirichlet \
        pipeline.dirichlet_logit_moe_combine_mode=residual \
        pipeline.dirichlet_logit_moe_gamma=0.5 \
        pipeline.dirichlet_logit_moe_aux_lambda=0.2 \
        pipeline.dirichlet_logit_moe_resp_lambda=0.1 \
        pipeline.dirichlet_logit_moe_tau=0.5 \
        pipeline.dirichlet_logit_moe_warmup_epochs=3
      ;;
    dgmobe_uniform_router)
      python scripts/train_fusion_full_to_json.py \
        "${COMMON_OVERRIDES[@]}" \
        pipeline.run_tag="deepweeds_full_${VARIANT}_run${run_idx}" \
        pipeline.seed="${seed}" \
        pipeline.emb_cache_root="${EMB_CACHE_BASE}/${VARIANT}/run${run_idx}" \
        pipeline.use_dirichlet_logit_moe=true \
        pipeline.dirichlet_logit_moe_router=uniform \
        pipeline.dirichlet_logit_moe_combine_mode=residual \
        pipeline.dirichlet_logit_moe_gamma=0.5 \
        pipeline.dirichlet_logit_moe_aux_lambda=0.2 \
        pipeline.dirichlet_logit_moe_resp_lambda=0.1 \
        pipeline.dirichlet_logit_moe_tau=0.5 \
        pipeline.dirichlet_logit_moe_warmup_epochs=3
      ;;
    dgmobe_experts_only_uniform)
      python scripts/train_fusion_full_to_json.py \
        "${COMMON_OVERRIDES[@]}" \
        pipeline.run_tag="deepweeds_full_${VARIANT}_run${run_idx}" \
        pipeline.seed="${seed}" \
        pipeline.emb_cache_root="${EMB_CACHE_BASE}/${VARIANT}/run${run_idx}" \
        pipeline.use_dirichlet_logit_moe=true \
        pipeline.dirichlet_logit_moe_router=uniform \
        pipeline.dirichlet_logit_moe_combine_mode=moe_only \
        pipeline.dirichlet_logit_moe_gamma=0.5 \
        pipeline.dirichlet_logit_moe_aux_lambda=0.2 \
        pipeline.dirichlet_logit_moe_resp_lambda=0.1 \
        pipeline.dirichlet_logit_moe_tau=0.5 \
        pipeline.dirichlet_logit_moe_warmup_epochs=3
      ;;
  esac
}

for run_idx in $(seq 1 "${FULL_RUNS}"); do
  run_one "${run_idx}"
done

echo "Finished FULL variant ${VARIANT}: $(date)"
