#!/bin/bash
#SBATCH --job-name=dw-longk-dgmobe
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
MANIFEST_ROOT="kshot/manifests_deepweeds_longk"
RESULT_ROOT_BASE="kshot/results_deepweeds_longk_professor"
EMB_CACHE_BASE="embedding_cache_deepweeds_longk_professor"

# Build/verify only if the separate long-k manifests are not present yet.
if [ ! -f "${MANIFEST_ROOT}/${DATASET}/50_shot_run_1.json" ]; then
  echo "Missing long-k manifests. Building them under ${MANIFEST_ROOT}."
  python scripts/build_kshot_manifests.py \
    dataset_key="${DATASET}" \
    dataset_type=deepweeds_csv \
    deepweeds_labels_dir=/datasets/DeepWeeds/labels \
    deepweeds_subset=subset0 \
    deepweeds_drop_label8=true \
    deepweeds_merge_val_into_train=true \
    manifest_root="${MANIFEST_ROOT}" \
    'shots=[50,100,500]' \
    repeats=20 \
    base_seed=42 \
    force_resample=false
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
case "${TASK_ID}" in
  0)
    VARIANT="attention_fusion_headdiv"
    ADDED_POSTFIX="attention_fusion_headdiv_longk"
    ;;
  1)
    VARIANT="dgmobe_dirichlet_router"
    ADDED_POSTFIX="dgmobe_dirichlet_router_headdiv_g0p5_aux0p2_resp0p1_tau0p5_longk"
    ;;
  2)
    VARIANT="dgmobe_uniform_router"
    ADDED_POSTFIX="dgmobe_uniform_router_headdiv_g0p5_aux0p2_resp0p1_tau0p5_longk"
    ;;
  3)
    VARIANT="dgmobe_experts_only_uniform"
    ADDED_POSTFIX="dgmobe_experts_only_uniform_headdiv_aux0p2_resp0p1_tau0p5_longk"
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID=${TASK_ID}; expected 0..3" >&2
    exit 2
    ;;
esac

echo "========================================"
echo "DeepWeeds long-k professor ablation"
echo "Variant: ${VARIANT}"
echo "Shots: 50,100,500"
echo "Runs: 1..20"
echo "Manifest root: ${MANIFEST_ROOT}"
echo "Result root: ${RESULT_ROOT_BASE}/${VARIANT}"
echo "Started: $(date)"

COMMON_OVERRIDES=(
  stage_a_source=img100_dirichlet_imagenet_1_acc85
  dataset="${DATASET}"
  kshot_dataset_key="${DATASET}"
  kshot_root="${MANIFEST_ROOT}"
  'kshot_shot=50,100,500'
  'kshot_run_idx=range(1,21)'
  kshot_expected_runs=20
  kshot_results_root="${RESULT_ROOT_BASE}/${VARIANT}"
  skip_existing_kshot_results=true
  pipeline.emb_cache_root="${EMB_CACHE_BASE}/${VARIANT}"
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
  export_kshot_predictions=false
)

case "${VARIANT}" in
  attention_fusion_headdiv)
    python scripts/sweep_kshot.py -m \
      "${COMMON_OVERRIDES[@]}" \
      pipeline.use_dirichlet_logit_moe=false
    ;;
  dgmobe_dirichlet_router)
    python scripts/sweep_kshot.py -m \
      "${COMMON_OVERRIDES[@]}" \
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
    python scripts/sweep_kshot.py -m \
      "${COMMON_OVERRIDES[@]}" \
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
    python scripts/sweep_kshot.py -m \
      "${COMMON_OVERRIDES[@]}" \
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

echo "Finished variant ${VARIANT}: $(date)"
