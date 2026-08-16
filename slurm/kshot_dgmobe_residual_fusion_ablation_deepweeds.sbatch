#!/bin/bash
#SBATCH --job-name=kshot-dgmobe-residual-dw
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --output=/home/magz94/logs/slurm/sb_%j.out
#SBATCH --error=/home/magz94/logs/slurm/sb_%j.err
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

run_variant() {
  local label="$1"
  local combine_mode="$2"
  local gamma="$3"

  echo "=== DG-MoBE residual-fusion ablation: ${label} started at $(date) ==="
  python scripts/sweep_kshot.py -m \
    stage_a_source=img100_dirichlet_imagenet_1_acc85 \
    dataset=deepweeds_no_bg_mergeval \
    kshot_dataset_key=deepweeds_no_bg_mergeval \
    'kshot_shot=1,5,10,20' \
    'kshot_run_idx=range(1,21)' \
    kshot_expected_runs=20 \
    kshot_results_root="kshot/results_dgmobe_residual_fusion_ablation_deepweeds/${label}" \
    skip_existing_kshot_results=true \
    pipeline.added_postfix="dgmobe_residual_ablation_${label}" \
    pipeline.use_dirichlet_logit_moe=true \
    pipeline.dirichlet_logit_moe_router=uniform \
    pipeline.dirichlet_logit_moe_combine_mode="${combine_mode}" \
    pipeline.dirichlet_logit_moe_gamma="${gamma}" \
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
}

status=0

# Tests whether expert heads/losses alone help when their logits are not added at inference.
run_variant fused_only fused_only 0.5 || status=1

# Tests whether replacing the fused classifier with routed expert logits is sufficient.
run_variant moe_only moe_only 0.5 || status=1

# Current idea: keep fused prediction and add the routed expert correction.
run_variant residual_g0p5 residual 0.5 || status=1

# Optional strength check: does a stronger residual correction help or overrule the fused path?
run_variant residual_g1p0 residual 1.0 || status=1

echo "DG-MoBE residual-fusion ablation finished at: $(date), status=${status}"
exit "${status}"
