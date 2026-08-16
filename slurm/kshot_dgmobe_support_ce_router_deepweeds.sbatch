#!/bin/bash
#SBATCH --job-name=kshot-dgmobe-sce-dw
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
  local router="$2"
  local beta="$3"
  local temperature="$4"
  local added_postfix="$5"

  echo "=== DG-MoBE Support-CE router: ${label} started at $(date) ==="
  python scripts/sweep_kshot.py -m \
    stage_a_source=img100_dirichlet_imagenet_1_acc85 \
    dataset=deepweeds_no_bg_mergeval \
    kshot_dataset_key=deepweeds_no_bg_mergeval \
    'kshot_shot=1,5,10,20' \
    'kshot_run_idx=range(1,21)' \
    kshot_expected_runs=20 \
    kshot_results_root="kshot/results_dgmobe_support_ce_router_deepweeds/${label}" \
    skip_existing_kshot_results=true \
    pipeline.added_postfix="${added_postfix}" \
    pipeline.use_dirichlet_logit_moe=true \
    pipeline.dirichlet_logit_moe_router="${router}" \
    pipeline.dirichlet_logit_moe_router_beta="${beta}" \
    pipeline.dirichlet_logit_moe_router_temperature="${temperature}" \
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
}

status=0

run_variant \
  support_ce \
  support_ce \
  0.5 \
  1.0 \
  dgmobe_support_ce_router_t1_headdiv_g0p5_aux0p2_resp0p1 \
  || status=1

run_variant \
  support_ce_mixed_b025 \
  support_ce_mixed \
  0.25 \
  1.0 \
  dgmobe_support_ce_mixed_b025_t1_headdiv_g0p5_aux0p2_resp0p1 \
  || status=1

run_variant \
  support_ce_mixed_b050 \
  support_ce_mixed \
  0.50 \
  1.0 \
  dgmobe_support_ce_mixed_b050_t1_headdiv_g0p5_aux0p2_resp0p1 \
  || status=1

run_variant \
  support_ce_mixed_b075 \
  support_ce_mixed \
  0.75 \
  1.0 \
  dgmobe_support_ce_mixed_b075_t1_headdiv_g0p5_aux0p2_resp0p1 \
  || status=1

echo "DG-MoBE Support-CE router DeepWeeds ablation finished at: $(date), status=${status}"
exit "${status}"
