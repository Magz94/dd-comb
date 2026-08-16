#!/bin/bash
#SBATCH --job-name=kshot-mp-ev-moe-div
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

DATASETS=(
  deepweeds_no_bg_mergeval
  plant_seedlings_v2_split
)

for dataset in "${DATASETS[@]}"; do
  python scripts/sweep_kshot.py -m \
    stage_a_source=img100_dirichlet_imagenet_1_acc85 \
    dataset="${dataset}" \
    kshot_dataset_key="${dataset}" \
    'kshot_shot=5,10' \
    'kshot_run_idx=range(1,21)' \
    kshot_expected_runs=20 \
    kshot_results_root=kshot/results_mean_preserving_evidential_logit_moe_headdiv \
    skip_existing_kshot_results=true \
    pipeline.added_postfix=mp_evid_logitmoe_headdiv_g0p5_aux0p2_resp0p1_edl0p03_str0p05 \
    pipeline.use_dirichlet_logit_moe=true \
    pipeline.dirichlet_logit_moe_gamma=0.5 \
    pipeline.dirichlet_logit_moe_aux_lambda=0.2 \
    pipeline.dirichlet_logit_moe_resp_lambda=0.1 \
    pipeline.dirichlet_logit_moe_tau=0.5 \
    pipeline.dirichlet_logit_moe_warmup_epochs=3 \
    pipeline.use_mean_preserving_evidential_moe=true \
    pipeline.mp_evidential_moe_objective=expected_ce_kl \
    pipeline.mp_evidential_moe_edl_lambda=0.03 \
    pipeline.mp_evidential_moe_kl_lambda=1e-3 \
    pipeline.mp_evidential_moe_strength_lambda=0.05 \
    pipeline.mp_evidential_moe_strength_scale=50.0 \
    pipeline.mp_evidential_moe_strength_tau=1.0 \
    pipeline.mp_evidential_moe_warmup_epochs=3 \
    pipeline.attention_loss_mode=head_diversity \
    pipeline.attention_loss_warmup_epochs=3 \
    pipeline.attention_head_diversity_lambda=0.05 \
    pipeline.enable_uncertainty_metrics=true \
    pipeline.enable_diagnostic_plots=false \
    pipeline.use_wandb=false \
    export_kshot_predictions=false
done
