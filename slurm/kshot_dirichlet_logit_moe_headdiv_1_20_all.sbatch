#!/bin/bash
#SBATCH --job-name=kshot-moe-div-1-20
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

# Continuation job: adds 1-shot and 20-shot to the existing
# results_dirichlet_logit_moe_headdiv root.
DATASETS=(
  deepweeds_no_bg_mergeval
  plant_seedlings_v2_split
  bloodmnist_224_imagefolder_mergeval
  aptos2019_split
  leafsugar_disease_split
  sar_cld_original_split
)

for dataset in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Dirichlet Logit-MoE + HeadDiv: ${dataset}"
  echo "Started: $(date)"

  python scripts/sweep_kshot.py -m \
    stage_a_source=img100_dirichlet_imagenet_1_acc85 \
    dataset="${dataset}" \
    kshot_dataset_key="${dataset}" \
    'kshot_shot=1,20' \
    'kshot_run_idx=range(1,21)' \
    kshot_expected_runs=20 \
    kshot_results_root=kshot/results_dirichlet_logit_moe_headdiv \
    skip_existing_kshot_results=true \
    pipeline.added_postfix=dirlogitmoe_headdiv_g0p5_aux0p2_resp0p1_tau0p5 \
    pipeline.use_dirichlet_logit_moe=true \
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

  echo "Finished: $(date)"
done

echo "All Dirichlet Logit-MoE + HeadDiv 1/20-shot runs finished at: $(date)"
