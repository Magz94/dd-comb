#!/bin/bash
#SBATCH --job-name=kshot-dir-kl-1-20
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
# results_dirichlet_attention_kl1e-4_warmup3_uncertainty_20runs root.
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
  echo "Dirichlet Attention + KL warmup: ${dataset}"
  echo "Started: $(date)"

  python scripts/sweep_kshot.py -m \
    stage_a_source=img100_dirichlet_imagenet_1_acc85 \
    dataset="${dataset}" \
    kshot_dataset_key="${dataset}" \
    'kshot_shot=1,20' \
    'kshot_run_idx=range(1,21)' \
    kshot_expected_runs=20 \
    kshot_results_root=kshot/results_dirichlet_attention_kl1e-4_warmup3_uncertainty_20runs \
    skip_existing_kshot_results=true \
    pipeline.added_postfix=dirichlet_attention_kl1e-4_warmup3_uncertainty_20runs \
    pipeline.dirichlet_kl_regularization=true \
    pipeline.dirichlet_kl_lambda=1e-4 \
    pipeline.dirichlet_kl_warmup_epochs=3 \
    pipeline.attention_loss_mode=none \
    pipeline.class_evidential_loss_mode=none \
    pipeline.use_pre_fusion_uncertainty=false \
    pipeline.use_dirichlet_logit_moe=false \
    pipeline.use_mean_preserving_evidential_moe=false \
    pipeline.use_evidential_fusion=false \
    pipeline.enable_uncertainty_metrics=true \
    pipeline.enable_diagnostic_plots=false \
    pipeline.use_wandb=false \
    export_kshot_predictions=false

  echo "Finished: $(date)"
done

echo "All Dirichlet Attention + KL 1/20-shot runs finished at: $(date)"
