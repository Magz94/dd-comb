#!/bin/bash
#SBATCH --job-name=kshot-attn-div-kl
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
KL_LAMBDAS=(1e-5 3e-5 1e-4 3e-4 1e-3)

for dataset in "${DATASETS[@]}"; do
  python scripts/sweep_kshot.py -m \
    stage_a_source=img100_dirichlet_imagenet_1_acc85 \
    dataset="${dataset}" \
    kshot_dataset_key="${dataset}" \
    'kshot_shot=5,10' \
    'kshot_run_idx=range(1,21)' \
    kshot_results_root=kshot/results_attention_head_diversity \
    pipeline.added_postfix=attn_head_diversity \
    pipeline.attention_loss_mode=head_diversity \
    pipeline.attention_loss_warmup_epochs=3 \
    pipeline.attention_head_diversity_lambda=0.05 \
    pipeline.use_wandb=false \
    export_kshot_predictions=false

  for lambda in "${KL_LAMBDAS[@]}"; do
    lambda_label="${lambda//./p}"
    python scripts/sweep_kshot.py -m \
      stage_a_source=img100_dirichlet_imagenet_1_acc85 \
      dataset="${dataset}" \
      kshot_dataset_key="${dataset}" \
      'kshot_shot=5,10' \
      'kshot_run_idx=range(1,21)' \
      kshot_results_root="kshot/results_attention_kl_lambda_${lambda_label}" \
      pipeline.added_postfix="dirkl_${lambda_label}_warmup3" \
      pipeline.dirichlet_kl_regularization=true \
      pipeline.dirichlet_kl_lambda="${lambda}" \
      pipeline.dirichlet_kl_warmup_epochs=3 \
      pipeline.use_wandb=false \
      export_kshot_predictions=false
  done
done
