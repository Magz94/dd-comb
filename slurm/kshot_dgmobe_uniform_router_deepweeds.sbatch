#!/bin/bash
#SBATCH --job-name=kshot-dgmobe-uniform-dw
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

python scripts/sweep_kshot.py -m \
  stage_a_source=img100_dirichlet_imagenet_1_acc85 \
  dataset=deepweeds_no_bg_mergeval \
  kshot_dataset_key=deepweeds_no_bg_mergeval \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_expected_runs=20 \
  kshot_results_root=kshot/results_dgmobe_uniform_router_deepweeds \
  skip_existing_kshot_results=true \
  pipeline.added_postfix=dgmobe_uniform_router_headdiv_g0p5_aux0p2_resp0p1_tau0p5 \
  pipeline.use_dirichlet_logit_moe=true \
  pipeline.dirichlet_logit_moe_router=uniform \
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

echo "DG-MoBE uniform-router DeepWeeds ablation finished at: $(date)"
