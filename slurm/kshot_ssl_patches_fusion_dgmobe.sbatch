#!/bin/bash
#SBATCH --job-name=kshot-ssl-dgmobe
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
export HYDRA_FULL_ERROR=1

python -m pip install -r environment/requirements.txt

if [ ! -f kshot/manifests/ssl_patches/1_shot_run_1.json ]; then
  echo "Missing k-shot manifests. Run: sbatch slurm/prepare_ssl_patches.sbatch"
  exit 1
fi

echo "========================================"
echo "SSL_Patches DG-MoBE + HeadDiv"
echo "Started: $(date)"

python scripts/sweep_kshot.py -m \
  stage_a_source=img100_dirichlet_imagenet_1_acc85 \
  dataset=ssl_patches \
  kshot_dataset_key=ssl_patches \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_expected_runs=20 \
  kshot_results_root=kshot/results_dirichlet_logit_moe_headdiv \
  skip_existing_kshot_results=true \
  pipeline.emb_cache_root=embedding_cache_ssl_patches/fusion_dgmobe \
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

echo "SSL_Patches DG-MoBE sweep finished at: $(date)"
