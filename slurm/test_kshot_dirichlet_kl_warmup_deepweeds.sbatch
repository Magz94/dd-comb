#!/bin/bash
#SBATCH --job-name=dirklwarm-kshot-deepweeds
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
export WANDB_API_KEY="$(cat ~/.secrets/wandb_api_key)"
export WANDB_ENTITY="kihug"
export PYTHONPATH=/home/magz94/ddcomb/src:${PYTHONPATH:-}

python -m pip install -r environment/requirements.txt

# Fusion sweep: DeepWeeds 1/5/10/20-shot, 20 runs each.
# Same KL regularizer as the constant-KL run, but linearly warmed up over 3 epochs.
# Safety:
#   - kshot_results_root is NOT kshot/results
#   - added_postfix makes checkpoints distinct from normal k-shot checkpoints
python scripts/sweep_kshot.py -m \
  stage_a_source=img100_dirichlet_imagenet_1_acc85 \
  dataset=deepweeds_no_bg_mergeval \
  kshot_dataset_key=deepweeds_no_bg_mergeval \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_results_root=kshot/results_dirichlet_kl_warmup_test \
  skip_existing_kshot_results=false \
  pipeline.added_postfix=dirklwarmup1e-4_test \
  pipeline.dirichlet_kl_regularization=true \
  pipeline.dirichlet_kl_lambda=1e-4 \
  pipeline.dirichlet_kl_warmup_epochs=3 \
  pipeline.use_wandb=false

# Single-backbone references on the same manifests, also in a separate result root.
# This is only a sanity/reference run; the KL regularizer applies to fusion only.
python scripts/sweep_single_backbone_kshot.py -m \
  single_backbone_dataset=deepweeds_no_bg_mergeval \
  model=vit_b16,densenet121,efficientnet_b0,resnet18,vgg16 \
  mode=head_only \
  init_base_models_path=models/base \
  init_base_subfolder=acc85 \
  kshot_dataset_key=deepweeds_no_bg_mergeval \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_results_root=kshot/results_single_backbone_dirichlet_kl_warmup_test \
  output_dir=models/single_backbone_kshot_dirichlet_kl_warmup_test \
  skip_existing_kshot_results=false \
  use_wandb=false
