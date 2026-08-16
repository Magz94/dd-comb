#!/bin/bash
#SBATCH --job-name=stagea-prefusion
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

python scripts/train_fusion.py \
  stage_a_source=img100_dirichlet_prefusion_uncertainty_acc85 \
  mode=stage_a \
  pipeline.use_jkk=false \
  pipeline.num_classes=100 \
  pipeline.run_tag=attn_stage_a_dirichlet_linear_prefusion_uncertainty_imagenet100_acc85_resnet_vgg_densenet_effnet_vit \
  pipeline.use_wandb=true
