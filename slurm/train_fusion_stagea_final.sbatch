#!/bin/bash
#SBATCH --job-name=stagea-fusion-final
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
  mode=stage_a \
  pipeline.use_jkk=false \
  pipeline.pretrained=false \
  pipeline.base_models_path=models/base \
  pipeline.attention_method=dirichlet \
  pipeline.classifier_method=linear \
  pipeline.num_classes=100 \
  pipeline.run_tag=attn_stage_a_dirichlet_linear_imagenet100_acc85 \
  pipeline.use_wandb=true \
  'pipeline.subfolders=[acc85]' \
  'pipeline.model_names=[ResNet18,VGG16,DenseNet121,EfficientNetB0,ViT-B16]'

python scripts/train_fusion.py \
  mode=stage_a \
  pipeline.use_jkk=false \
  pipeline.pretrained=false \
  pipeline.base_models_path=models/base \
  pipeline.attention_method=softmax \
  pipeline.classifier_method=linear \
  pipeline.num_classes=100 \
  pipeline.run_tag=attn_stage_a_softmax_linear_imagenet100_acc85_resnetOrder_v1 \
  pipeline.shuffle_model_order=false \
  pipeline.use_wandb=true \
  'pipeline.subfolders=[acc85]' \
  'pipeline.model_names=[ResNet18,VGG16,DenseNet121,EfficientNetB0,ViT-B16]'
