#!/bin/bash
#SBATCH --job-name=backbone-char
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-9%3
#SBATCH --output=/home/magz94/logs/slurm/sb_%A_%a.out
#SBATCH --error=/home/magz94/logs/slurm/sb_%A_%a.err
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

# Backbone-selection study: per-backbone params + inference latency + test
# embeddings (for CKA/UMAP). One array task per backbone. 5 original torchvision
# backbones + 5 modern foundation backbones = 10 tasks. DeepWeeds, original weights.
BACKBONES=(resnet18 vgg16 densenet121 efficientnet_b0 vit_b16 \
           dinov2_vitb14 openclip_vitb16 convnextv2_tiny siglip2_base mae_vitb16)
BB="${BACKBONES[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "Backbone characterization | ${BB} | $(date)"
python scripts/backbone_characterization.py \
  --backbone "${BB}" \
  --dataset deepweeds_no_bg_mergeval \
  --deepweeds-root /datasets/DeepWeeds \
  --out-root kshot/backbone_study
echo "Done ${BB} $(date)"
