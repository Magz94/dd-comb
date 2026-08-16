#!/bin/bash
#SBATCH --job-name=dino3-dino3-dempster-dw
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
  ftfy regex submitit termcolor   # DINOv3 backbone deps (from dinov3/requirements.txt)

# DINOv3 source repo (public code; weights gated). Backbones imported directly,
# bypassing hubconf (which pulls the segmentation/depth stack).
export DINOV3_REPO="${DINOV3_REPO:-/home/magz94/dinov3}"
if [ ! -d "${DINOV3_REPO}/dinov3" ]; then
  git clone --depth 1 https://github.com/facebookresearch/dinov3 "${DINOV3_REPO}"
fi
export DINOV3_VITB16_WEIGHTS="${DINOV3_VITB16_WEIGHTS:-/home/magz94/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
export DINOV3_CONVNEXT_TINY_WEIGHTS="${DINOV3_CONVNEXT_TINY_WEIGHTS:-/home/magz94/weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth}"

# Dempster combination of the two DINOv3 backbones (ViT-B/16 + ConvNeXt-Tiny),
# same protocol as the other modern runs (mlp head, native width, k in {1,5,10,20},
# 20 runs, with logits export for TS / vacuity).
python scripts/run_modern_backbone_experiment.py \
  --dataset deepweeds_no_bg_mergeval --head mlp --width native --shots 1,5,10,20 --runs 1-20 \
  --expected-runs 20 --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds --export-logits \
  --backbones dinov3_vitb16,dinov3_convnext_tiny --combine dempster \
  --results-root kshot/results_calib/dino3vit_dino3cnx_dempster
echo "Done $(date)"
