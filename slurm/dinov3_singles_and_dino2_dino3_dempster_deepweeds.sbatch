#!/bin/bash
#SBATCH --job-name=dinov3-dw
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-2
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
  ftfy regex submitit termcolor   # DINOv3 backbone deps (from dinov3/requirements.txt)

# DINOv3 source repo (code is public; only the weights are gated). We import the
# backbone builders directly from here, bypassing the repo's hubconf.py (which
# imports the segmentation/depth stack and needs extra deps like termcolor).
export DINOV3_REPO="${DINOV3_REPO:-/home/magz94/dinov3}"
if [ ! -d "${DINOV3_REPO}/dinov3" ]; then
  git clone --depth 1 https://github.com/facebookresearch/dinov3 "${DINOV3_REPO}"
fi

# Gated weights: local .pth paths (download once after accepting the DINOv3 license).
export DINOV3_VITB16_WEIGHTS="${DINOV3_VITB16_WEIGHTS:-/home/magz94/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}"
export DINOV3_CONVNEXT_TINY_WEIGHTS="${DINOV3_CONVNEXT_TINY_WEIGHTS:-/home/magz94/weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth}"

# Same protocol as the existing modern runs: mlp head, native width, k in {1,5,10,20},
# 20 runs, with --export-logits for temperature scaling / vacuity scoring.
FUS="--dataset deepweeds_no_bg_mergeval --head mlp --width native --shots 1,5,10,20 --runs 1-20 \
     --expected-runs 20 --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds --export-logits"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  # Single backbones (the single-backbone path = one backbone + softmax_ensemble combine).
  0) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov3_vitb16 \
        --combine softmax_ensemble --results-root kshot/results_calib/dinov3_vitb16_standard ;;
  1) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov3_convnext_tiny \
        --combine softmax_ensemble --results-root kshot/results_calib/dinov3_convnext_tiny_standard ;;
  # Dempster combination: existing DINOv2 ViT-B/14 + new DINOv3 ConvNeXt-Tiny.
  2) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14,dinov3_convnext_tiny \
        --combine dempster --results-root kshot/results_calib/dino2_dino3cnx_dempster ;;
esac
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} $(date)"
