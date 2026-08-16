#!/bin/bash
#SBATCH --job-name=emb-dinov3cnx
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=01:00:00
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

# --- DINOv3-ConvNeXt (license-gated): same lines as your other dinov3 job scripts.
#     The repo auto-clones if missing; the weights path is the one already on the cluster.
export DINOV3_REPO="${DINOV3_REPO:-/home/magz94/dinov3}"
if [ ! -d "${DINOV3_REPO}/dinov3" ]; then
  git clone --depth 1 https://github.com/facebookresearch/dinov3 "${DINOV3_REPO}"
fi
export DINOV3_CONVNEXT_TINY_WEIGHTS="${DINOV3_CONVNEXT_TINY_WEIGHTS:-/home/magz94/weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth}"

echo "SLURM node: ${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# Project deps (torch/torchvision come from the container and are left untouched).
python -m pip install -r environment/requirements.txt
  matplotlib seaborn scikit-learn umap-learn datasets timm open_clip_torch pillow numpy

# ---------------------------------------------------------------------------
# Extract the frozen DeepWeeds test embedding for DINOv3-ConvNeXt-Tiny.
# This is the ONLY missing piece for the three-way frozen CKA (DINOv2, ConvNeXt-V2
# and DINOv3-ConvNeXt already have their logits saved for the disagreement side).
# Without --latency-only, the script runs extract() and writes
#   kshot/backbone_study/emb_dinov3_convnext_tiny.npz   (emb, y_true)
#   kshot/backbone_study/cost_dinov3_convnext_tiny.json
# ---------------------------------------------------------------------------
python scripts/backbone_characterization.py --backbone dinov3_convnext_tiny --classes 8

echo "Done $(date)"
ls -la kshot/backbone_study/emb_dinov3_convnext_tiny.npz
