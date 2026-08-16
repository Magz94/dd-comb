#!/bin/bash
#SBATCH --job-name=e2e-latency
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=02:00:00
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

# Record the hardware so the latency numbers are reproducible and comparable.
echo "SLURM node: ${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# The package is imported via PYTHONPATH=src (above), so it need not be built/installed;
# it only needs its dependencies present. Importing ddcomb pulls in the full
# __init__ (config -> omegaconf, evaluation, pipeline, ...), so install the project deps from
# pyproject.toml plus timm/open_clip (backbone libs not listed there). torch/torchvision are
# provided by the container and are left untouched to avoid breaking the CUDA build.
python -m pip install -r environment/requirements.txt
  matplotlib seaborn scikit-learn umap-learn datasets timm open_clip_torch pillow numpy

# ---------------------------------------------------------------------------
# End-to-end per-image latency (frozen backbone + native-width MLP head) at
# b = 1, 32, 64, in inference mode. --latency-only updates cost_<name>.json in
# place and leaves the cached embeddings (emb_<name>.npz) untouched.
# DD-CoMB latency = sum of its backbones' latencies (five extractors run in
# sequence); the single-backbone latency is that backbone's value.
# ---------------------------------------------------------------------------
for bb in dinov2_vitb14 convnextv2_tiny resnet18 vgg16 densenet121 efficientnet_b0 vit_b16; do
  echo "=== ${bb} ==="
  python scripts/backbone_characterization.py --backbone "${bb}" --latency-only --classes 8
done
echo "Done $(date)"
