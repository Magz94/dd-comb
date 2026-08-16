#!/bin/bash
#SBATCH --job-name=mcdrop-latency
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

# Record the hardware so these numbers are comparable to the existing cost_*.json
# (same container/GPU as measure_e2e_latency.sbatch).
echo "SLURM node: ${SLURMD_NODENAME:-unknown}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

python -m pip install -r environment/requirements.txt
  matplotlib seaborn scikit-learn umap-learn datasets timm open_clip_torch pillow numpy

# ---------------------------------------------------------------------------
# End-to-end MC-Dropout inference latency (frozen backbone + T=50 stochastic
# dropout-head passes) at b = 1, 32, 64. This is the only cost row missing for
# the modern methods: the single-backbone / EDL / deep-ensemble / DD-CoMB
# latencies are already in cost_dinov2_vitb14.json and cost_convnextv2_tiny.json
# (EDL adds no cost, deep ensemble and DD-CoMB = the two end-to-end times summed).
# Writes cost_mcdropout_<backbone>.json next to the other cost_*.json files.
# ---------------------------------------------------------------------------
for bb in dinov2_vitb14 convnextv2_tiny; do
  echo "=== MC-Dropout (T=50) on ${bb} ==="
  python scripts/mc_dropout_latency.py --backbone "${bb}" --classes 8 \
    --hidden-dim 512 --dropout-p 0.5 --samples 50
done
echo "Done $(date)"
