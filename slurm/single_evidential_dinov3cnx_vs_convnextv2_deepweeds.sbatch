#!/bin/bash
#SBATCH --job-name=single-evid-d3cnx-cv2
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-1
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
  ftfy regex submitit termcolor   # DINOv3 backbone deps

export DINOV3_REPO="${DINOV3_REPO:-/home/magz94/dinov3}"
if [ ! -d "${DINOV3_REPO}/dinov3" ]; then
  git clone --depth 1 https://github.com/facebookresearch/dinov3 "${DINOV3_REPO}"
fi
export DINOV3_CONVNEXT_TINY_WEIGHTS="${DINOV3_CONVNEXT_TINY_WEIGHTS:-/home/magz94/weights/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth}"

# SMOKING GUN: single-backbone EVIDENTIAL models (one backbone + mean_evidence =
# that backbone's raw Dirichlet, no fusion). --export-logits saves the per-sample
# alpha so we can compare evidence magnitude / wrong-class evidence / calibration
# of DINOv3-ConvNeXt vs the ConvNeXt-V2 that works in the same combination.
ARGS="--dataset deepweeds_no_bg_mergeval --head mlp --width native --shots 1,5,10,20 --runs 1-20 \
      --expected-runs 20 --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds \
      --combine mean_evidence --export-logits"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) python scripts/run_modern_backbone_experiment.py ${ARGS} --backbones dinov3_convnext_tiny \
        --results-root kshot/results_calib/dinov3cnx_evidential_single ;;
  1) python scripts/run_modern_backbone_experiment.py ${ARGS} --backbones convnextv2_tiny \
        --results-root kshot/results_calib/convnextv2_evidential_single ;;
esac
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} $(date)"
