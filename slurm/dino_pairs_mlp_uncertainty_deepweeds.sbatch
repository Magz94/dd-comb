#!/bin/bash
#SBATCH --job-name=dino-pairs-mlp-unc-dw
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-4
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

# Goal: keep accuracy ~ DINOv2-B (0.866) but IMPROVE uncertainty (ECE / AUROC-H /
# FPR95-H). Base tier only (no -L models). MLP head, native widths (the standard
# adapter). Evidential Dempster rule only (the method).
# One array task per combo (run concurrently). cnx = ConvNeXt-V2-Tiny,
# cnxB = ConvNeXt-V2-Base (size-matched to the base ViTs). convnextv2_base
# embeddings are extracted + cached automatically on first use.
NAMES=(dino_cnx dino_siglip dino_cnx_siglip dino_cnxB dino_cnxB_siglip)
BBS=("dinov2_vitb14,convnextv2_tiny" \
     "dinov2_vitb14,siglip2_base" \
     "dinov2_vitb14,convnextv2_tiny,siglip2_base" \
     "dinov2_vitb14,convnextv2_base" \
     "dinov2_vitb14,convnextv2_base,siglip2_base")
NAME="${NAMES[${SLURM_ARRAY_TASK_ID:-0}]}"
BB="${BBS[${SLURM_ARRAY_TASK_ID:-0}]}"

echo "==== ${NAME} (${BB}) rule=dempster mlp/native $(date) ===="
python scripts/run_modern_backbone_experiment.py \
  --dataset deepweeds_no_bg_mergeval --head mlp --width native \
  --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
  --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds \
  --backbones "${BB}" --combine dempster \
  --results-root kshot/results_dino_pairs_mlp/${NAME}_dempster
echo "Done ${NAME} $(date)"
