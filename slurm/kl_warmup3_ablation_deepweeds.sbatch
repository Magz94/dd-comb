#!/bin/bash
#SBATCH --job-name=klwarm3-dw
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

# KL-warmup ablation on DeepWeeds: identical to the results_calib export but with Tw=3 instead of 10.
# epochs stays 10, so the KL ramps over epochs 1-3 then sits at full weight for epochs 4-10
# (the sustained full-weight phase that Tw=10 never reaches). Fresh results root, nothing overwritten.
# Both evidential rules are re-run so the vacuity comparison of sec:modern-pair can be redone.
FUS="--dataset deepweeds_no_bg_mergeval --head mlp --width native --shots 1,5,10,20 --runs 1-20 \
     --expected-runs 20 --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds \
     --export-logits --epochs 10 --kl-warmup 3"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14,convnextv2_tiny \
        --combine dempster       --results-root kshot/results_calib_tw3/dino_cnx_dempster ;;
  1) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14,convnextv2_tiny \
        --combine mean_evidence  --results-root kshot/results_calib_tw3/dino_cnx_mean_evidence ;;
esac
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} $(date)"
