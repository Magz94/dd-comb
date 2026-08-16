#!/bin/bash
#SBATCH --job-name=dino-cnx-base-rules-dw
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-3
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

# Completes the DINOv2 + ConvNeXt-V2-Tiny study (the winning pair):
#  tasks 0-1: single-backbone baselines (Sensoy EDL + MC Dropout, standard recipe)
#             for DINOv2 and ConvNeXt-V2-Tiny -- same as for the original backbones.
#  tasks 2-3: the SAME pair fused with the other two rules (softmax_ensemble = deep
#             ensemble, mean_evidence = Jensen averaging), mlp head + native width,
#             identical config to the Dempster run, for a like-for-like comparison.
COMMON_FUS="--dataset deepweeds_no_bg_mergeval --head mlp --width native \
            --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
            --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds \
            --backbones dinov2_vitb14,convnextv2_tiny"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    python scripts/run_modern_baselines.py --backbone dinov2_vitb14   --method both \
      --dataset deepweeds_no_bg_mergeval --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
      --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds \
      --results-root kshot/results_modern_baselines ;;
  1)
    python scripts/run_modern_baselines.py --backbone convnextv2_tiny --method both \
      --dataset deepweeds_no_bg_mergeval --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
      --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds \
      --results-root kshot/results_modern_baselines ;;
  2)
    python scripts/run_modern_backbone_experiment.py ${COMMON_FUS} --combine softmax_ensemble \
      --results-root kshot/results_dino_pairs_mlp/dino_cnx_softmax_ensemble ;;
  3)
    python scripts/run_modern_backbone_experiment.py ${COMMON_FUS} --combine mean_evidence \
      --results-root kshot/results_dino_pairs_mlp/dino_cnx_mean_evidence ;;
esac
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} $(date)"
