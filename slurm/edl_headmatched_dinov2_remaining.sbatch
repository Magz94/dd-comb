#!/bin/bash
#SBATCH --job-name=edl-hm-multi
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

# ---------------------------------------------------------------------------
# Head-matched single-backbone EDL (DINOv2, shared MLP head + softplus), the
# remaining datasets to complete the full-evaluation baseline + the BloodMNIST
# collapse control:
#   tasks 0-2 : Brier      loss on Plant Seedlings V2 / SSL-Patches / BloodMNIST
#   task  3   : expected-CE loss on BloodMNIST (CE-vs-Brier collapse control)
# Written to results_calib/edl_headmatched_dinov2_{brier,ce}/<DS>/ (same root as the
# DeepWeeds head-matched runs); make_four_dataset_table.py's base_root special-cases
# "edl_headmatched" -> results_calib for all datasets, and results_calib is in every
# dataset's LOGIT_ROOTS, so both JSON and npz resolve there.
# Private per-task embedding cache avoids concurrent-write races on BloodMNIST.
# ---------------------------------------------------------------------------
RR="kshot/results_calib"
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) DS=plant_seedlings_v2_split;            IMROOT=data/plant_seedlings_v2_split;   LM=brier;       OUT=edl_headmatched_dinov2_brier ;;
  1) DS=ssl_patches;                         IMROOT=data/ssl_patches;                LM=brier;       OUT=edl_headmatched_dinov2_brier ;;
  2) DS=bloodmnist_224_imagefolder_mergeval; IMROOT=data/bloodmnist_224_imagefolder; LM=brier;       OUT=edl_headmatched_dinov2_brier ;;
  3) DS=bloodmnist_224_imagefolder_mergeval; IMROOT=data/bloodmnist_224_imagefolder; LM=expected_ce; OUT=edl_headmatched_dinov2_ce ;;
esac
CACHE="embedding_cache_headmatched/t${SLURM_ARRAY_TASK_ID:-0}"

python scripts/run_modern_backbone_experiment.py \
  --dataset "${DS}" --imagefolder-root "${IMROOT}" \
  --backbones dinov2_vitb14 --combine dempster --loss-mode "${LM}" \
  --head mlp --width native --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
  --export-logits --cache-root "${CACHE}" \
  --results-root "${RR}/${OUT}"
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} ${DS} ${LM} $(date)"
