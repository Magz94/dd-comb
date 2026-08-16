#!/bin/bash
#SBATCH --job-name=edl-lossctrl-dw
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
# Head-matched single-backbone EDL (POINT 1: the loss control).
# One backbone (K=1) through DDCoBEEnsemble = the SHARED MLP head + softplus
# evidence, reported under BOTH evidential losses:
#   --loss-mode expected_ce  -> DD-CoMB objective  (== DD-CoMB at K=1)
#   --loss-mode brier        -> Sensoy Eq.(5) Bayes-risk (the original recipe)
# Only the loss differs, so this isolates it: if both collapse, the collapse is
# the single-backbone evidential setting, not the loss.
# ---------------------------------------------------------------------------
COMMON="--dataset deepweeds_no_bg_mergeval --combine dempster --head mlp --width native \
        --shots 1,5,10,20 --runs 1-20 --expected-runs 20 --export-logits \
        --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) python scripts/run_modern_backbone_experiment.py ${COMMON} --backbones dinov2_vitb14 \
        --loss-mode expected_ce --results-root kshot/results_calib/edl_headmatched_dinov2_ce ;;
  1) python scripts/run_modern_backbone_experiment.py ${COMMON} --backbones dinov2_vitb14 \
        --loss-mode brier       --results-root kshot/results_calib/edl_headmatched_dinov2_brier ;;
  2) python scripts/run_modern_backbone_experiment.py ${COMMON} --backbones convnextv2_tiny \
        --loss-mode expected_ce --results-root kshot/results_calib/edl_headmatched_cnx_ce ;;
  3) python scripts/run_modern_backbone_experiment.py ${COMMON} --backbones convnextv2_tiny \
        --loss-mode brier       --results-root kshot/results_calib/edl_headmatched_cnx_brier ;;
esac
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} $(date)"
