#!/bin/bash
#SBATCH --job-name=calib-exports-psv2
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-5
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
# Plant Seedlings V2 (ImageFolder train/test; NO val -> calib carved from train).
# Same 8-model protocol as the DeepWeeds calibration exports, mlp head / native
# widths, k in {1,5,10,20}, 20 runs, with --export-logits so every method saves
# per-(shot,run) TEST + CALIB logits (+ evidential alpha for Dempster & Sensoy)
# for post-hoc MCE / temperature scaling / vacuity. Sensoy uses its standard
# LINEAR head (no --hidden). Results land under kshot/results_calib/<model>/
# plant_seedlings_v2_split/, disjoint from the DeepWeeds subdirs.
# ---------------------------------------------------------------------------
DS="plant_seedlings_v2_split"
IMROOT="data/plant_seedlings_v2_split"
FUS="--dataset ${DS} --imagefolder-root ${IMROOT} --head mlp --width native \
     --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
     --cache-root embedding_cache_modern --export-logits"
BASE="--dataset ${DS} --imagefolder-root ${IMROOT} --shots 1,5,10,20 --runs 1-20 \
      --expected-runs 20 --cache-root embedding_cache_modern --export-logits"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14 \
        --combine softmax_ensemble --results-root kshot/results_calib/dinov2_standard ;;
  1) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones convnextv2_tiny \
        --combine softmax_ensemble --results-root kshot/results_calib/convnextv2_standard ;;
  2) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14,convnextv2_tiny \
        --combine dempster --results-root kshot/results_calib/dino_cnx_dempster ;;
  3) python scripts/run_vanilla_deep_ensemble.py ${BASE} --backbones dinov2_vitb14,convnextv2_tiny \
        --results-root kshot/results_calib/vanilla_ens ;;
  4) python scripts/run_modern_baselines.py ${BASE} --backbone dinov2_vitb14 --method both \
        --results-root kshot/results_calib/baselines ;;
  5) python scripts/run_modern_baselines.py ${BASE} --backbone convnextv2_tiny --method both \
        --results-root kshot/results_calib/baselines ;;
esac
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} $(date)"
