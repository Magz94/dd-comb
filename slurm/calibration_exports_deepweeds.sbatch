#!/bin/bash
#SBATCH --job-name=calib-exports-dw
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-7
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

# Re-run every method with --export-logits into FRESH dirs (kshot/results_calib/*),
# saving per-(shot,run) TEST + CALIB(val) logits (+ evidential alpha) for post-hoc
# temperature scaling and vacuity-based scoring. mlp head, native widths throughout.
FUS="--dataset deepweeds_no_bg_mergeval --head mlp --width native --shots 1,5,10,20 --runs 1-20 \
     --expected-runs 20 --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds --export-logits"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14,convnextv2_tiny \
        --combine dempster       --results-root kshot/results_calib/dino_cnx_dempster ;;
  1) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14,convnextv2_tiny \
        --combine mean_evidence  --results-root kshot/results_calib/dino_cnx_mean_evidence ;;
  2) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14,convnextv2_tiny \
        --combine softmax_ensemble --results-root kshot/results_calib/dino_cnx_softmax_ensemble ;;
  3) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones dinov2_vitb14 \
        --combine softmax_ensemble --results-root kshot/results_calib/dinov2_standard ;;
  4) python scripts/run_modern_backbone_experiment.py ${FUS} --backbones convnextv2_tiny \
        --combine softmax_ensemble --results-root kshot/results_calib/convnextv2_standard ;;
  5) python scripts/run_vanilla_deep_ensemble.py --backbones dinov2_vitb14,convnextv2_tiny \
        --dataset deepweeds_no_bg_mergeval --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
        --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds --export-logits \
        --results-root kshot/results_calib/vanilla_ens ;;
  6) python scripts/run_modern_baselines.py --backbone dinov2_vitb14 --method both \
        --dataset deepweeds_no_bg_mergeval --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
        --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds --export-logits \
        --results-root kshot/results_calib/baselines ;;
  7) python scripts/run_modern_baselines.py --backbone convnextv2_tiny --method both \
        --dataset deepweeds_no_bg_mergeval --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
        --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds --export-logits \
        --results-root kshot/results_calib/baselines ;;
esac
echo "Done task ${SLURM_ARRAY_TASK_ID:-0} $(date)"
