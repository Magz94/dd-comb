#!/bin/bash
#SBATCH --job-name=kshot-base-unc-all
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
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

python -m pip install -r environment/requirements.txt

# Baseline fusion = softmax attention + linear classifier stage-A source.
# Results are written to a separate root so old baseline k-shot outputs stay untouched.
DATASETS=(
  deepweeds_no_bg_mergeval
  plant_seedlings_v2_split
  bloodmnist_224_imagefolder_mergeval
  aptos2019_split
  leafsugar_disease_split
  sar_cld_original_split
)

for dataset in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Baseline fusion uncertainty sweep: ${dataset}"
  echo "Started: $(date)"

  python scripts/sweep_kshot.py -m \
    stage_a_source=img100_stagea_softmax_linear_resnet_order_v1 \
    dataset="${dataset}" \
    kshot_dataset_key="${dataset}" \
    'kshot_shot=1,5,10,20' \
    'kshot_run_idx=range(1,21)' \
    kshot_expected_runs=20 \
    kshot_results_root=kshot/results_baseline_fusion_uncertainty \
    skip_existing_kshot_results=true \
    pipeline.enable_uncertainty_metrics=true \
    pipeline.enable_diagnostic_plots=false \
    pipeline.use_wandb=false \
    export_kshot_predictions=false

  echo "Finished: $(date)"
done

echo "All baseline fusion uncertainty sweeps finished at: $(date)"
