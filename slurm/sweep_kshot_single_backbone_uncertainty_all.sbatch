#!/bin/bash
#SBATCH --job-name=kshot-sb-unc-all
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

# Keep this dependency step light; the container already has torch/torchvision.
python -m pip install -r environment/requirements.txt

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
  echo "K-shot single-backbone uncertainty sweep: ${dataset}"
  echo "Started: $(date)"

  python scripts/sweep_single_backbone_kshot.py -m \
    single_backbone_dataset="${dataset}" \
    model=vit_b16,densenet121,efficientnet_b0,resnet18,vgg16 \
    mode=head_only \
    preprocess=basic \
    init_base_models_path=models/base \
    init_base_subfolder=acc85 \
    kshot_dataset_key="${dataset}" \
    'kshot_shot=1,5,10,20' \
    'kshot_run_idx=range(1,21)' \
    kshot_results_root=kshot/results_single_backbone_uncertainty \
    kshot_expected_runs=20 \
    skip_existing_kshot_results=true \
    enable_uncertainty_metrics=true \
    export_kshot_predictions=false \
    use_wandb=false

  echo "Finished: $(date)"
done

echo "All k-shot single-backbone uncertainty sweeps finished at: $(date)"
