#!/bin/bash
#SBATCH --job-name=sb-full-unc
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

MODELS=(
  vit_b16
  densenet121
  efficientnet_b0
  resnet18
  vgg16
)

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Dataset: ${dataset}"
    echo "Model:   ${model}"
    echo "Started: $(date)"

    python scripts/train_single_backbone.py \
      single_backbone_dataset="${dataset}" \
      model="${model}" \
      mode=head_only \
      preprocess=basic \
      init_base_models_path=models/base \
      init_base_subfolder=acc85 \
      output_dir=models/single_backbone_uncertainty_full \
      run_name="${dataset}_${model}_base_acc85_head_only_uncertainty" \
      enable_uncertainty_metrics=true \
      use_wandb=false

    echo "Finished: $(date)"
  done
done

echo "All single-backbone uncertainty runs finished at: $(date)"
