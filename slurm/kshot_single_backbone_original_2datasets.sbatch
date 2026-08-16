#!/bin/bash
#SBATCH --job-name=kshot-single-orig
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

# Single backbones on the ORIGINAL ImageNet-1k pretrained weights (head_only,
# NO acc85 init -> stock IMAGENET1K_V1 frozen backbone + trained linear head).
# Distinct results root so it does not collide with the acc85 single-backbone
# results. Two datasets, all five backbones.

DATASETS=(deepweeds_no_bg_mergeval plant_seedlings_v2_split)

for dataset in "${DATASETS[@]}"; do
  echo "==== single-backbone ORIGINAL | ${dataset} | $(date) ===="
  python scripts/sweep_single_backbone_kshot.py -m \
    single_backbone_dataset="${dataset}" \
    model=vit_b16,densenet121,efficientnet_b0,resnet18,vgg16 \
    mode=head_only \
    init_base_models_path=null \
    init_base_subfolder=null \
    kshot_dataset_key="${dataset}" \
    kshot_results_root=kshot/results_single_backbone_original \
    'kshot_shot=1,5,10,20' \
    'kshot_run_idx=range(1,21)' \
    export_kshot_predictions=false
done
