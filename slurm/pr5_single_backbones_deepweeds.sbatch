#!/bin/bash
#SBATCH --job-name=pr5-sb-deepweeds
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

export WANDB_API_KEY="$(cat ~/.secrets/wandb_api_key)"
export WANDB_ENTITY="kihug"

cd /home/magz94/ddcomb
export PYTHONPATH=/home/magz94/ddcomb/src:${PYTHONPATH:-}
python -m pip install -r environment/requirements.txt

python scripts/sweep_single_backbone_kshot.py -m \
  single_backbone_dataset=deepweeds_no_bg_mergeval \
  model=vit_b16,densenet121,efficientnet_b0,resnet18,vgg16 \
  mode=head_only \
  init_base_models_path=models/base \
  init_base_subfolder=acc85 \
  kshot_dataset_key=deepweeds_no_bg_mergeval \
  kshot_shot=5 \
  'kshot_run_idx=range(1,21)' \
  export_kshot_predictions=true \
  kshot_results_root=results_single_backbone
