#!/bin/bash
#SBATCH --job-name=kshot-sb-export-dw
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

# Re-export per-sample test probabilities for the five DeepWeeds single
# backbones (the deep-ensemble members). The seeds are deterministic per
# (shot, run_idx), so the re-run reproduces the exact same heads/predictions
# as the existing results_single_backbone_uncertainty JSONs; with
# export_kshot_predictions=true it additionally writes the *_predictions.npz
# files the ensemble averaging script needs. skip_existing leaves the JSONs
# intact and only adds the missing prediction exports.
python scripts/sweep_single_backbone_kshot.py -m \
  single_backbone_dataset=deepweeds_no_bg_mergeval \
  model=vit_b16,densenet121,efficientnet_b0,resnet18,vgg16 \
  mode=head_only \
  preprocess=basic \
  init_base_models_path=models/base \
  init_base_subfolder=acc85 \
  kshot_dataset_key=deepweeds_no_bg_mergeval \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_results_root=kshot/results_single_backbone_uncertainty \
  kshot_expected_runs=20 \
  skip_existing_kshot_results=true \
  enable_uncertainty_metrics=true \
  export_kshot_predictions=true \
  use_wandb=false

echo "DeepWeeds single-backbone prediction export finished at: $(date)"
