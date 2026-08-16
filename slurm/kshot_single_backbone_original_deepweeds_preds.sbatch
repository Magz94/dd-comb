#!/bin/bash
#SBATCH --job-name=kshot-single-orig-preds
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

# Single backbones on the ORIGINAL ImageNet-1k weights, DeepWeeds ONLY, WITH
# prediction export (export_kshot_predictions=true) so we can build the vanilla
# deep ensemble (average the 5 backbones' softmax). Fresh results root so it
# regenerates the *_predictions.npz the metrics-only run didn't save. Safe to
# run alongside the still-running plant_seedlings single-backbone job.
echo "==== single-backbone ORIGINAL deepweeds (with preds) | $(date) ===="
python scripts/sweep_single_backbone_kshot.py -m \
  single_backbone_dataset=deepweeds_no_bg_mergeval \
  model=vit_b16,densenet121,efficientnet_b0,resnet18,vgg16 \
  mode=head_only \
  init_base_models_path=null \
  init_base_subfolder=null \
  kshot_dataset_key=deepweeds_no_bg_mergeval \
  kshot_results_root=kshot/results_single_backbone_original_preds \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  export_kshot_predictions=true
echo "Done $(date)"
