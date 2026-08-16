#!/bin/bash
#SBATCH --job-name=kshot-ssl-sb-a
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
export HYDRA_FULL_ERROR=1

python -m pip install -r environment/requirements.txt

if [ ! -f kshot/manifests/ssl_patches/1_shot_run_1.json ]; then
  echo "Missing k-shot manifests. Run: sbatch slurm/prepare_ssl_patches.sbatch"
  exit 1
fi

python scripts/sweep_single_backbone_kshot.py -m \
  single_backbone_dataset=ssl_patches \
  model=vit_b16,densenet121 \
  mode=head_only \
  preprocess=basic \
  init_base_models_path=models/base \
  init_base_subfolder=acc85 \
  kshot_dataset_key=ssl_patches \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_results_root=kshot/results_single_backbone_uncertainty \
  kshot_expected_runs=20 \
  skip_existing_kshot_results=true \
  enable_uncertainty_metrics=true \
  export_kshot_predictions=false \
  use_wandb=false

echo "SSL_Patches single-backbone group A finished at: $(date)"
