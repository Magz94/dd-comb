#!/bin/bash
#SBATCH --job-name=prepare-ssl-patches
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
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

python scripts/prepare_ssl_patches.py \
  --source-root /datasets/SSL_Patches/Cropped_VariableSize \
  --output-root data/ssl_patches \
  --val-ratio 0.2 \
  --seed 42 \
  --mode copy \
  --overwrite

echo "Prepared SSL patches:"
find data/ssl_patches -maxdepth 2 -type d | sort
echo "File counts:"
find data/ssl_patches/train -type f,l | wc -l
find data/ssl_patches/val -type f,l | wc -l

python scripts/build_kshot_manifests.py \
  dataset_key=ssl_patches \
  dataset_type=imagefolder \
  imagefolder_root=/home/magz94/ddcomb/data/ssl_patches \
  imagefolder_train_split=train \
  shots='[1,5,10,20]' \
  repeats=20 \
  base_seed=42

echo "K-shot manifests:"
find kshot/manifests/ssl_patches -type f | sort | wc -l
