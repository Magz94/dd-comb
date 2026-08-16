#!/bin/bash
#SBATCH --job-name=build-ssl-kshot
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
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

if [ ! -d data/ssl_patches/train ]; then
  echo "Missing data/ssl_patches/train. Run slurm/prepare_ssl_patches.sbatch first."
  exit 1
fi

python scripts/build_kshot_manifests.py \
  dataset_key=ssl_patches \
  dataset_type=imagefolder \
  imagefolder_root=/home/magz94/ddcomb/data/ssl_patches \
  imagefolder_train_split=train \
  'shots=[1,5,10,20]' \
  repeats=20 \
  base_seed=42

echo "Manifest count:"
find kshot/manifests/ssl_patches -type f | sort | wc -l

echo "Manifest preview:"
find kshot/manifests/ssl_patches -type f | sort | head -20
