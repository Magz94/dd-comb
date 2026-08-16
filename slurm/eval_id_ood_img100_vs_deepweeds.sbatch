#!/bin/bash
#SBATCH --job-name=idood-img100-deepweeds
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=04:00:00
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

python -u scripts/eval_id_ood_pred_entropy.py \
  --imagenet100-root data/ImageNet100/ImageNet100 \
  --deepweeds-root /datasets/DeepWeeds \
  --base-models-path models/base \
  --base-subfolder acc85 \
  --model-save-path models \
  --batch-size 64 \
  --num-workers 5 \
  --output-csv id_ood_img100_vs_deepweeds_pred_entropy.csv
