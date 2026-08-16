#!/bin/bash
#SBATCH --job-name=kshot-fusion-final
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
export WANDB_API_KEY="$(cat ~/.secrets/wandb_api_key)"
export WANDB_ENTITY="kihug"
export PYTHONPATH=/home/magz94/ddcomb/src:${PYTHONPATH:-}

python -m pip install -r environment/requirements.txt

DATASETS=(
  deepweeds_no_bg_mergeval
  plant_seedlings_v2_split
  bloodmnist_224_imagefolder_mergeval
  aptos2019_split
)

SOURCES=(
  img100_dirichlet_imagenet_1_acc85
  img100_stagea_softmax_linear_resnet_order_v1
)

for source in "${SOURCES[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    if [ "${source}" = "img100_dirichlet_imagenet_1_acc85" ]; then
      python scripts/sweep_kshot.py -m \
        stage_a_source="${source}" \
        dataset="${dataset}" \
        kshot_dataset_key="${dataset}" \
        'kshot_shot=1,5,10,20' \
        'kshot_run_idx=range(1,21)' \
        pipeline.added_postfix=dirklwarmup1e-4_test \
        pipeline.dirichlet_kl_regularization=true \
        pipeline.dirichlet_kl_lambda=1e-4 \
        pipeline.dirichlet_kl_warmup_epochs=3 \
        export_kshot_predictions=false
    else
      python scripts/sweep_kshot.py -m \
        stage_a_source="${source}" \
        dataset="${dataset}" \
        kshot_dataset_key="${dataset}" \
        'kshot_shot=1,5,10,20' \
        'kshot_run_idx=range(1,21)' \
        export_kshot_predictions=false
    fi
  done
done
