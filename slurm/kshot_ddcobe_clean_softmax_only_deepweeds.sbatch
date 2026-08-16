#!/bin/bash
#SBATCH --job-name=kshot-ddcobe-softmax-dw
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

# Non-evidential floor for the clean DD-CoBE architecture: the same per-backbone
# MLP heads (jointly trained) combined by averaging their softmax probabilities.
# Writes to its own results folder, so it is SAFE to run in parallel with the
# evidential-arm job (kshot_ddcobe_clean_deepweeds.sbatch) that is still running.
echo "Clean DD-CoBE softmax-ensemble floor (DeepWeeds) started: $(date)"
python scripts/sweep_kshot_ddcobe.py -m \
  stage_a_source=img100_dirichlet_imagenet_1_acc85 \
  dataset=deepweeds_no_bg_mergeval \
  kshot_dataset_key=deepweeds_no_bg_mergeval \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_expected_runs=20 \
  kshot_results_root=kshot/results_ddcobe_clean_deepweeds/softmax_ensemble \
  skip_existing_kshot_results=true \
  pipeline.ddcobe_combine=softmax_ensemble \
  pipeline.ddcobe_kl_lambda=0.0 \
  pipeline.ddcobe_kl_warmup_epochs=10 \
  pipeline.use_wandb=false

echo "Clean DD-CoBE softmax-ensemble floor finished: $(date)"
