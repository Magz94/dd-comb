#!/bin/bash
#SBATCH --job-name=kshot-headpreds-orig
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

# DD-CoBE (Dempster) on the five ORIGINAL backbones, DeepWeeds, exporting the
# per-backbone-HEAD test predictions (*_headpreds.npz) so we can run the
# backbone-diversity analysis INSIDE the ensemble (B): oracle / leave-one-out /
# pairwise diversity on the jointly-trained heads. Fresh results root.
echo "DD-CoBE head-pred export (ORIGINAL backbones) | DeepWeeds | $(date)"
python scripts/sweep_kshot_ddcobe.py -m \
  stage_a_source=imagenet1k_original_5bb \
  dataset=deepweeds_no_bg_mergeval \
  kshot_dataset_key=deepweeds_no_bg_mergeval \
  'kshot_shot=1,5,10,20' \
  'kshot_run_idx=range(1,21)' \
  kshot_expected_runs=20 \
  kshot_results_root=kshot/results_ddcobe_headpreds_original \
  skip_existing_kshot_results=true \
  export_kshot_predictions=true \
  pipeline.ddcobe_combine=dempster \
  pipeline.ddcobe_kl_lambda=0.1 \
  pipeline.ddcobe_kl_warmup_epochs=10 \
  pipeline.ddcobe_head_type=mlp \
  pipeline.use_wandb=false

echo "Done head-pred export $(date)"
