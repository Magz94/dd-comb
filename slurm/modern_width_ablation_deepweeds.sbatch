#!/bin/bash
#SBATCH --job-name=modern-width
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60G
#SBATCH --time=20-00:00:00
#SBATCH --array=0-2
#SBATCH --output=/home/magz94/logs/slurm/sb_%A_%a.out
#SBATCH --error=/home/magz94/logs/slurm/sb_%A_%a.err
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

# Head + width ablation on DeepWeeds: 3-backbone DD-CoBE (Dempster).
# Single-factor arms (each changes ONE thing):
#   0  mlp  common  - MLP head (adapter+LN), common-min hidden width  (the default)
#   1  mlp  native  - MLP head, each backbone's native hidden width
#   2  linear_ln    - linear probe + input LayerNorm, no hidden adapter
#                     (width is irrelevant for this head -> a single run)
HEADS=(mlp mlp linear_ln)
WIDTHS=(common native common)
LABELS=(mlp_common mlp_native linear_ln)
I="${SLURM_ARRAY_TASK_ID:-0}"
HEAD="${HEADS[$I]}"; WIDTH="${WIDTHS[$I]}"; LABEL="${LABELS[$I]}"

echo "Modern head/width ablation | DeepWeeds | ${LABEL} (head=${HEAD}, width=${WIDTH}) | $(date)"
python scripts/run_modern_backbone_experiment.py \
  --dataset deepweeds_no_bg_mergeval \
  --backbones dinov2_vitb14,openclip_vitb16,convnextv2_tiny \
  --combine dempster --kl 0.1 --width "${WIDTH}" --head "${HEAD}" \
  --shots 1,5,10,20 --runs 1-20 \
  --results-root "kshot/results_modern_headwidth/${LABEL}"
echo "Done ${LABEL} $(date)"
