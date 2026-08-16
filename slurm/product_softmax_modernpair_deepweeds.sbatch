#!/bin/bash
#SBATCH --job-name=prodsoftmax-dw
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

# ---------------------------------------------------------------------------
# Product-of-softmax combine on the modern pair (POINT 3).
# Non-evidential *product* pool: p_hat ~ prod_m softmax(l_m), renormalised.
# It fills the missing cell of the 2x2 {linear|product} x {non-evidential|evidential},
# isolating "product vs. average pooling" from the Dempster/evidential machinery.
# Same protocol as the Dempster modern-pair run (mlp head, native width, k in
# {1,5,10,20}, 20 runs, --export-logits for TS / entropy scoring).
# ---------------------------------------------------------------------------
python scripts/run_modern_backbone_experiment.py \
  --dataset deepweeds_no_bg_mergeval \
  --backbones dinov2_vitb14,convnextv2_tiny --combine product_softmax \
  --head mlp --width native --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
  --export-logits --cache-root embedding_cache_modern --deepweeds-root /datasets/DeepWeeds \
  --results-root kshot/results_calib/product_softmax_dino2_cnx
echo "Done $(date)"
