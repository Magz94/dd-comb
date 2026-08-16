#!/bin/bash
#SBATCH --job-name=beat-dino-large-dw
#SBATCH --container-image='/home/magz94/containers/elan_container.sqfs'
#SBATCH --container-writable
#SBATCH --no-container-remap-root
#SBATCH --gpus=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80G
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

# Goal: a backbone combination that BEATS single DINOv2 (0.866 @20-shot on DeepWeeds).
# DINOv2 (base, ONE variant) stays in the set; we add strong, diverse non-DINO large
# partners so the fusion starts from 0.866 and only needs them to NET-ADD. Partners
# chosen for strength + paradigm diversity: EVA-02-L (MIM+CLIP), ConvNeXt-V2-L (conv),
# SigLIP2-L (VLM). Dempster rule, linear_ln head, native width.
TRIO="dinov2_vitb14,eva02_large,convnextv2_large"
QUAD="dinov2_vitb14,eva02_large,convnextv2_large,siglip2_large"
COMMON="--dataset deepweeds_no_bg_mergeval --combine dempster --head linear_ln --width native \
        --shots 1,5,10,20 --runs 1-20 --expected-runs 20 \
        --cache-root embedding_cache_modern_large --deepweeds-root /datasets/DeepWeeds"

status=0

# (1) Single backbones (export preds -> gain/risk vs DINOv2 + individual accuracy).
#     DINOv2 base single preds already exist (kshot/results_modern_singles_preds).
for BB in eva02_large convnextv2_large siglip2_large; do
  echo "==== single ${BB} $(date) ===="
  if ! python scripts/run_modern_backbone_experiment.py ${COMMON} \
       --backbones "${BB}" --combine softmax_ensemble --export-preds \
       --results-root kshot/results_large_singles/${BB}; then
    echo "FAILED single ${BB}" >&2; status=1
  fi
done

# (2) Dempster fusion: DINOv2 + 2 diverse large partners, then + SigLIP2-L (4-way).
echo "==== fusion ${TRIO} dempster $(date) ===="
if ! python scripts/run_modern_backbone_experiment.py ${COMMON} \
     --backbones "${TRIO}" \
     --results-root kshot/results_large_combos/dino_eva_cnxL_dempster_linear_ln; then
  echo "FAILED fusion trio" >&2; status=1
fi

echo "==== fusion ${QUAD} dempster $(date) ===="
if ! python scripts/run_modern_backbone_experiment.py ${COMMON} \
     --backbones "${QUAD}" \
     --results-root kshot/results_large_combos/dino_eva_cnxL_siglipL_dempster_linear_ln; then
  echo "FAILED fusion quad" >&2; status=1
fi

echo "beat-dinov2 large fusion finished $(date)"
exit "${status}"
