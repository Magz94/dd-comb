#!/bin/bash
#SBATCH --job-name=kshot-ddcobe-clean-dw
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

# Clean, attention-free DD-CoBE ensemble (models/ddcobe_ensemble.py): each frozen
# backbone has its own complete head (Linear-ReLU-Linear-LayerNorm-Linear, with
# an independent LayerNorm), turned into a Dirichlet opinion; the opinions are
# combined by the reduced Dempster rule (DD-CoBE) or by evidence averaging
# (mean-evidence control). No attention, no router, no fused view, no head-div.
status=0
for variant in softmax_ensemble dempster_kl0p1 mean_evidence_kl1p0 mean_evidence_kl0p1; do
  case "${variant}" in
    softmax_ensemble)    combine=softmax_ensemble; kl=0.0 ;;
    dempster_kl0p1)      combine=dempster;         kl=0.1 ;;
    mean_evidence_kl1p0) combine=mean_evidence;    kl=1.0 ;;
    mean_evidence_kl0p1) combine=mean_evidence;    kl=0.1 ;;
    *) echo "Unknown variant: ${variant}" >&2; exit 1 ;;
  esac

  echo "========================================"
  echo "Clean DD-CoBE DeepWeeds variant=${variant} (combine=${combine}, kl=${kl})"
  echo "Started: $(date)"

  if ! python scripts/sweep_kshot_ddcobe.py -m \
    stage_a_source=img100_dirichlet_imagenet_1_acc85 \
    dataset=deepweeds_no_bg_mergeval \
    kshot_dataset_key=deepweeds_no_bg_mergeval \
    'kshot_shot=1,5,10,20' \
    'kshot_run_idx=range(1,21)' \
    kshot_expected_runs=20 \
    kshot_results_root=kshot/results_ddcobe_clean_deepweeds/${variant} \
    skip_existing_kshot_results=true \
    pipeline.ddcobe_combine=${combine} \
    pipeline.ddcobe_kl_lambda=${kl} \
    pipeline.ddcobe_kl_warmup_epochs=10 \
    pipeline.use_wandb=false; then
    echo "FAILED variant=${variant}: $(date)" >&2
    status=1
  else
    echo "Finished variant=${variant}: $(date)"
  fi
done

echo "Clean DD-CoBE DeepWeeds sweep finished at: $(date)"
exit "${status}"
