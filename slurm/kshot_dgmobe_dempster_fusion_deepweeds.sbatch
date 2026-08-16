#!/bin/bash
#SBATCH --job-name=kshot-dgmobe-dempster-dw
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

# TMC-style Dempster-Shafer Dirichlet fusion (Han et al., ICLR 2021):
# per-expert Dirichlet opinions (+ fused head as extra view) combined with the
# reduced Dempster rule. No parameterized router - each expert is discounted
# per sample by its own evidential uncertainty. Loss = expected-CE + annealed
# wrong-evidence KL on the combined and per-view Dirichlets (pure mode), or
# plain CE on the fused head with the evidential block as auxiliary (anchored).
status=0
for variant in mean_ev_kl1_nofv mean_ev_kl0p1_fv pure_kl0p1_nofv pure_kl1_fv pure_kl0p1_fv pure_kl1_nofv anchored_edl0p5_kl1 mean_ev_kl1_fv; do
  case "${variant}" in
    pure_kl1_fv)         rule=dempster;      loss_mode=pure;        kl=1.0; edl=1.0; fv=true  ;;
    pure_kl0p1_fv)       rule=dempster;      loss_mode=pure;        kl=0.1; edl=1.0; fv=true  ;;
    pure_kl1_nofv)       rule=dempster;      loss_mode=pure;        kl=1.0; edl=1.0; fv=false ;;
    pure_kl0p1_nofv)     rule=dempster;      loss_mode=pure;        kl=0.1; edl=1.0; fv=false ;;
    anchored_edl0p5_kl1) rule=dempster;      loss_mode=ce_anchored; kl=1.0; edl=0.5; fv=true  ;;
    mean_ev_kl1_fv)      rule=mean_evidence; loss_mode=pure;        kl=1.0; edl=1.0; fv=true  ;;
    mean_ev_kl0p1_fv)    rule=mean_evidence; loss_mode=pure;        kl=0.1; edl=1.0; fv=true  ;;
    mean_ev_kl1_nofv)    rule=mean_evidence; loss_mode=pure;        kl=1.0; edl=1.0; fv=false ;;
    *) echo "Unknown variant: ${variant}" >&2; exit 1 ;;
  esac

  echo "========================================"
  echo "DG-MoBE Dempster fusion DeepWeeds variant=${variant} (rule=${rule}, loss=${loss_mode}, kl=${kl}, edl=${edl}, fused_view=${fv})"
  echo "Started: $(date)"

  if ! python scripts/sweep_kshot.py -m \
    stage_a_source=img100_dirichlet_imagenet_1_acc85 \
    dataset=deepweeds_no_bg_mergeval \
    kshot_dataset_key=deepweeds_no_bg_mergeval \
    'kshot_shot=1,5,10,20' \
    'kshot_run_idx=range(1,21)' \
    kshot_expected_runs=20 \
    kshot_results_root=kshot/results_dgmobe_dempster_fusion_deepweeds/${variant} \
    skip_existing_kshot_results=true \
    pipeline.added_postfix=dgmobe_dempster_${variant}_headdiv \
    pipeline.use_dirichlet_logit_moe=true \
    pipeline.dirichlet_logit_moe_combine_mode=dempster \
    pipeline.dirichlet_logit_moe_router=uniform \
    pipeline.dirichlet_logit_moe_dempster_rule="${rule}" \
    pipeline.dirichlet_logit_moe_dempster_loss_mode="${loss_mode}" \
    pipeline.dirichlet_logit_moe_dempster_kl_lambda="${kl}" \
    pipeline.dirichlet_logit_moe_dempster_edl_lambda="${edl}" \
    pipeline.dirichlet_logit_moe_dempster_fused_view="${fv}" \
    pipeline.dirichlet_logit_moe_warmup_epochs=10 \
    pipeline.attention_loss_mode=head_diversity \
    pipeline.attention_loss_warmup_epochs=3 \
    pipeline.attention_head_diversity_lambda=0.05 \
    pipeline.enable_uncertainty_metrics=true \
    pipeline.enable_diagnostic_plots=false \
    pipeline.use_wandb=false \
    export_kshot_predictions=false; then
    echo "FAILED variant=${variant}: $(date)" >&2
    status=1
  else
    echo "Finished variant=${variant}: $(date)"
  fi
done

echo "DG-MoBE Dempster fusion DeepWeeds sweep finished at: $(date)"
exit "${status}"
