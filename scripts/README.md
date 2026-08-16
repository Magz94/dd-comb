# Scripts

The scripts are kept flat in this directory on purpose: the Hydra entry points
resolve their config directory as `config_path="../conf"` and the repository root
as `Path(__file__).resolve().parents[1]`, and the SLURM jobs call them as
`python scripts/<name>.py`. Moving them into subfolders would break all three, so
they are grouped here by documentation rather than by directory.

Most experiment entry points are also reachable through the unified CLI, e.g.
`ddcomb train-fusion ...` (see `ddcomb --help`).

## Data preparation
Build the image-folder / split layout each dataset config expects.

- `prepare_bloodmnist_224.py`, `prepare_plant_seedlings_v2.py`,
  `prepare_ssl_patches.py`, `prepare_cassava_leaf_disease.py`,
  `prepare_leafsugar_disease.py`, `prepare_sar_cld_original.py`

## Experiments (Hydra entry points)
Training, sweeps, evaluation and manifest building. These take Hydra overrides.

- `build_kshot_manifests.py` — build the few-shot manifests under `kshot/manifests/`
- `train_fusion.py`, `train_fusion_full_to_json.py`, `train_single_backbone.py`
- `sweep_kshot.py`, `sweep_kshot_baselines.py`, `sweep_kshot_ddcobe.py`,
  `sweep_single_backbone_kshot.py`
- `run_modern_backbone_experiment.py`, `run_modern_baselines.py`,
  `run_vanilla_deep_ensemble.py`
- `eval.py`, `eval_id_ood_pred_entropy.py`, `backbone_characterization.py`

## Analysis, tables and figures (post-hoc)
Consume the result directories and emit the thesis tables/figures. Run after the
experiments.

- Tables/reports: `make_four_dataset_table.py`, `build_calibration_table.py`,
  `build_ddcobe_report_html.py`, `build_dinov2_table.py`,
  `build_deep_ensemble_results.py`, `build_dgmobe_router_ablation_html.py`,
  `build_dgmobe_residual_ablation_html.py`, `add_ddcobe_to_comparison_tables.py`,
  `add_ece_to_comparison_tables.py`
- Metrics: `compute_nll_brier.py`, `compute_combiner_metrics.py`,
  `compute_kshot5_accuracy_ci.py`, `make_selective_prediction.py`
- Figures: `make_thesis_curves.py`, `make_thesis_heatmaps.py`,
  `make_calibration_bars.py`, `make_calibration_heatmaps.py`,
  `output_curves_macro.py`, `output_curves_macro_legacy_all_datasets.py`,
  `plot_pr_curves_kshot10.py`, `plot_pr_curves_kshot5_macro.py`,
  `build_latent_umap_figs.py`, `backbone_study_analysis.py`,
  `mc_dropout_latency.py`
