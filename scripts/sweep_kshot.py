#!/usr/bin/env python3
"""Hydra entry point for k-shot multirun fusion training.

Replaces the manual ``for k_shot in shots: for run_idx in repeats:`` loop
that lived in the deleted ``train_stage_d_kshot_selective.py``. Each multirun
job picks one (shot, run_idx) combination; ``conf/sweep_kshot.yaml``
interpolates ``pipeline.kshot_manifest_path`` and ``pipeline.run_tag`` from
the sweep dimensions so each job is fully determined. Results are mirrored to
the legacy repo-local layout ``kshot/results/<stage_a_source>/<dataset_key>/``.

Usage:
    # 1. Build the manifests once for your dataset under ./kshot/manifests:
    python scripts/build_kshot_manifests.py \\
        dataset_key=plantvillage_color_split \\
        dataset_type=imagefolder \\
        imagefolder_root=/home/magz94/ddcomb/data/plantvillage_color_split

    # 2. Sweep training (4 shots × 20 runs = 80 jobs):
    python scripts/sweep_kshot.py -m \\
        stage_a_source=img100_dirichlet_imagenet_1_acc85 \\
        dataset=plantvillage_color_split \\
        kshot_dataset_key=plantvillage_color_split \\
        kshot_shot=1,5,10,20 \\
        kshot_run_idx=range(1,21)
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from ddcomb.config import KShotSweepConfig, register_configs
from ddcomb.kshot_results import (
    flatten_one_metrics,
    metrics_from_legacy_row,
    normalize_fusion_repo_paths,
    resolve_repo_path,
    update_legacy_shot_summary,
    write_json_atomic,
)
from ddcomb.pipeline import run_experiment


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _hydra_choice(name: str) -> str | None:
    choices = HydraConfig.get().runtime.choices
    value = choices.get(name)
    return None if value in (None, "null") else str(value)


def _legacy_source_slug(stage_a_source: str, frozen_fusion_tag: str | None) -> str:
    source_slugs = {
        "img100_dirichlet_imagenet_1_acc85": "attn_stage_a_dirichlet_imagenet_1_acc85",
        "img100_dirichlet_prefusion_uncertainty_acc85": (
            "attn_stage_a_dirichlet_linear_prefusion_uncertainty_"
            "imagenet100_acc85_resnet_vgg_densenet_effnet_vit"
        ),
        "img100_stagea_softmax_linear_resnet_order_v1": (
            "attn_stage_a_softmax_linear_imagenet100_acc85_resnetOrder_v1"
        ),
    }
    return source_slugs.get(stage_a_source, frozen_fusion_tag or stage_a_source)


def _legacy_run_tag(cfg: KShotSweepConfig, source_slug: str) -> str:
    return (
        f"attn_stage_d_kshot_{cfg.kshot_shot}shot_run{cfg.kshot_run_idx}_"
        f"{cfg.pipeline.attention_method}_{cfg.pipeline.classifier_method}_"
        f"{cfg.kshot_dataset_key}_from_{source_slug}"
    )


@hydra.main(config_path="../conf", config_name="sweep_kshot", version_base=None)
def main(cfg) -> None:
    typed_cfg: KShotSweepConfig = OmegaConf.to_object(cfg)
    repo_root = _repo_root()
    stage_a_source = _hydra_choice("stage_a_source") or typed_cfg.frozen_fusion_tag or "unknown_stage_a_source"
    source_slug = _legacy_source_slug(stage_a_source, typed_cfg.frozen_fusion_tag)
    typed_cfg.pipeline.run_tag = _legacy_run_tag(typed_cfg, source_slug)
    normalize_fusion_repo_paths(typed_cfg.pipeline, repo_root)
    manifest_path = resolve_repo_path(typed_cfg.pipeline.kshot_manifest_path, repo_root)
    results_root = resolve_repo_path(typed_cfg.kshot_results_root, repo_root)
    results_dir = results_root / stage_a_source / typed_cfg.kshot_dataset_key
    results_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = results_dir / f"{typed_cfg.kshot_shot}_shot_run_{typed_cfg.kshot_run_idx}.json"
    prediction_path = run_json_path.with_name(
        f"{typed_cfg.kshot_shot}_shot_run_{typed_cfg.kshot_run_idx}_predictions.npz"
    )
    if typed_cfg.export_kshot_predictions:
        typed_cfg.pipeline.prediction_output_path = str(prediction_path)
    prediction_missing = typed_cfg.export_kshot_predictions and not prediction_path.is_file()
    typed_cfg.pipeline.kshot_manifest_path = str(manifest_path)

    print("Resolved kshot sweep config:")
    print(OmegaConf.to_yaml(cfg))
    print("Repo-local paths:")
    print(f"  base_models_path: {typed_cfg.pipeline.base_models_path}")
    print(f"  model_save_path: {typed_cfg.pipeline.model_save_path}")
    print(f"  emb_cache_root: {typed_cfg.pipeline.emb_cache_root}")
    print(f"  img_plot_root: {typed_cfg.pipeline.img_plot_root}")
    print(f"[KSHOT] Using manifest: {manifest_path}")
    print(f"[KSHOT] Result JSON: {run_json_path}")

    if not run_json_path.is_file() and not manifest_path.is_file():
        raise FileNotFoundError(f"k-shot manifest does not exist: {manifest_path}")

    if run_json_path.is_file() and typed_cfg.skip_existing_kshot_results and not prediction_missing:
        row = json.loads(run_json_path.read_text())
        group_name = str(row.get("postfix", "legacy"))
        metrics = metrics_from_legacy_row(row, group_name)
        print(f"[KSHOT] Skip existing result: {run_json_path}")
    else:
        if run_json_path.is_file() and prediction_missing:
            print(f"[KSHOT] Existing result has no prediction export; rerunning to write: {prediction_path}")
        metrics = run_experiment(
            cfg=typed_cfg.pipeline,
            mode=typed_cfg.mode,
            frozen_fusion_tag=typed_cfg.frozen_fusion_tag,
            frozen_fusion_added_postfix=typed_cfg.frozen_fusion_added_postfix,
            skip_train=False,
            return_full_metrics=typed_cfg.return_full_metrics,
        )
        group_name, flat_metrics = flatten_one_metrics(metrics)
        row = {
            "dataset": typed_cfg.kshot_dataset_key,
            "stage_a_source": stage_a_source,
            "shot": typed_cfg.kshot_shot,
            "run_index": typed_cfg.kshot_run_idx,
            "manifest_path": str(manifest_path),
            "postfix": group_name,
            "run_tag": typed_cfg.pipeline.run_tag,
            "test_accuracy": float(flat_metrics["test_accuracy"]),
            "test_precision": float(flat_metrics["test_precision"]),
            "test_recall": float(flat_metrics["test_recall"]),
        }
        optional_key_map = {
            "test_ece": "test_ece",
            "test_mce": "test_mce",
            "auroc_err_pred_entropy": "auroc_err_pred_entropy",
            "fpr@95_err_pred_entropy": "fpr95_err_pred_entropy",
            "uncertainty/p_a_given_c_mean": "p_a_given_c_mean",
            "uncertainty/p_u_given_i_mean": "p_u_given_i_mean",
            "uncertainty/p_i_given_u_mean": "p_i_given_u_mean",
            "uncertainty/avu_mean": "avu_mean",
            "dgmobe/router_entropy_mean": "dgmobe_router_entropy_mean",
            "dgmobe/router_max_mean": "dgmobe_router_max_mean",
            "dgmobe/router_l1_from_uniform_mean": "dgmobe_router_l1_from_uniform_mean",
            "dgmobe/expert_logit_cos_mean": "dgmobe_expert_logit_cos_mean",
            "dgmobe/expert_logit_std_mean": "dgmobe_expert_logit_std_mean",
        }
        for source_key, row_key in optional_key_map.items():
            if source_key in flat_metrics and flat_metrics[source_key] is not None:
                row[row_key] = float(flat_metrics[source_key])
        write_json_atomic(run_json_path, row)
        print(f"[KSHOT] Wrote legacy run result: {run_json_path}")

    summary_path = update_legacy_shot_summary(
        results_dir=results_dir,
        shot=typed_cfg.kshot_shot,
        expected_runs=typed_cfg.kshot_expected_runs,
        summary_prefix={
            "dataset": typed_cfg.kshot_dataset_key,
            "stage_a_source": stage_a_source,
            "base_seed": typed_cfg.kshot_base_seed,
        },
    )
    print(f"[KSHOT] Updated legacy summary: {summary_path}")
    print(f"Metrics for {typed_cfg.kshot_shot}-shot run {typed_cfg.kshot_run_idx}:", metrics)


if __name__ == "__main__":
    register_configs()
    main()
