#!/usr/bin/env python3
"""K-shot multirun sweep for the standard single-backbone baselines.

Mirrors ``scripts/sweep_kshot_ddcobe.py`` (same manifest / results / summary
plumbing) but trains a single-backbone baseline from
``ddcomb.models.baselines_runtime`` on ONE selected backbone:

* ``pipeline.baseline_method=sensoy``      -> Sensoy EDL (NeurIPS 2018)
* ``pipeline.baseline_method=mc_dropout``  -> MC Dropout (ICML 2016)

Select the backbone with ``pipeline.baseline_backbone`` (canonical names:
ResNet18 | VGG16 | DenseNet121 | EfficientNetB0 | ViT-B16).

Example (Sensoy EDL, ResNet18, DeepWeeds):
    python scripts/sweep_kshot_baselines.py -m \\
        stage_a_source=img100_dirichlet_imagenet_1_acc85 \\
        dataset=deepweeds_no_bg_mergeval kshot_dataset_key=deepweeds_no_bg_mergeval \\
        kshot_shot=1,5,10,20 kshot_run_idx=range(1,21) \\
        kshot_results_root=kshot/results_baseline_sensoy/ResNet18 \\
        pipeline.baseline_method=sensoy pipeline.baseline_backbone=ResNet18
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
    normalize_fusion_repo_paths,
    resolve_repo_path,
    update_legacy_shot_summary,
    write_json_atomic,
)
from ddcomb.models.baselines_runtime import run_single_backbone_baseline_experiment


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _hydra_choice(name: str) -> str | None:
    choices = HydraConfig.get().runtime.choices
    value = choices.get(name)
    return None if value in (None, "null") else str(value)


@hydra.main(config_path="../conf", config_name="sweep_kshot", version_base=None)
def main(cfg) -> None:
    typed_cfg: KShotSweepConfig = OmegaConf.to_object(cfg)
    repo_root = _repo_root()
    stage_a_source = _hydra_choice("stage_a_source") or typed_cfg.frozen_fusion_tag or "unknown_stage_a_source"
    p = typed_cfg.pipeline
    typed_cfg.pipeline.run_tag = (
        f"baseline_{p.baseline_method}_{p.baseline_backbone}_kshot_"
        f"{typed_cfg.kshot_shot}shot_run{typed_cfg.kshot_run_idx}_{typed_cfg.kshot_dataset_key}"
    )
    normalize_fusion_repo_paths(typed_cfg.pipeline, repo_root)
    manifest_path = resolve_repo_path(typed_cfg.pipeline.kshot_manifest_path, repo_root)
    results_root = resolve_repo_path(typed_cfg.kshot_results_root, repo_root)
    results_dir = results_root / stage_a_source / typed_cfg.kshot_dataset_key
    results_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = results_dir / f"{typed_cfg.kshot_shot}_shot_run_{typed_cfg.kshot_run_idx}.json"
    typed_cfg.pipeline.kshot_manifest_path = str(manifest_path)

    print("Resolved baseline kshot sweep config:")
    print(OmegaConf.to_yaml(cfg))
    print(f"[BASELINE] method={p.baseline_method}  backbone={p.baseline_backbone}")
    print(f"[BASELINE] Using manifest: {manifest_path}")
    print(f"[BASELINE] Result JSON: {run_json_path}")

    if not run_json_path.is_file() and not manifest_path.is_file():
        raise FileNotFoundError(f"k-shot manifest does not exist: {manifest_path}")

    if run_json_path.is_file() and typed_cfg.skip_existing_kshot_results:
        row = json.loads(run_json_path.read_text())
        print(f"[BASELINE] Skip existing result: {run_json_path}")
    else:
        metrics = run_single_backbone_baseline_experiment(typed_cfg.pipeline)
        group_name, flat_metrics = flatten_one_metrics(metrics)
        row = {
            "dataset": typed_cfg.kshot_dataset_key,
            "stage_a_source": stage_a_source,
            "shot": typed_cfg.kshot_shot,
            "run_index": typed_cfg.kshot_run_idx,
            "manifest_path": str(manifest_path),
            "postfix": group_name,
            "run_tag": typed_cfg.pipeline.run_tag,
            "baseline_method": p.baseline_method,
            "baseline_backbone": p.baseline_backbone,
            "test_accuracy": float(flat_metrics["test_accuracy"]),
            "test_precision": float(flat_metrics["test_precision"]),
            "test_recall": float(flat_metrics["test_recall"]),
            "test_ece": float(flat_metrics["test_ece"]),
            "auroc_err_pred_entropy": float(flat_metrics["auroc_err_pred_entropy"]),
            "fpr95_err_pred_entropy": float(flat_metrics["fpr95_err_pred_entropy"]),
        }
        write_json_atomic(run_json_path, row)
        print(f"[BASELINE] Wrote run result: {run_json_path}")

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
    print(f"[BASELINE] Updated summary: {summary_path}")


if __name__ == "__main__":
    register_configs()
    main()
