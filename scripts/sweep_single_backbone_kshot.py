#!/usr/bin/env python3
"""Hydra entry point for k-shot multirun single-backbone training.

Each Hydra multirun job selects one (model, shot, run_idx) combination and one
shared k-shot manifest. This is the single-backbone counterpart to
``scripts/sweep_kshot.py`` for fusion experiments. Results are mirrored to the
legacy repo-local layout
``kshot/results_single_backbone/<dataset_key>/<model>/<mode>_<preprocess>/``.

Usage:
    python scripts/sweep_single_backbone_kshot.py -m \\
        single_backbone_dataset=plantvillage_color_split \\
        model=vgg16 \\
        mode=head_only \\
        kshot_dataset_key=plantvillage_color_split \\
        kshot_shot=1,5,10,20 \\
        kshot_run_idx=range(1,21)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from ddcomb.config import SingleBackboneKShotSweepConfig, register_configs
from ddcomb.kshot_results import (
    metrics_from_legacy_row,
    normalize_single_backbone_repo_paths,
    resolve_repo_path,
    update_legacy_shot_summary,
    write_json_atomic,
)
from ddcomb.training.single_backbone import run_experiment


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _to_namespace(cfg: SingleBackboneKShotSweepConfig) -> argparse.Namespace:
    return argparse.Namespace(
        model=cfg.model,
        dataset=cfg.dataset_name,
        mode=cfg.mode,
        preprocess=cfg.preprocess,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        optimizer=cfg.optimizer,
        label_smoothing=cfg.label_smoothing,
        amp=cfg.amp,
        merge_val_into_train=cfg.merge_val_into_train,
        deepweeds_root=Path(cfg.deepweeds_root),
        deepweeds_subset=cfg.deepweeds_subset,
        imagenet100_root=Path(cfg.imagenet100_root),
        ssl_patches_root=Path(cfg.ssl_patches_root),
        ninco_ood_root=Path(cfg.ninco_ood_root),
        pathmnist_root=Path(cfg.pathmnist_root),
        bloodmnist_root=Path(cfg.bloodmnist_root),
        aptos_root=Path(cfg.aptos_root),
        plantvillage_root=Path(cfg.plantvillage_root),
        cassava_root=Path(cfg.cassava_root),
        leafsugar_root=Path(cfg.leafsugar_root),
        plant_seedlings_root=Path(cfg.plant_seedlings_root),
        sar_cld_root=Path(cfg.sar_cld_root),
        output_dir=Path(cfg.output_dir),
        export_base_model_path=(Path(cfg.export_base_model_path) if cfg.export_base_model_path else None),
        init_base_model_path=(Path(cfg.init_base_model_path) if cfg.init_base_model_path else None),
        init_base_models_path=(Path(cfg.init_base_models_path) if cfg.init_base_models_path else None),
        init_base_subfolder=cfg.init_base_subfolder,
        run_name=cfg.run_name,
        kshot_manifest_path=Path(cfg.kshot_manifest_path),
        prediction_output_path=(Path(cfg.prediction_output_path) if cfg.prediction_output_path else None),
        enable_uncertainty_metrics=cfg.enable_uncertainty_metrics,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
    )


def _headline_metrics(summary: dict) -> dict:
    test_metrics = summary["test_metrics"]
    flat = {
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
    }
    optional_key_map = {
        "ece": "test_ece",
        "mce": "test_mce",
        "auroc_err_pred_entropy": "auroc_err_pred_entropy",
        "fpr@95_err_pred_entropy": "fpr95_err_pred_entropy",
        "p_a_given_c_mean": "p_a_given_c_mean",
        "p_u_given_i_mean": "p_u_given_i_mean",
        "p_i_given_u_mean": "p_i_given_u_mean",
        "avu_mean": "avu_mean",
    }
    for source_key, row_key in optional_key_map.items():
        if source_key in test_metrics and test_metrics[source_key] is not None:
            flat[row_key] = float(test_metrics[source_key])
    return flat


@hydra.main(config_path="../conf", config_name="sweep_single_backbone_kshot", version_base=None)
def main(cfg) -> None:
    typed_cfg: SingleBackboneKShotSweepConfig = OmegaConf.to_object(cfg)
    repo_root = _repo_root()
    normalize_single_backbone_repo_paths(typed_cfg, repo_root)
    results_root = resolve_repo_path(typed_cfg.kshot_results_root, repo_root)
    results_dir = (
        results_root
        / typed_cfg.kshot_dataset_key
        / typed_cfg.model
        / f"{typed_cfg.mode}_{typed_cfg.preprocess}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = results_dir / f"{typed_cfg.kshot_shot}_shot_run_{typed_cfg.kshot_run_idx}.json"
    prediction_path = run_json_path.with_name(
        f"{typed_cfg.kshot_shot}_shot_run_{typed_cfg.kshot_run_idx}_predictions.npz"
    )
    if typed_cfg.export_kshot_predictions:
        typed_cfg.prediction_output_path = str(prediction_path)
    prediction_missing = typed_cfg.export_kshot_predictions and not prediction_path.is_file()

    print("Resolved single-backbone kshot sweep config:")
    print(OmegaConf.to_yaml(cfg))
    print("Repo-local paths:")
    print(f"  output_dir: {typed_cfg.output_dir}")
    print(f"[KSHOT-SB] Using manifest: {typed_cfg.kshot_manifest_path}")
    print(f"[KSHOT-SB] Result JSON: {run_json_path}")

    args = _to_namespace(typed_cfg)
    if not run_json_path.is_file() and not args.kshot_manifest_path.is_file():
        raise FileNotFoundError(f"k-shot manifest does not exist: {args.kshot_manifest_path}")

    if run_json_path.is_file() and typed_cfg.skip_existing_kshot_results and not prediction_missing:
        row = json.loads(run_json_path.read_text())
        flat_metrics = metrics_from_legacy_row(row, "single_backbone")["single_backbone"]
        print(f"[KSHOT-SB] Skip existing result: {run_json_path}")
    else:
        if run_json_path.is_file() and prediction_missing:
            print(f"[KSHOT-SB] Existing result has no prediction export; rerunning to write: {prediction_path}")
        summary = run_experiment(args)
        flat_metrics = _headline_metrics(summary)
        row = {
            "dataset": typed_cfg.dataset_name,
            "dataset_key": typed_cfg.kshot_dataset_key,
            "model": typed_cfg.model,
            "mode": typed_cfg.mode,
            "preprocess": typed_cfg.preprocess,
            "shot": typed_cfg.kshot_shot,
            "run_index": typed_cfg.kshot_run_idx,
            "manifest_path": typed_cfg.kshot_manifest_path,
            "run_name": typed_cfg.run_name,
            **flat_metrics,
        }
        write_json_atomic(run_json_path, row)
        print(f"[KSHOT-SB] Wrote legacy run result: {run_json_path}")

    summary_path = update_legacy_shot_summary(
        results_dir=results_dir,
        shot=typed_cfg.kshot_shot,
        expected_runs=typed_cfg.kshot_expected_runs,
        summary_prefix={
            "dataset": typed_cfg.dataset_name,
            "dataset_key": typed_cfg.kshot_dataset_key,
            "model": typed_cfg.model,
            "mode": typed_cfg.mode,
            "preprocess": typed_cfg.preprocess,
            "base_seed": typed_cfg.kshot_base_seed,
            "vary_training_seed_per_run": False,
        },
    )
    print(f"[KSHOT-SB] Updated legacy summary: {summary_path}")

    print(
        f"Metrics for {typed_cfg.model} {typed_cfg.kshot_shot}-shot "
        f"run {typed_cfg.kshot_run_idx}: {flat_metrics}"
    )


if __name__ == "__main__":
    register_configs()
    main()
