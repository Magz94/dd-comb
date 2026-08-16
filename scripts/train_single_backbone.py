#!/usr/bin/env python3
"""Hydra entry point for single-backbone training/eval.

Usage:
    python scripts/train_single_backbone.py \
        single_backbone_dataset=plantvillage_color_split \
        model=resnet18 mode=head_only \
        use_wandb=true wandb_entity=kihug

Schema: ``SingleBackboneConfig``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from ddcomb.config import SingleBackboneConfig, register_configs
from ddcomb.kshot_results import normalize_single_backbone_repo_paths
from ddcomb.training.single_backbone import run_experiment


def _to_namespace(cfg: SingleBackboneConfig) -> argparse.Namespace:
    """Bridge SingleBackboneConfig → argparse.Namespace with Path conversions.

    ``training.single_backbone.run_experiment`` was built around the argparse
    Namespace produced by its parse_args, including ``Path``-typed fields. The
    structured config stores these as strings; we materialise the Paths here.
    """
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
        kshot_manifest_path=(Path(cfg.kshot_manifest_path) if cfg.kshot_manifest_path else None),
        prediction_output_path=(Path(cfg.prediction_output_path) if cfg.prediction_output_path else None),
        enable_uncertainty_metrics=cfg.enable_uncertainty_metrics,
        use_wandb=cfg.use_wandb,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@hydra.main(config_path="../conf", config_name="train_single_backbone", version_base=None)
def main(cfg) -> None:
    typed_cfg: SingleBackboneConfig = OmegaConf.to_object(cfg)
    normalize_single_backbone_repo_paths(typed_cfg, _repo_root())
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg))
    print("Repo-local paths:")
    print(f"  output_dir: {typed_cfg.output_dir}")
    if typed_cfg.export_base_model_path:
        print(f"  export_base_model_path: {typed_cfg.export_base_model_path}")
    if typed_cfg.init_base_model_path:
        print(f"  init_base_model_path: {typed_cfg.init_base_model_path}")
    if typed_cfg.init_base_models_path:
        print(f"  init_base_models_path: {typed_cfg.init_base_models_path}")

    args = _to_namespace(typed_cfg)
    results = run_experiment(args)
    print("Final results:", results)


if __name__ == "__main__":
    register_configs()
    main()
