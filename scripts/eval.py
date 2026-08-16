#!/usr/bin/env python3
"""Hydra entry point for evaluating a saved Stage-{A,B,D} fusion checkpoint.

Reuses ``TrainFusionConfig`` (shared with ``scripts/train_fusion.py``); the
only practical difference is that ``conf/eval.yaml`` defaults
``skip_train=true`` and ``return_full_metrics=true`` so a single CLI invocation
loads the checkpoint, runs evaluate_fusion_classifier + log_evaluation_artifacts,
and returns the metrics dict.

Usage:
    python scripts/eval.py mode=stage_d \\
        stage_a_source=img100_dirichlet_imagenet_1_acc85 \\
        dataset=plantvillage_color_split \\
        pipeline.run_tag=<the_run_tag_used_at_training>
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from ddcomb.config import TrainFusionConfig, register_configs
from ddcomb.kshot_results import normalize_fusion_repo_paths
from ddcomb.pipeline import run_experiment


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@hydra.main(config_path="../conf", config_name="eval", version_base=None)
def main(cfg) -> None:
    typed_cfg: TrainFusionConfig = OmegaConf.to_object(cfg)
    normalize_fusion_repo_paths(typed_cfg.pipeline, _repo_root())
    print("Resolved eval config:")
    print(OmegaConf.to_yaml(cfg))
    print("Repo-local paths:")
    print(f"  base_models_path: {typed_cfg.pipeline.base_models_path}")
    print(f"  model_save_path: {typed_cfg.pipeline.model_save_path}")
    print(f"  emb_cache_root: {typed_cfg.pipeline.emb_cache_root}")
    print(f"  img_plot_root: {typed_cfg.pipeline.img_plot_root}")

    metrics = run_experiment(
        cfg=typed_cfg.pipeline,
        mode=typed_cfg.mode,
        frozen_fusion_tag=typed_cfg.frozen_fusion_tag,
        frozen_fusion_added_postfix=typed_cfg.frozen_fusion_added_postfix,
        skip_train=typed_cfg.skip_train,
        return_full_metrics=typed_cfg.return_full_metrics,
    )
    print("Eval metrics:", metrics)


if __name__ == "__main__":
    register_configs()
    main()
