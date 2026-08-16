#!/usr/bin/env python3
"""Hydra entry point for producing stratified k-shot manifests.

Drives a single call to ``data.kshot.ensure_kshot_manifests`` for one
dataset. Manifests land at::

    ${manifest_root}/${dataset_key}/{k_shot}_shot_run_{run_idx}.json

matching the file pattern that ``conf/sweep_kshot.yaml`` later interpolates.
By default, ``manifest_root`` is repo-local: ``kshot/manifests``.

Examples:
    # ImageFolder split (PlantVillage):
    python scripts/build_kshot_manifests.py \\
        dataset_key=plantvillage_color_split \\
        dataset_type=imagefolder \\
        imagefolder_root=/home/magz94/ddcomb/data/plantvillage_color_split

    # DeepWeeds with label-8 dropped and val merged into train:
    python scripts/build_kshot_manifests.py \\
        dataset_key=deepweeds_no_bg_mergeval \\
        dataset_type=deepweeds_csv \\
        deepweeds_labels_dir=/datasets/DeepWeeds/labels
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from ddcomb.config import KShotManifestConfig, register_configs
from ddcomb.data.kshot import (
    build_deepweeds_pool,
    build_imagefolder_pool,
    ensure_kshot_manifests,
)
from ddcomb.kshot_results import resolve_repo_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@hydra.main(config_path="../conf", config_name="build_kshot_manifests", version_base=None)
def main(cfg) -> None:
    typed_cfg: KShotManifestConfig = OmegaConf.to_object(cfg)
    repo_root = _repo_root()
    manifest_root = resolve_repo_path(typed_cfg.manifest_root, repo_root)
    print("Resolved manifest-build config:")
    print(OmegaConf.to_yaml(cfg))

    if typed_cfg.dataset_type == "deepweeds_csv":
        pool_by_label = build_deepweeds_pool(
            labels_dir=Path(typed_cfg.deepweeds_labels_dir),
            subset=typed_cfg.deepweeds_subset,
            drop_label8=typed_cfg.deepweeds_drop_label8,
            merge_val_into_train=typed_cfg.deepweeds_merge_val_into_train,
        )
        meta = {
            "subset": typed_cfg.deepweeds_subset,
            "drop_label8": typed_cfg.deepweeds_drop_label8,
            "merge_val_into_train": typed_cfg.deepweeds_merge_val_into_train,
        }
    else:  # imagefolder — validated in __post_init__
        pool_by_label = build_imagefolder_pool(
            dataset_root=Path(typed_cfg.imagefolder_root),
            train_split=typed_cfg.imagefolder_train_split,
            merge_split=typed_cfg.imagefolder_merge_split,
        )
        meta = {
            "train_pool": (
                f"{typed_cfg.imagefolder_train_split} + {typed_cfg.imagefolder_merge_split}"
                if typed_cfg.imagefolder_merge_split
                else typed_cfg.imagefolder_train_split
            ),
        }

    manifests_by_shot = ensure_kshot_manifests(
        dataset_key=typed_cfg.dataset_key,
        dataset_type=typed_cfg.dataset_type,
        pool_by_label=pool_by_label,
        shots=list(typed_cfg.shots),
        repeats=typed_cfg.repeats,
        base_seed=typed_cfg.base_seed,
        manifest_root=manifest_root,
        force_resample=typed_cfg.force_resample,
        extra_meta=meta,
    )

    total = sum(len(paths) for paths in manifests_by_shot.values())
    target = manifest_root / typed_cfg.dataset_key
    print(f"Wrote/verified {total} manifest file(s) under {target}/")


if __name__ == "__main__":
    register_configs()
    main()
