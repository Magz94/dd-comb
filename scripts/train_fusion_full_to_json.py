#!/usr/bin/env python3
"""Train/evaluate a full-data fusion run and write k-shot-like JSON results.

This is for "full" controls that deliberately do not use a k-shot manifest.
It mirrors ``scripts/train_fusion.py`` but stores one ``full_run_<idx>.json``
plus a ``full_summary.json`` so the output can sit next to k-shot ablations
without pretending that full-data training is an integer-shot manifest.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from ddcomb.config import TrainFusionConfig, register_configs
from ddcomb.kshot_results import (
    aggregate_run_rows,
    flatten_one_metrics,
    normalize_fusion_repo_paths,
    resolve_repo_path,
    write_json_atomic,
)
from ddcomb.pipeline import run_experiment


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _hydra_choice(name: str) -> str | None:
    choices = HydraConfig.get().runtime.choices
    value = choices.get(name)
    return None if value in (None, "null") else str(value)


@hydra.main(config_path="../conf", config_name="train_fusion", version_base=None)
def main(cfg) -> None:
    typed_cfg: TrainFusionConfig = OmegaConf.to_object(cfg)
    repo_root = _repo_root()
    normalize_fusion_repo_paths(typed_cfg.pipeline, repo_root)

    variant = os.environ.get("FULL_VARIANT", "full_variant")
    dataset_key = os.environ.get("FULL_DATASET_KEY", "deepweeds_no_bg_mergeval")
    run_idx = int(os.environ.get("FULL_RUN_IDX", "1"))
    expected_runs = int(os.environ.get("FULL_EXPECTED_RUNS", "1"))
    result_root = resolve_repo_path(
        os.environ.get("FULL_RESULTS_ROOT", "kshot/results_deepweeds_longk_professor_full"),
        repo_root,
    )
    stage_a_source = _hydra_choice("stage_a_source") or typed_cfg.frozen_fusion_tag or "unknown_stage_a_source"

    print("Resolved full-data fusion config:")
    print(OmegaConf.to_yaml(cfg))
    print("Full-data result metadata:")
    print(f"  variant: {variant}")
    print(f"  dataset_key: {dataset_key}")
    print(f"  run_idx: {run_idx}")
    print(f"  result_root: {result_root}")

    results = run_experiment(
        cfg=typed_cfg.pipeline,
        mode=typed_cfg.mode,
        frozen_fusion_tag=typed_cfg.frozen_fusion_tag,
        frozen_fusion_added_postfix=typed_cfg.frozen_fusion_added_postfix,
        skip_train=typed_cfg.skip_train,
        return_full_metrics=True,
    )
    group_name, flat_metrics = flatten_one_metrics(results)

    results_dir = result_root / variant / stage_a_source / dataset_key
    results_dir.mkdir(parents=True, exist_ok=True)
    run_json_path = results_dir / f"full_run_{run_idx}.json"

    row = {
        "dataset": dataset_key,
        "stage_a_source": stage_a_source,
        "shot": "full",
        "run_index": run_idx,
        "manifest_path": None,
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
    print(f"[FULL] Wrote run result: {run_json_path}")

    run_rows = []
    for idx in range(1, expected_runs + 1):
        path = results_dir / f"full_run_{idx}.json"
        if path.is_file():
            run_rows.append(json.loads(path.read_text()))
    summary_payload = {
        "dataset": dataset_key,
        "stage_a_source": stage_a_source,
        "shot": "full",
        "runs": expected_runs,
        "summary": aggregate_run_rows(run_rows),
        "run_files": [f"full_run_{idx}.json" for idx in range(1, expected_runs + 1)],
    }
    summary_path = results_dir / "full_summary.json"
    write_json_atomic(summary_path, summary_payload)
    print(f"[FULL] Updated summary: {summary_path}")
    print("Final results:", results)


if __name__ == "__main__":
    register_configs()
    main()
