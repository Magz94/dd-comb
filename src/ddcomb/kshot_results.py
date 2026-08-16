from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev, pvariance
from typing import Any, Dict, List, Tuple


def resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    """Resolve relative project paths against the repository root.

    Hydra multiruns execute from per-job output directories, so plain relative
    paths like ``models`` or ``kshot/manifests`` must not be interpreted from
    the Hydra cwd.
    """
    resolved = Path(path)
    return resolved if resolved.is_absolute() else repo_root / resolved


def normalize_fusion_repo_paths(cfg: Any, repo_root: Path) -> None:
    """Make non-dataset fusion paths absolute under the repo when relative."""
    for name in ("base_models_path", "emb_cache_root", "img_plot_root", "model_save_path", "prediction_output_path"):
        value = getattr(cfg, name, None)
        if value is not None:
            setattr(cfg, name, str(resolve_repo_path(value, repo_root)))


def normalize_single_backbone_repo_paths(cfg: Any, repo_root: Path) -> None:
    """Make non-dataset single-backbone paths absolute under the repo when relative."""
    cfg.output_dir = str(resolve_repo_path(cfg.output_dir, repo_root))
    if getattr(cfg, "export_base_model_path", None):
        cfg.export_base_model_path = str(resolve_repo_path(cfg.export_base_model_path, repo_root))
    if getattr(cfg, "init_base_model_path", None):
        cfg.init_base_model_path = str(resolve_repo_path(cfg.init_base_model_path, repo_root))
    if getattr(cfg, "init_base_models_path", None):
        cfg.init_base_models_path = str(resolve_repo_path(cfg.init_base_models_path, repo_root))
    if getattr(cfg, "kshot_manifest_path", None):
        cfg.kshot_manifest_path = str(resolve_repo_path(cfg.kshot_manifest_path, repo_root))
    if getattr(cfg, "prediction_output_path", None):
        cfg.prediction_output_path = str(resolve_repo_path(cfg.prediction_output_path, repo_root))


def aggregate_run_rows(run_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    metric_keys = {
        "test_accuracy": "accuracy",
        "test_precision": "precision",
        "test_recall": "recall",
        "test_ece": "ece",
        "test_mce": "mce",
        "auroc_err_pred_entropy": "auroc_err_pred_entropy",
        "fpr95_err_pred_entropy": "fpr95_err_pred_entropy",
        "p_a_given_c_mean": "p_a_given_c_mean",
        "p_u_given_i_mean": "p_u_given_i_mean",
        "p_i_given_u_mean": "p_i_given_u_mean",
        "avu_mean": "avu_mean",
        "dgmobe_router_entropy_mean": "dgmobe_router_entropy",
        "dgmobe_router_max_mean": "dgmobe_router_max",
        "dgmobe_router_l1_from_uniform_mean": "dgmobe_router_l1_from_uniform",
        "dgmobe_expert_logit_cos_mean": "dgmobe_expert_logit_cos",
        "dgmobe_expert_logit_std_mean": "dgmobe_expert_logit_std",
    }
    summary = {
        "num_runs": float(len(run_rows)),
    }
    for row_key, metric_name in metric_keys.items():
        vals = [float(r[row_key]) for r in run_rows if row_key in r and r[row_key] is not None]
        if not vals:
            continue
        summary[f"{metric_name}_mean"] = mean(vals)
        summary[f"{metric_name}_variance"] = pvariance(vals)
        summary[f"{metric_name}_std"] = pstdev(vals)
    return summary


def flatten_one_metrics(metrics: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return the single nested metrics dict from a k-shot job."""
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError(f"Unexpected metrics shape: {metrics!r}")
    if len(metrics) > 1:
        raise ValueError(
            f"K-shot run produced {len(metrics)} metric groups; expected one. "
            f"Keys: {sorted(metrics.keys())}"
        )
    key = next(iter(metrics))
    return str(key), metrics[key]


def metrics_from_legacy_row(row: Dict[str, Any], group_name: str) -> Dict[str, Dict[str, float]]:
    metrics = {
        "test_accuracy": float(row["test_accuracy"]),
        "test_precision": float(row["test_precision"]),
        "test_recall": float(row["test_recall"]),
    }
    for key in (
        "test_ece",
        "test_mce",
        "auroc_err_pred_entropy",
        "fpr95_err_pred_entropy",
        "p_a_given_c_mean",
        "p_u_given_i_mean",
        "p_i_given_u_mean",
        "avu_mean",
    ):
        if key in row and row[key] is not None:
            metrics[key] = float(row[key])
    return {group_name: metrics}


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=float))
    tmp_path.replace(path)


def update_legacy_shot_summary(
    *,
    results_dir: Path,
    shot: int,
    summary_prefix: Dict[str, Any],
    expected_runs: int,
) -> Path:
    """Write the old ``{shot}_shot_summary.json`` from available run files."""
    run_rows: List[Dict[str, Any]] = []
    for run_idx in range(1, expected_runs + 1):
        run_path = results_dir / f"{shot}_shot_run_{run_idx}.json"
        if run_path.is_file():
            run_rows.append(json.loads(run_path.read_text()))

    if not run_rows:
        raise FileNotFoundError(f"No {shot}-shot run JSON files under {results_dir}")

    summary_payload = {k: v for k, v in summary_prefix.items() if k not in {"base_seed", "vary_training_seed_per_run"}}
    summary_payload["shot"] = shot
    summary_payload["runs"] = expected_runs
    if "base_seed" in summary_prefix:
        summary_payload["base_seed"] = summary_prefix["base_seed"]
    if "vary_training_seed_per_run" in summary_prefix:
        summary_payload["vary_training_seed_per_run"] = summary_prefix["vary_training_seed_per_run"]
    summary_payload["summary"] = aggregate_run_rows(run_rows)
    summary_payload["run_files"] = [f"{shot}_shot_run_{i}.json" for i in range(1, expected_runs + 1)]
    summary_path = results_dir / f"{shot}_shot_summary.json"
    write_json_atomic(summary_path, summary_payload)
    return summary_path
