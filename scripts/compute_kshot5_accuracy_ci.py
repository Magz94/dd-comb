#!/usr/bin/env python3
"""Compute a paper-ready 5-shot Accuracy/F1 table.

The table is intended for the final few-shot datasets:
DeepWeeds, Plant Seedlings V2, BloodMNIST, and APTOS 2019.

By default, DeepWeeds and APTOS single-backbone rows use the legacy macro
summary values used by scripts/output_curves_macro.py, while fusion rows and
the other datasets are read from repo-local k-shot result JSONs.

Output columns:
    Dataset, Model, Accuracy (95% CI), Macro-F1 (std)
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


SHOT = 5
EXPECTED_RUNS = 20
T_CRITICAL_95_DF19 = 2.093024054408263

DATASETS = {
    "DeepWeeds": "deepweeds_no_bg_mergeval",
    "Plant Seedlings V2": "plant_seedlings_v2_split",
    "BloodMNIST": "bloodmnist_224_imagefolder_mergeval",
    "APTOS 2019": "aptos2019_split",
}

MODEL_DIRS = {
    "ViT-B16": ("single", "vit_b16/head_only_basic"),
    "DenseNet121": ("single", "densenet121/head_only_basic"),
    "EfficientNet-B0": ("single", "efficientnet_b0/head_only_basic"),
    "ResNet18": ("single", "resnet18/head_only_basic"),
    "VGG16": ("single", "vgg16/head_only_basic"),
    "Fusion Baseline": ("fusion", "img100_stagea_softmax_linear_resnet_order_v1"),
    "Dirichlet Attention": ("fusion", "img100_dirichlet_imagenet_1_acc85"),
}

MODEL_ORDER = list(MODEL_DIRS)
SINGLE_BACKBONE_MODEL_NAMES = {"ViT-B16", "DenseNet121", "EfficientNet-B0", "ResNet18", "VGG16"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--shot", type=int, default=SHOT)
    parser.add_argument("--expected-runs", type=int, default=EXPECTED_RUNS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("kshot5_accuracy_ci95.csv"),
        help="CSV output path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--no-legacy-single-backbone",
        action="store_true",
        help="Read DeepWeeds/APTOS single-backbone rows from kshot JSONs instead of legacy summary values.",
    )
    return parser.parse_args()


def result_dir(repo_root: Path, dataset_key: str, model_name: str) -> Path:
    kind, key = MODEL_DIRS[model_name]
    if kind == "single":
        return repo_root / "kshot" / "results_single_backbone" / dataset_key / key
    return repo_root / "kshot" / "results" / key / dataset_key


def load_legacy_single_backbone_metrics(repo_root: Path) -> dict:
    """Read the legacy DeepWeeds/APTOS single-backbone table without importing matplotlib."""
    source_path = repo_root / "scripts" / "output_curves_macro.py"
    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "LEGACY_SINGLE_BACKBONE_METRICS":
                return ast.literal_eval(node.value)
    raise RuntimeError(f"Could not find LEGACY_SINGLE_BACKBONE_METRICS in {source_path}")


def load_metric_runs(
    repo_root: Path,
    dataset_key: str,
    model_name: str,
    shot: int,
    expected_runs: int,
) -> dict[str, list[float]]:
    directory = result_dir(repo_root, dataset_key, model_name)
    values = {"accuracy": [], "precision": [], "recall": []}
    missing = []
    for run_idx in range(1, expected_runs + 1):
        path = directory / f"{shot}_shot_run_{run_idx}.json"
        if not path.is_file():
            missing.append(path)
            continue
        payload = json.loads(path.read_text())
        values["accuracy"].append(float(payload["test_accuracy"]))
        values["precision"].append(float(payload["test_precision"]))
        values["recall"].append(float(payload["test_recall"]))
    if missing:
        missing_preview = "\n".join(str(p) for p in missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} run JSONs for {dataset_key} / {model_name}. "
            f"First missing:\n{missing_preview}"
        )
    return values


def mean_std_ci95(values: list[float]) -> tuple[float, float, float, int]:
    if len(values) < 2:
        raise ValueError("At least two runs are required for a confidence interval.")
    avg = mean(values)
    sample_std = stdev(values)
    half_width = T_CRITICAL_95_DF19 * sample_std / math.sqrt(len(values))
    return avg, sample_std, half_width, len(values)


def ci95_from_summary_std(mean_value: float, std_value: float, n: int) -> float:
    half_width = T_CRITICAL_95_DF19 * std_value / math.sqrt(n)
    return half_width


def f1_from_precision_recall(precision: float, recall: float) -> float:
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    return 2.0 * precision * recall / denom


def propagated_f1_std(precision: float, recall: float, precision_std: float, recall_std: float) -> float:
    denom = precision + recall
    if denom == 0.0:
        return 0.0
    d_f1_d_precision = 2.0 * recall**2 / denom**2
    d_f1_d_recall = 2.0 * precision**2 / denom**2
    return math.sqrt((d_f1_d_precision * precision_std) ** 2 + (d_f1_d_recall * recall_std) ** 2)


def format_pm(center: float, spread: float) -> str:
    return f"{center:.4f} ± {spread:.4f}"


def legacy_metrics_row(
    legacy_metrics: dict,
    dataset_key: str,
    model_name: str,
    shot: int,
    expected_runs: int,
) -> tuple[str, str]:
    shot_idx = [1, 5, 10, 20].index(shot)
    row = legacy_metrics[dataset_key][model_name]
    accuracy = float(row["Accuracy"][shot_idx])
    accuracy_std = float(row["Accuracy_err"][shot_idx])
    precision = float(row["Precision"][shot_idx])
    precision_std = float(row["Precision_err"][shot_idx])
    recall = float(row["Recall"][shot_idx])
    recall_std = float(row["Recall_err"][shot_idx])
    accuracy_ci = ci95_from_summary_std(accuracy, accuracy_std, expected_runs)
    f1 = f1_from_precision_recall(precision, recall)
    f1_std = propagated_f1_std(precision, recall, precision_std, recall_std)
    return format_pm(accuracy, accuracy_ci), format_pm(f1, f1_std)


def kshot_json_metrics_row(
    repo_root: Path,
    dataset_key: str,
    model_name: str,
    shot: int,
    expected_runs: int,
) -> tuple[str, str]:
    metric_runs = load_metric_runs(repo_root, dataset_key, model_name, shot, expected_runs)
    accuracy_mean, _, accuracy_ci, _ = mean_std_ci95(metric_runs["accuracy"])
    precision_mean, precision_std, _, _ = mean_std_ci95(metric_runs["precision"])
    recall_mean, recall_std, _, _ = mean_std_ci95(metric_runs["recall"])
    f1 = f1_from_precision_recall(precision_mean, recall_mean)
    f1_std = propagated_f1_std(precision_mean, recall_mean, precision_std, recall_std)
    return format_pm(accuracy_mean, accuracy_ci), format_pm(f1, f1_std)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    legacy_metrics = load_legacy_single_backbone_metrics(repo_root)

    rows = []
    for dataset_label, dataset_key in DATASETS.items():
        for model_name in MODEL_ORDER:
            use_legacy = (
                not args.no_legacy_single_backbone
                and dataset_key in legacy_metrics
                and model_name in SINGLE_BACKBONE_MODEL_NAMES
            )
            if use_legacy:
                accuracy_display, f1_display = legacy_metrics_row(
                    legacy_metrics,
                    dataset_key,
                    model_name,
                    args.shot,
                    args.expected_runs,
                )
            else:
                accuracy_display, f1_display = kshot_json_metrics_row(
                    repo_root,
                    dataset_key,
                    model_name,
                    args.shot,
                    args.expected_runs,
                )
            rows.append(
                {
                    "Dataset": dataset_label,
                    "Model": model_name,
                    "Accuracy (95% CI)": accuracy_display,
                    "Macro-F1 (std)": f1_display,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote: {output_path}")
    print("Dataset,Model,Accuracy (95% CI),Macro-F1 (std)")
    for row in rows:
        print(
            f"{row['Dataset']},{row['Model']},"
            f"{row['Accuracy (95% CI)']},{row['Macro-F1 (std)']}"
        )


if __name__ == "__main__":
    main()
