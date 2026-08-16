#!/usr/bin/env python3
"""Plot 5-shot macro-averaged PR curves with run-to-run std bands.

The script expects prediction exports saved next to k-shot run JSONs:

    5_shot_run_1_predictions.npz
    ...
    5_shot_run_20_predictions.npz

Each ``.npz`` must contain:

    y_true: shape [N]
    y_prob: shape [N, C]

The k-shot sweep scripts can write these files with:

    export_kshot_predictions=true

For multiclass datasets, curves and scalar scores are macro-averaged
one-vs-rest: a PR curve is computed per class, then averaged across classes.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, average_precision_score, precision_recall_curve


SHOT = 5
EXPECTED_RUNS = 20
RECALL_GRID = np.linspace(0.0, 1.0, 201)


DATASETS = {
    "DeepWeeds": "deepweeds_no_bg_mergeval",
    "LeafSugar": "leafsugar_disease_split",
    "APTOS 2019": "aptos2019_split",
}

DATASET_NUM_CLASSES = {
    "deepweeds_no_bg_mergeval": 8,
    "leafsugar_disease_split": 5,
    "aptos2019_split": 5,
}


MODEL_STYLES = {
    "Dirichlet Attention": {"color": "#DD8452", "marker": "s"},
    "ViT-B16": {"color": "#4C72B0", "marker": "o"},
    "DenseNet121": {"color": "#55A868", "marker": "^"},
    "EfficientNet-B0": {"color": "#C44E52", "marker": "D"},
    "VGG16": {"color": "#8172B3", "marker": "P"},
    "Fusion Baseline": {"color": "#64B5CD", "marker": "v"},
    "ResNet18": {"color": "#937860", "marker": "X"},
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    kind: str
    key: str


MODEL_SPECS = [
    ModelSpec("ViT-B16", "single", "vit_b16"),
    ModelSpec("DenseNet121", "single", "densenet121"),
    ModelSpec("EfficientNet-B0", "single", "efficientnet_b0"),
    ModelSpec("ResNet18", "single", "resnet18"),
    ModelSpec("VGG16", "single", "vgg16"),
    ModelSpec("Fusion Baseline", "fusion", "img100_stagea_softmax_linear_resnet_order_v1"),
    ModelSpec("Dirichlet Attention", "fusion", "img100_dirichlet_imagenet_1_acc85"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--shot", type=int, default=SHOT)
    parser.add_argument("--expected-runs", type=int, default=EXPECTED_RUNS)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("pr_curves_kshot5_macro.png"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("pr_curves_kshot5_macro_metrics.csv"))
    return parser.parse_args()


def candidate_result_dirs(repo_root: Path, dataset_key: str, spec: ModelSpec) -> list[Path]:
    if spec.kind == "single":
        return [
            repo_root / "kshot" / "results_single_backbone" / dataset_key / spec.key / "head_only_basic",
        ]

    if spec.key == "img100_stagea_softmax_linear_resnet_order_v1":
        return [
            repo_root / "kshot" / "results" / spec.key / dataset_key,
        ]

    if spec.key == "img100_dirichlet_imagenet_1_acc85":
        return [
            repo_root / "kshot" / "results" / spec.key / dataset_key,
        ]

    return [repo_root / "kshot" / "results" / spec.key / dataset_key]


def prediction_path_for_run(result_dir: Path, shot: int, run_idx: int) -> Path:
    return result_dir / f"{shot}_shot_run_{run_idx}_predictions.npz"


def find_prediction_files(
    repo_root: Path,
    dataset_key: str,
    spec: ModelSpec,
    shot: int,
    expected_runs: int,
) -> tuple[list[Path], list[Path]]:
    best_found: list[Path] = []
    best_missing: list[Path] = []
    for result_dir in candidate_result_dirs(repo_root, dataset_key, spec):
        paths = [prediction_path_for_run(result_dir, shot, i) for i in range(1, expected_runs + 1)]
        found = [path for path in paths if path.is_file()]
        missing = [path for path in paths if not path.is_file()]
        if found and not missing:
            return found, missing
        if len(found) > len(best_found):
            best_found = found
            best_missing = missing
    if best_found:
        return best_found, best_missing
    first_dir = candidate_result_dirs(repo_root, dataset_key, spec)[0]
    return [], [prediction_path_for_run(first_dir, shot, i) for i in range(1, expected_runs + 1)]


def unique_recall_precision(recall: np.ndarray, precision: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(recall)
    recall_sorted = recall[order]
    precision_sorted = precision[order]
    unique_recall, inverse = np.unique(recall_sorted, return_inverse=True)
    precision_unique = np.zeros_like(unique_recall, dtype=float)
    np.maximum.at(precision_unique, inverse, precision_sorted)
    return unique_recall, precision_unique


def load_macro_pr(path: Path, recall_grid: np.ndarray) -> tuple[np.ndarray, float, float]:
    data = np.load(path)
    y_true = data["y_true"].astype(int)
    y_prob = data["y_prob"].astype(float)
    if y_prob.ndim != 2:
        raise ValueError(f"{path} has invalid y_prob shape {y_prob.shape}; expected [N, C].")
    num_classes = y_prob.shape[1]
    if y_true.size == 0:
        raise ValueError(f"{path} has empty y_true.")
    if y_true.min() < 0 or y_true.max() >= num_classes:
        raise ValueError(
            f"{path} labels are outside [0, {num_classes - 1}]: "
            f"min={y_true.min()}, max={y_true.max()}"
        )

    y_true_onehot = np.eye(num_classes, dtype=int)[y_true]
    class_curves: list[np.ndarray] = []
    ap_values: list[float] = []
    pr_auc_values: list[float] = []

    for class_idx in range(num_classes):
        y_true_class = y_true_onehot[:, class_idx]
        y_score_class = y_prob[:, class_idx]
        if y_true_class.sum() == 0:
            raise ValueError(f"{path} has no positive samples for class {class_idx}.")

        precision, recall, _ = precision_recall_curve(y_true_class, y_score_class)
        recall_unique, precision_unique = unique_recall_precision(recall, precision)
        precision_interp = np.interp(
            recall_grid,
            recall_unique,
            precision_unique,
            left=precision_unique[0],
            right=precision_unique[-1],
        )
        class_curves.append(precision_interp)
        ap_values.append(float(average_precision_score(y_true_class, y_score_class)))
        pr_auc_values.append(float(auc(recall, precision)))

    return (
        np.vstack(class_curves).mean(axis=0),
        float(np.mean(ap_values)),
        float(np.mean(pr_auc_values)),
    )


def prediction_num_classes(path: Path) -> int:
    data = np.load(path, allow_pickle=False)
    if "num_classes" in data.files:
        return int(np.asarray(data["num_classes"]).reshape(-1)[0])
    return int(data["y_prob"].shape[1])


def mean_std_band(values: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.vstack(list(values))
    mean_curve = arr.mean(axis=0)
    std_curve = arr.std(axis=0, ddof=0)
    lower = np.clip(mean_curve - std_curve, 0.0, 1.0)
    upper = np.clip(mean_curve + std_curve, 0.0, 1.0)
    return mean_curve, lower, upper


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    metrics_path = args.metrics_csv if args.metrics_csv.is_absolute() else repo_root / args.metrics_csv

    sns.set_theme(style="darkgrid", palette="muted")
    num_datasets = len(DATASETS)
    fig, axes = plt.subplots(1, num_datasets, figsize=(5.2 * num_datasets, 4.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    metric_rows: list[dict[str, object]] = []
    missing_messages: list[str] = []

    for ax, (dataset_label, dataset_key) in zip(axes, DATASETS.items()):
        plotted_any = False
        baseline_classes = DATASET_NUM_CLASSES[dataset_key]
        for spec in MODEL_SPECS:
            found, missing = find_prediction_files(
                repo_root=repo_root,
                dataset_key=dataset_key,
                spec=spec,
                shot=args.shot,
                expected_runs=args.expected_runs,
            )
            if missing and not args.allow_partial:
                missing_messages.append(
                    f"{dataset_label} / {spec.label}: found {len(found)}/{args.expected_runs}; "
                    f"first missing: {missing[0]}"
                )
                continue
            if not found:
                missing_messages.append(f"{dataset_label} / {spec.label}: no prediction files found.")
                continue
            baseline_classes = prediction_num_classes(found[0])

            curves, ap_values, pr_auc_values = [], [], []
            for path in found:
                precision_interp, ap_macro, pr_auc_macro = load_macro_pr(path, RECALL_GRID)
                curves.append(precision_interp)
                ap_values.append(ap_macro)
                pr_auc_values.append(pr_auc_macro)

            mean_curve, lower_curve, upper_curve = mean_std_band(curves)
            style = MODEL_STYLES[spec.label]
            ax.plot(
                RECALL_GRID,
                mean_curve,
                color=style["color"],
                linewidth=2.1,
                label=spec.label,
            )
            ax.fill_between(
                RECALL_GRID,
                lower_curve,
                upper_curve,
                color=style["color"],
                alpha=0.15,
                linewidth=0,
            )
            plotted_any = True

            metric_rows.append(
                {
                    "dataset": dataset_key,
                    "model": spec.label,
                    "shot": args.shot,
                    "runs_found": len(found),
                    "ap_macro_mean": float(np.mean(ap_values)),
                    "ap_macro_std": float(np.std(ap_values, ddof=0)),
                    "pr_auc_macro_mean": float(np.mean(pr_auc_values)),
                    "pr_auc_macro_std": float(np.std(pr_auc_values, ddof=0)),
                }
            )

        baseline_y = 1.0 / baseline_classes
        ax.axhline(
            baseline_y,
            color="#3F3F3F",
            linestyle="--",
            linewidth=1.6,
            alpha=0.75,
            label="Random baseline" if dataset_label == next(iter(DATASETS)) else None,
        )
        ax.text(
            0.985,
            baseline_y + 0.015,
            f"random={baseline_y:.3f}",
            ha="right",
            va="bottom",
            fontsize=8,
            color="#3F3F3F",
            alpha=0.85,
        )
        ax.set_title(dataset_label, fontsize=13, fontweight="bold")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(np.arange(0.0, 1.01, 0.2))
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        if not plotted_any:
            ax.text(
                0.5,
                0.5,
                "No prediction files found",
                ha="center",
                va="center",
                fontsize=11,
                transform=ax.transAxes,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes[1:]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, title="5-shot models")
    fig.suptitle("5-shot Macro-Averaged Precision-Recall Curves", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.18, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model",
                "shot",
                "runs_found",
                "ap_macro_mean",
                "ap_macro_std",
                "pr_auc_macro_mean",
                "pr_auc_macro_std",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    print(f"Wrote plot: {output_path}")
    print(f"Wrote metrics: {metrics_path}")
    if missing_messages:
        print("\nMissing prediction exports:")
        for msg in missing_messages[:40]:
            print(f"  - {msg}")
        if len(missing_messages) > 40:
            print(f"  ... and {len(missing_messages) - 40} more")


if __name__ == "__main__":
    main()
