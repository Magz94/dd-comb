#!/usr/bin/env python3
"""Plot final-paper k-shot benchmark curves from repo-local k-shot summaries.

This active script intentionally contains only the reported datasets:
DeepWeeds, Plant Seedlings V2, BloodMNIST, and APTOS 2019.

Legacy exploratory hard-coded values are preserved in:
    scripts/output_curves_macro_legacy_all_datasets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUSION_RESULTS_ROOT = PROJECT_ROOT / "kshot" / "results"
SINGLE_BACKBONE_RESULTS_ROOT = PROJECT_ROOT / "kshot" / "results_single_backbone"

SHOTS = [1, 5, 10, 20]
METRICS = ["Accuracy", "Precision", "Recall", "F1-Score"]

DATASETS = {
    "DeepWeeds": "deepweeds_no_bg_mergeval",
    "Plant Seedlings V2": "plant_seedlings_v2_split",
    "BloodMNIST": "bloodmnist_224_imagefolder_mergeval",
    "APTOS 2019": "aptos2019_split",
}

MODEL_SOURCES = {
    "ViT-B16": SINGLE_BACKBONE_RESULTS_ROOT / "{dataset_key}" / "vit_b16" / "head_only_basic",
    "DenseNet121": SINGLE_BACKBONE_RESULTS_ROOT / "{dataset_key}" / "densenet121" / "head_only_basic",
    "EfficientNet-B0": SINGLE_BACKBONE_RESULTS_ROOT / "{dataset_key}" / "efficientnet_b0" / "head_only_basic",
    "ResNet18": SINGLE_BACKBONE_RESULTS_ROOT / "{dataset_key}" / "resnet18" / "head_only_basic",
    "VGG16": SINGLE_BACKBONE_RESULTS_ROOT / "{dataset_key}" / "vgg16" / "head_only_basic",
    "Fusion Baseline": FUSION_RESULTS_ROOT / "img100_stagea_softmax_linear_resnet_order_v1" / "{dataset_key}",
    "Dirichlet Attention": FUSION_RESULTS_ROOT / "img100_dirichlet_imagenet_1_acc85" / "{dataset_key}",
}

MODEL_ORDER = list(MODEL_SOURCES.keys())

SINGLE_BACKBONE_MODEL_NAMES = {
    "ViT-B16",
    "DenseNet121",
    "EfficientNet-B0",
    "ResNet18",
    "VGG16",
}

# These two datasets keep the legacy single-backbone macro values. Fusion rows
# are still loaded from kshot/results so updated fusion reruns are reflected.
LEGACY_SINGLE_BACKBONE_METRICS = {
    "deepweeds_no_bg_mergeval": {
        "ViT-B16": {
            "Accuracy": [0.2789, 0.4164, 0.5197, 0.5971],
            "Accuracy_err": [0.0376, 0.0227, 0.0213, 0.0146],
            "Precision": [0.2925, 0.4514, 0.5289, 0.6018],
            "Precision_err": [0.0355, 0.0275, 0.0219, 0.0130],
            "Recall": [0.2776, 0.4158, 0.5199, 0.5974],
            "Recall_err": [0.0381, 0.0229, 0.0208, 0.0144],
        },
        "DenseNet121": {
            "Accuracy": [0.2364, 0.4380, 0.5444, 0.6143],
            "Accuracy_err": [0.0303, 0.0175, 0.0149, 0.0130],
            "Precision": [0.2601, 0.4527, 0.5504, 0.6168],
            "Precision_err": [0.0345, 0.0202, 0.0210, 0.0125],
            "Recall": [0.2377, 0.4389, 0.5452, 0.6146],
            "Recall_err": [0.0308, 0.0176, 0.0143, 0.0130],
        },
        "EfficientNet-B0": {
            "Accuracy": [0.2489, 0.4249, 0.5217, 0.5814],
            "Accuracy_err": [0.0295, 0.0272, 0.0211, 0.0165],
            "Precision": [0.2640, 0.4451, 0.5240, 0.5810],
            "Precision_err": [0.0342, 0.0272, 0.0236, 0.0153],
            "Recall": [0.2494, 0.4247, 0.5218, 0.5811],
            "Recall_err": [0.0301, 0.0273, 0.0204, 0.0167],
        },
        "VGG16": {
            "Accuracy": [0.2372, 0.3213, 0.3801, 0.4565],
            "Accuracy_err": [0.0273, 0.0280, 0.0240, 0.0183],
            "Precision": [0.2685, 0.3612, 0.4241, 0.4758],
            "Precision_err": [0.0615, 0.0612, 0.0432, 0.0143],
            "Recall": [0.2387, 0.3229, 0.3817, 0.4573],
            "Recall_err": [0.0276, 0.0287, 0.0242, 0.0186],
        },
        "ResNet18": {
            "Accuracy": [0.1991, 0.3058, 0.4407, 0.5316],
            "Accuracy_err": [0.0229, 0.0260, 0.0193, 0.0152],
            "Precision": [0.2345, 0.3590, 0.4521, 0.5357],
            "Precision_err": [0.0279, 0.0311, 0.0237, 0.0121],
            "Recall": [0.1998, 0.3078, 0.4413, 0.5322],
            "Recall_err": [0.0237, 0.0253, 0.0189, 0.0148],
        },
    },
    "aptos2019_split": {
        "ViT-B16": {
            "Accuracy": [0.3239, 0.4811, 0.5042, 0.5427],
            "Accuracy_err": [0.1190, 0.0389, 0.0331, 0.0261],
            "Precision": [0.3079, 0.3640, 0.3890, 0.4226],
            "Precision_err": [0.0455, 0.0274, 0.0184, 0.0276],
            "Recall": [0.2915, 0.3881, 0.4102, 0.4183],
            "Recall_err": [0.0488, 0.0329, 0.0274, 0.0226],
        },
        "DenseNet121": {
            "Accuracy": [0.3665, 0.5454, 0.5958, 0.6387],
            "Accuracy_err": [0.1098, 0.0779, 0.0361, 0.0199],
            "Precision": [0.3016, 0.3947, 0.4491, 0.4808],
            "Precision_err": [0.0345, 0.0423, 0.0355, 0.0196],
            "Recall": [0.2878, 0.3805, 0.4691, 0.5111],
            "Recall_err": [0.0390, 0.0377, 0.0305, 0.0241],
        },
        "EfficientNet-B0": {
            "Accuracy": [0.4296, 0.5421, 0.5845, 0.6239],
            "Accuracy_err": [0.1065, 0.0441, 0.0540, 0.0224],
            "Precision": [0.2747, 0.3438, 0.4176, 0.4574],
            "Precision_err": [0.0567, 0.0481, 0.0288, 0.0210],
            "Recall": [0.2681, 0.3104, 0.4069, 0.4733],
            "Recall_err": [0.0425, 0.0375, 0.0259, 0.0199],
        },
        "ResNet18": {
            "Accuracy": [0.4059, 0.5761, 0.5521, 0.5926],
            "Accuracy_err": [0.1230, 0.0626, 0.0440, 0.0188],
            "Precision": [0.3154, 0.3704, 0.3982, 0.4216],
            "Precision_err": [0.0369, 0.0342, 0.0173, 0.0161],
            "Recall": [0.2953, 0.3481, 0.4138, 0.4304],
            "Recall_err": [0.0399, 0.0239, 0.0233, 0.0201],
        },
        "VGG16": {
            "Accuracy": [0.3669, 0.4350, 0.5827, 0.5573],
            "Accuracy_err": [0.1533, 0.1056, 0.0558, 0.0327],
            "Precision": [0.2604, 0.3565, 0.3801, 0.3660],
            "Precision_err": [0.0554, 0.0516, 0.0656, 0.0432],
            "Recall": [0.2710, 0.3249, 0.3684, 0.4015],
            "Recall_err": [0.0521, 0.0320, 0.0268, 0.0198],
        },
    },
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


def add_f1_to_row(row: dict[str, float | int | str]) -> dict[str, float | int | str]:
    precision = float(row["Precision"])
    recall = float(row["Recall"])
    precision_std = float(row["Precision_err"])
    recall_std = float(row["Recall_err"])
    denom = precision + recall

    if denom == 0.0:
        row["F1-Score"] = 0.0
        row["F1-Score_err"] = 0.0
        return row

    row["F1-Score"] = 2.0 * precision * recall / denom
    d_f1_d_precision = 2.0 * recall**2 / denom**2
    d_f1_d_recall = 2.0 * precision**2 / denom**2
    row["F1-Score_err"] = float(
        np.sqrt(
            (d_f1_d_precision * precision_std) ** 2
            + (d_f1_d_recall * recall_std) ** 2
        )
    )
    return row


def result_dir_for(model_name: str, dataset_key: str) -> Path:
    template = str(MODEL_SOURCES[model_name])
    return Path(template.format(dataset_key=dataset_key))


def load_summary_metric_row(summary_path: Path, model_name: str, shot: int) -> dict[str, float | int | str]:
    payload = json.loads(summary_path.read_text())
    summary = payload["summary"]
    return add_f1_to_row({
        "Shot": shot,
        "Model": model_name,
        "Accuracy": float(summary["accuracy_mean"]),
        "Accuracy_err": float(summary["accuracy_std"]),
        "Precision": float(summary["precision_mean"]),
        "Precision_err": float(summary["precision_std"]),
        "Recall": float(summary["recall_mean"]),
        "Recall_err": float(summary["recall_std"]),
    })


def load_legacy_single_backbone_row(dataset_key: str, model_name: str, shot: int) -> dict[str, float | int | str]:
    values = LEGACY_SINGLE_BACKBONE_METRICS[dataset_key][model_name]
    idx = SHOTS.index(shot)
    return add_f1_to_row({
        "Shot": shot,
        "Model": model_name,
        "Accuracy": values["Accuracy"][idx],
        "Accuracy_err": values["Accuracy_err"][idx],
        "Precision": values["Precision"][idx],
        "Precision_err": values["Precision_err"][idx],
        "Recall": values["Recall"][idx],
        "Recall_err": values["Recall_err"][idx],
    })


def build_summary_df(dataset_key: str, dataset_label: str) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        for shot in SHOTS:
            if (
                dataset_key in LEGACY_SINGLE_BACKBONE_METRICS
                and model_name in SINGLE_BACKBONE_MODEL_NAMES
            ):
                rows.append(load_legacy_single_backbone_row(dataset_key, model_name, shot))
                continue

            result_dir = result_dir_for(model_name, dataset_key)
            summary_path = result_dir / f"{shot}_shot_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Missing {dataset_label} summary: {summary_path}")
            rows.append(load_summary_metric_row(summary_path, model_name, shot))
    return pd.DataFrame(rows)


def plot_benchmark(datasets: dict[str, pd.DataFrame], output_path: Path) -> None:
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(
        len(datasets),
        len(METRICS),
        figsize=(20, 4 * len(datasets) + 1),
        sharey=False,
    )
    axes = np.atleast_2d(axes)

    for row_idx, (dataset_name, df) in enumerate(datasets.items()):
        for col_idx, metric in enumerate(METRICS):
            ax = axes[row_idx][col_idx]
            for model_name in MODEL_ORDER:
                sub = df[df["Model"] == model_name].sort_values("Shot")
                style = MODEL_STYLES[model_name]
                ax.errorbar(
                    sub["Shot"].to_numpy(),
                    sub[metric].to_numpy(),
                    yerr=sub[f"{metric}_err"].to_numpy(),
                    label=model_name,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=2,
                    markersize=7,
                    capsize=4,
                    capthick=1.5,
                    linestyle="-",
                    alpha=0.9,
                )

            if col_idx == 0:
                ax.set_ylabel(dataset_name, fontsize=12, fontweight="bold", labelpad=10)
            if row_idx == 0:
                ax.set_title(metric, fontsize=13, fontweight="bold", pad=10)

            ax.set_xlabel("Shots", fontsize=10)
            ax.set_xscale("log")
            ax.set_xticks(SHOTS)
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks(np.arange(0.0, 1.01, 0.1))

    legend_map = {}
    for ax_row in axes:
        for ax in ax_row:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_map[label] = handle

    fig.legend(
        list(legend_map.values()),
        list(legend_map.keys()),
        loc="lower center",
        ncol=4,
        fontsize=11,
        bbox_to_anchor=(0.5, -0.01),
        frameon=True,
        title="K-shot Models",
    )
    fig.suptitle("Few-Shot Benchmark Results", fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.08, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    datasets = {
        dataset_label: build_summary_df(dataset_key, dataset_label)
        for dataset_label, dataset_key in DATASETS.items()
    }
    plot_benchmark(datasets, PROJECT_ROOT / "benchmark_plots_macro_selected.png")
    plot_benchmark(datasets, PROJECT_ROOT / "benchmark_plots_macro_selected.pdf")
    # Keep the historical output name for existing paper/Typst references.
    plot_benchmark(datasets, PROJECT_ROOT / "benchmark_plots_macro.png")


if __name__ == "__main__":
    main()
