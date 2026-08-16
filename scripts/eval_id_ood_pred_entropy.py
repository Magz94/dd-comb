#!/usr/bin/env python3
"""Evaluate ID-vs-OOD detection from predictive entropy.

Default experiment:
  ID  = ImageNet-100 validation split
  OOD = DeepWeeds held-out test split

The score is predictive entropy over the ImageNet-100 classifier output.
Higher entropy means "more OOD-like". Metrics are binary sample-level metrics:
OOD is the positive class.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ddcomb.config import FusionPipelineConfig
from ddcomb.data.datasets import DeepWeedsCSVDataset
from ddcomb.evaluation.metrics import (
    binary_auroc,
    fpr_at_tpr,
    uncertainty_predictive_entropy_from_logits,
)
from ddcomb.models.embedding_models import prepare_embedding_models
from ddcomb.runtime import (
    FUSION_CLASSIFIER_CKPT_PREFIX,
    FUSION_MODEL_CKPT_PREFIX,
    checkpoint_path,
)
from ddcomb.training.build import build_fusion_model


MODEL_ORDER = ["ResNet18", "VGG16", "DenseNet121", "EfficientNetB0", "ViT-B16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imagenet100-root", type=Path, default=Path("data/ImageNet100/ImageNet100"))
    parser.add_argument("--deepweeds-root", type=Path, default=Path("/datasets/DeepWeeds"))
    parser.add_argument("--deepweeds-subset", type=str, default="subset0")
    parser.add_argument("--keep-deepweeds-label8", action="store_true")
    parser.add_argument("--base-models-path", type=Path, default=Path("models/base"))
    parser.add_argument("--base-subfolder", type=str, default="acc85")
    parser.add_argument("--model-save-path", type=Path, default=Path("models"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-id-samples", type=int, default=None)
    parser.add_argument("--max-ood-samples", type=int, default=None)
    parser.add_argument("--output-csv", type=Path, default=Path("id_ood_img100_vs_deepweeds_pred_entropy.csv"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dirichlet-tags",
        nargs="+",
        default=[
            "attn_stage_a_dirichlet_linear_imagenet100_acc85",
            "attn_stage_a_dirichlet_imagenet_1_acc85",
            "attn_stage_a_dirichlet_imagenet_1",
        ],
        help="Candidate Stage-A Dirichlet fusion tags; the first existing checkpoint is used.",
    )
    parser.add_argument(
        "--baseline-tags",
        nargs="+",
        default=["attn_stage_a_softmax_linear_imagenet100_acc85_resnetOrder_v1"],
        help="Candidate Stage-A softmax fusion tags; the first existing checkpoint is used.",
    )
    parser.add_argument(
        "--prefusion-tags",
        nargs="+",
        default=[
            "attn_stage_a_dirichlet_linear_prefusion_uncertainty_imagenet100_acc85_resnet_vgg_densenet_effnet_vit"
        ],
        help="Candidate Stage-A pre-fusion Dirichlet fusion tags; the first existing checkpoint is used.",
    )
    return parser.parse_args()


def _base_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def _maybe_limit_dataset(dataset, max_samples: int | None):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    indices = list(range(max_samples))
    return torch.utils.data.Subset(dataset, indices)


def build_id_ood_loaders(args: argparse.Namespace):
    transform = _base_transform()
    id_root = args.imagenet100_root / "val"
    if not id_root.is_dir():
        raise FileNotFoundError(f"ImageNet-100 val split not found: {id_root}")
    id_dataset = datasets.ImageFolder(root=id_root, transform=transform)

    deepweeds_images = args.deepweeds_root / "images"
    deepweeds_csv = args.deepweeds_root / "labels" / f"test_{args.deepweeds_subset}.csv"
    if not deepweeds_csv.is_file():
        raise FileNotFoundError(f"DeepWeeds test CSV not found: {deepweeds_csv}")
    ood_dataset = DeepWeedsCSVDataset(
        images_dir=str(deepweeds_images),
        csv_path=str(deepweeds_csv),
        transform=transform,
        drop_labels=set() if args.keep_deepweeds_label8 else {8},
    )

    id_dataset = _maybe_limit_dataset(id_dataset, args.max_id_samples)
    ood_dataset = _maybe_limit_dataset(ood_dataset, args.max_ood_samples)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": max(args.num_workers, 0),
        "pin_memory": args.device.startswith("cuda"),
    }
    return (
        DataLoader(id_dataset, **loader_kwargs),
        DataLoader(ood_dataset, **loader_kwargs),
        len(id_dataset),
        len(ood_dataset),
    )


def load_base_embedding_models(args: argparse.Namespace):
    models_dict = prepare_embedding_models(
        load_path=str(args.base_models_path),
        subfolder=args.base_subfolder,
        num_classes=100,
        pretrained=False,
    )
    return {name: models_dict[name].to(args.device).eval() for name in MODEL_ORDER}


@torch.no_grad()
def collect_single_backbone_entropy(model, id_loader, ood_loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    scores = []
    labels = []
    for ood_label, loader in [(0, id_loader), (1, ood_loader)]:
        for images, _ in loader:
            logits = model(images.to(device))
            entropy = uncertainty_predictive_entropy_from_logits(logits).detach().cpu()
            scores.append(entropy)
            labels.append(torch.full_like(entropy, float(ood_label)))
    return torch.cat(scores), torch.cat(labels)


def _first_existing_fusion_tag(args: argparse.Namespace, candidates: Iterable[str]) -> str:
    for tag in candidates:
        fusion_ckpt = checkpoint_path(
            str(args.model_save_path),
            FUSION_MODEL_CKPT_PREFIX,
            tag,
            args.base_subfolder,
        )
        classifier_ckpt = checkpoint_path(
            str(args.model_save_path),
            FUSION_CLASSIFIER_CKPT_PREFIX,
            tag,
            args.base_subfolder,
        )
        if Path(fusion_ckpt).is_file() and Path(classifier_ckpt).is_file():
            return tag
    raise FileNotFoundError(
        "None of the candidate fusion checkpoints exists. Tried:\n"
        + "\n".join(str(tag) for tag in candidates)
    )


def build_stage_a_fusion_classifier(
    args: argparse.Namespace,
    base_models: dict[str, torch.nn.Module],
    *,
    tag: str,
    attention_method: str,
    use_pre_fusion_uncertainty: bool,
):
    cfg = FusionPipelineConfig(
        use_wandb=False,
        attention_method=attention_method,
        classifier_method="linear",
        pretrained=False,
        use_jkk=False,
        num_classes=100,
        base_models_path=str(args.base_models_path),
        model_save_path=str(args.model_save_path),
        subfolders=[args.base_subfolder],
        model_names=list(MODEL_ORDER),
        use_pre_fusion_uncertainty=use_pre_fusion_uncertainty,
    )
    models_list = [(name, base_models[name], None) for name in MODEL_ORDER]
    fusion_model, classifier = build_fusion_model(models_list, cfg=cfg)

    fusion_ckpt = checkpoint_path(
        str(args.model_save_path),
        FUSION_MODEL_CKPT_PREFIX,
        tag,
        args.base_subfolder,
    )
    classifier_ckpt = checkpoint_path(
        str(args.model_save_path),
        FUSION_CLASSIFIER_CKPT_PREFIX,
        tag,
        args.base_subfolder,
    )
    fusion_model.load_state_dict(torch.load(fusion_ckpt, map_location="cpu"), strict=True)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location="cpu"), strict=True)
    return classifier.to(args.device).eval()


@torch.no_grad()
def _batch_embeddings(images: torch.Tensor, base_models: dict[str, torch.nn.Module]) -> list[torch.Tensor]:
    return [base_models[name](images, return_embedding=True) for name in MODEL_ORDER]


@torch.no_grad()
def collect_fusion_entropy(classifier, base_models, id_loader, ood_loader, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    scores = []
    labels = []
    for ood_label, loader in [(0, id_loader), (1, ood_loader)]:
        for images, _ in loader:
            images = images.to(device)
            embeddings = _batch_embeddings(images, base_models)
            out = classifier(embeddings)
            logits = out["logits"] if isinstance(out, dict) else out
            entropy = uncertainty_predictive_entropy_from_logits(logits).detach().cpu()
            scores.append(entropy)
            labels.append(torch.full_like(entropy, float(ood_label)))
    return torch.cat(scores), torch.cat(labels)


def summarize_scores(model_name: str, model_type: str, scores: torch.Tensor, labels: torch.Tensor) -> dict[str, float | str]:
    id_scores = scores[labels == 0]
    ood_scores = scores[labels == 1]
    return {
        "model": model_name,
        "model_type": model_type,
        "id_samples": int(id_scores.numel()),
        "ood_samples": int(ood_scores.numel()),
        "auroc_id_ood_pred_entropy": float(binary_auroc(scores, labels)),
        "fpr95_id_ood_pred_entropy": float(fpr_at_tpr(scores, labels, target_tpr=0.95)),
        "id_entropy_mean": float(id_scores.mean().item()),
        "ood_entropy_mean": float(ood_scores.mean().item()),
        "id_entropy_std": float(id_scores.std(unbiased=False).item()),
        "ood_entropy_std": float(ood_scores.std(unbiased=False).item()),
    }


def main() -> None:
    args = parse_args()
    args.imagenet100_root = args.imagenet100_root.resolve()
    args.base_models_path = args.base_models_path.resolve()
    args.model_save_path = args.model_save_path.resolve()

    print(f"Device: {args.device}")
    print(f"ID root: {args.imagenet100_root / 'val'}")
    print(f"OOD root: {args.deepweeds_root}")
    id_loader, ood_loader, n_id, n_ood = build_id_ood_loaders(args)
    print(f"ID samples: {n_id}")
    print(f"OOD samples: {n_ood}")

    base_models = load_base_embedding_models(args)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_type",
        "stage_a_tag",
        "id_samples",
        "ood_samples",
        "auroc_id_ood_pred_entropy",
        "fpr95_id_ood_pred_entropy",
        "id_entropy_mean",
        "ood_entropy_mean",
        "id_entropy_std",
        "ood_entropy_std",
    ]

    def write_live_row(writer: csv.DictWriter, f, row: dict) -> None:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
        f.flush()
        os.fsync(f.fileno())
        print(f"[LIVE] Appended result to {args.output_csv}: {row['model']}", flush=True)

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())

        for name in MODEL_ORDER:
            print(f"[Single] {name}", flush=True)
            scores, labels = collect_single_backbone_entropy(base_models[name], id_loader, ood_loader, args.device)
            row = summarize_scores(name, "single_backbone", scores, labels)
            print(row, flush=True)
            write_live_row(writer, f, row)

        fusion_specs = [
            (
                "Fusion Baseline",
                "fusion_softmax",
                _first_existing_fusion_tag(args, args.baseline_tags),
                "softmax",
                False,
            ),
            (
                "Fusion + Dirichlet Attention",
                "fusion_dirichlet",
                _first_existing_fusion_tag(args, args.dirichlet_tags),
                "dirichlet",
                False,
            ),
            (
                "Fusion + Dirichlet + Pre-Fusion",
                "fusion_dirichlet_prefusion",
                _first_existing_fusion_tag(args, args.prefusion_tags),
                "dirichlet",
                True,
            ),
        ]

        for display_name, model_type, tag, attention_method, use_prefusion in fusion_specs:
            print(f"[Fusion] {display_name} from tag={tag}", flush=True)
            classifier = build_stage_a_fusion_classifier(
                args,
                base_models,
                tag=tag,
                attention_method=attention_method,
                use_pre_fusion_uncertainty=use_prefusion,
            )
            scores, labels = collect_fusion_entropy(classifier, base_models, id_loader, ood_loader, args.device)
            row = summarize_scores(display_name, model_type, scores, labels)
            row["stage_a_tag"] = tag
            print(row, flush=True)
            write_live_row(writer, f, row)

    print(f"Wrote final CSV: {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
