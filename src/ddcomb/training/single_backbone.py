#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from ddcomb.evaluation.metrics import (
    avu_means_from_predictive_entropy_logits,
    uncertainty_predictive_entropy_from_logits,
    binary_auroc,
    roc_curve_binary,
    fpr_at_tpr,
)
from ddcomb.evaluation.plots import plot_roc_curve
from ddcomb.models.embedding_models import EmbeddingModel, prepare_embedding_models


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeepWeedsCSVDataset(Dataset):
    """Image classification dataset from DeepWeeds CSV labels."""

    def __init__(
        self,
        images_dir: Path,
        csv_path: Path,
        transform=None,
        drop_labels: Optional[set[int]] = None,
    ):
        self.images_dir = images_dir
        self.csv_path = csv_path
        self.transform = transform
        self.drop_labels = drop_labels or set()
        self.samples: list[tuple[str, int]] = []

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row["Filename"].strip()
                label = int(row["Label"])
                if label in self.drop_labels:
                    continue
                self.samples.append((filename, label))

        self._basename_to_path: Dict[str, str] = {}
        for root, _, files in os.walk(images_dir):
            for fn in files:
                self._basename_to_path.setdefault(fn, os.path.join(root, fn))

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, filename: str) -> str:
        direct = self.images_dir / filename
        if direct.exists():
            return str(direct)
        found = self._basename_to_path.get(os.path.basename(filename))
        if found is not None:
            return found
        raise FileNotFoundError(
            f"Image '{filename}' listed in '{self.csv_path}' not found under '{self.images_dir}'."
        )

    def __getitem__(self, idx: int):
        filename, label = self.samples[idx]
        image = Image.open(self._resolve_image_path(filename)).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: DataLoader
    num_classes: int


def make_grad_scaler(use_amp: bool):
    # PyTorch >=2.6 prefers torch.amp.GradScaler("cuda", ...).
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def safe_torch_load(path, map_location="cpu"):
    # PyTorch 2.6 changed torch.load default to weights_only=True.
    # Our own checkpoints include metadata (e.g., pathlib.Path in args), so we need weights_only=False.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Backward compatibility for older PyTorch versions without weights_only arg.
        return torch.load(path, map_location=map_location)


def _load_kshot_manifest(manifest_path: Optional[Path]) -> Optional[Dict[str, object]]:
    if manifest_path is None:
        return None
    with Path(manifest_path).open("r") as f:
        manifest = json.load(f)
    selected = manifest.get("selected")
    if not isinstance(selected, list) or len(selected) == 0:
        raise ValueError(
            f"k-shot manifest '{manifest_path}' must contain non-empty list field 'selected'."
        )
    for i, item in enumerate(selected):
        if not isinstance(item, dict) or "id" not in item:
            raise ValueError(
                f"k-shot manifest '{manifest_path}' has invalid selected[{i}] entry: {item}"
            )
    return manifest


def _filter_imagefolder_train_dataset(
    train_ds: datasets.ImageFolder,
    dataset_root: Path,
    selected_relpaths: set[str],
) -> None:
    filtered_samples = []
    found_relpaths = set()
    for path, label in train_ds.samples:
        relpath = os.path.relpath(path, dataset_root).replace("\\", "/")
        if relpath in selected_relpaths:
            filtered_samples.append((path, label))
            found_relpaths.add(relpath)
    missing = sorted(selected_relpaths - found_relpaths)
    if missing:
        preview = missing[:10]
        raise ValueError(
            "k-shot manifest references train samples not found in ImageFolder train pool.\n"
            f"Missing count: {len(missing)}\nFirst missing: {preview}"
        )
    train_ds.samples = filtered_samples
    train_ds.targets = [label for _, label in filtered_samples]
    if hasattr(train_ds, "imgs"):
        train_ds.imgs = train_ds.samples


def _filter_deepweeds_train_dataset(
    train_ds: DeepWeedsCSVDataset,
    selected_filenames: set[str],
) -> None:
    filtered_samples = []
    found_filenames = set()
    for filename, label in train_ds.samples:
        if filename in selected_filenames:
            filtered_samples.append((filename, label))
            found_filenames.add(filename)
    missing = sorted(selected_filenames - found_filenames)
    if missing:
        preview = missing[:10]
        raise ValueError(
            "k-shot manifest references train filenames not found in DeepWeeds train pool.\n"
            f"Missing count: {len(missing)}\nFirst missing: {preview}"
        )
    train_ds.samples = filtered_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate one ImageNet1k-pretrained backbone (no fusion)."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "vgg16", "densenet121", "efficientnet_b0", "vit_b16", "convnext_tiny"],
        help="Backbone to train/evaluate.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "imagenet100",
            "deepweeds_no_bg",
            "plant_vs_meadow",
            "ninco_ood_split",
            "pathmnist_224",
            "bloodmnist_224",
            "aptos2019",
            "plantvillage",
            "cassava_leaf_disease",
            "leafsugar_disease",
            "plant_seedlings_v2",
            "sar_cld_original",
        ],
        help="Dataset selection.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["full_finetune", "head_only"],
        help="Training mode.",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="basic",
        choices=["basic", "weights"],
        help="Input preprocessing: basic=Resize+ToTensor, weights=torchvision weights.transforms().",
    )

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    # HPC-safe default: avoid worker forking issues (e.g., UCX segfaults).
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision when CUDA is available.")
    parser.add_argument("--merge-val-into-train", action="store_true")
    parser.add_argument(
        "--enable-uncertainty-metrics",
        action="store_true",
        help="Compute AVU-style uncertainty metrics: P(A|C), P(U|I), P(I|U), and AvU.",
    )

    parser.add_argument("--deepweeds-root", type=Path, default=Path("/datasets/DeepWeeds"))
    parser.add_argument("--deepweeds-subset", type=str, default="subset0")
    parser.add_argument("--imagenet100-root", type=Path, default=Path("/home/magz94/ddcomb/data/ImageNet100/ImageNet100"))
    parser.add_argument("--ssl-patches-root", type=Path, default=Path("/home/magz94/Uncertainty/ssl_patches"))
    parser.add_argument("--ninco-ood-root", type=Path, default=Path("/home/magz94/ddcomb/data/ninco_ood_split"))
    parser.add_argument("--pathmnist-root", type=Path, default=Path("/home/magz94/ddcomb/data/pathmnist_224_imagefolder"))
    parser.add_argument("--bloodmnist-root", type=Path, default=Path("/home/magz94/ddcomb/data/bloodmnist_224_imagefolder"))
    parser.add_argument("--aptos-root", type=Path, default=Path("/home/magz94/ddcomb/data/aptos2019_split"))
    parser.add_argument("--plantvillage-root", type=Path, default=Path("/home/magz94/ddcomb/data/plantvillage_color_split"))
    parser.add_argument("--cassava-root", type=Path, default=Path("/home/magz94/ddcomb/data/cassava_leaf_disease_split"))
    parser.add_argument("--leafsugar-root", type=Path, default=Path("/home/magz94/ddcomb/data/leafsugar_disease_split"))
    parser.add_argument("--plant-seedlings-root", type=Path, default=Path("/home/magz94/ddcomb/data/plant_seedlings_v2_split"))
    parser.add_argument("--sar-cld-root", type=Path, default=Path("/home/magz94/ddcomb/data/sar_cld_original_split"))

    parser.add_argument("--output-dir", type=Path, default=Path("models/single_backbone"))
    parser.add_argument(
        "--export-base-model-path",
        type=Path,
        default=None,
        help=(
            "Optional path for exporting the trained model as an EmbeddingModel state_dict, "
            "for example models/base/acc85/convnext_tiny_imgnet100.pt."
        ),
    )
    parser.add_argument(
        "--init-base-model-path",
        type=Path,
        default=None,
        help=(
            "Optional single EmbeddingModel checkpoint to initialise from, "
            "for example models/base/acc85/vgg16_imgnet100.pt."
        ),
    )
    parser.add_argument(
        "--init-base-models-path",
        type=Path,
        default=None,
        help=(
            "Optional base checkpoint root to initialise from, for example models/base. "
            "Use with --init-base-subfolder acc85."
        ),
    )
    parser.add_argument("--init-base-subfolder", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--kshot-manifest-path",
        type=Path,
        default=None,
        help="Optional path to k-shot manifest JSON for filtering train split.",
    )
    parser.add_argument(
        "--prediction-output-path",
        type=Path,
        default=None,
        help="Optional .npz path for held-out labels/probabilities/logits after eval.",
    )

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="single_backbone_finetune")
    parser.add_argument("--wandb-entity", type=str, default=None)

    return parser.parse_args()


def resolve_deepweeds_csv(labels_dir: Path, subset: str, split: str) -> Path:
    candidates = [
        labels_dir / f"{split}_{subset}.csv",
        labels_dir / subset / f"{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    options = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find DeepWeeds '{split}' CSV.\nChecked:\n{options}"
    )


def get_weights_for_model(model_name: str):
    if model_name == "resnet18":
        return models.ResNet18_Weights.IMAGENET1K_V1
    if model_name == "vgg16":
        return models.VGG16_Weights.IMAGENET1K_V1
    if model_name == "densenet121":
        return models.DenseNet121_Weights.IMAGENET1K_V1
    if model_name == "efficientnet_b0":
        return models.EfficientNet_B0_Weights.IMAGENET1K_V1
    if model_name == "vit_b16":
        return models.ViT_B_16_Weights.IMAGENET1K_V1
    if model_name == "convnext_tiny":
        return models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    raise ValueError(f"Unsupported model: {model_name}")


def build_input_transform(preprocess: str, weights):
    if preprocess == "weights":
        return weights.transforms()
    if preprocess == "basic":
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )
    raise ValueError(f"Unsupported preprocess mode: {preprocess}")


def build_model(model_name: str, num_classes: int, weights):
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head = model.fc
    elif model_name == "vgg16":
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        head = model.classifier[6]
    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        head = model.classifier
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head = model.classifier[1]
    elif model_name == "vit_b16":
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        head = model.heads.head
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        head = model.classifier[2]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, head


def build_base_initialized_model(args: argparse.Namespace, num_classes: int) -> tuple[nn.Module, nn.Module, Optional[str]]:
    """Build a single-backbone model from fusion-style /models/base checkpoints.

    Those checkpoints are saved as EmbeddingModel state_dicts. This keeps the
    feature extractor initialisation identical to the fusion pipeline while
    replacing/skipping the ImageNet-100 classifier when the target class count
    differs.
    """
    init_base_model_path = getattr(args, "init_base_model_path", None)
    init_base_models_path = getattr(args, "init_base_models_path", None)

    if init_base_model_path is None and init_base_models_path is None:
        return None, None, None
    if init_base_model_path is not None and init_base_models_path is not None:
        raise ValueError("Use either init_base_model_path or init_base_models_path, not both.")

    embedding_name, _ = embedding_export_meta(args.model)
    if init_base_model_path is not None:
        model = build_empty_embedding_model(args.model, num_classes=num_classes)
        model.load(str(init_base_model_path))
        return model, model.classifier, str(init_base_model_path)

    models_by_name = prepare_embedding_models(
        load_path=str(init_base_models_path),
        subfolder=getattr(args, "init_base_subfolder", None),
        num_classes=num_classes,
        pretrained=False,
    )
    if embedding_name not in models_by_name:
        raise ValueError(
            f"Could not build base-initialized model '{embedding_name}'. "
            f"Available keys: {sorted(models_by_name.keys())}"
        )
    model = models_by_name[embedding_name]
    subfolder = getattr(args, "init_base_subfolder", None)
    source = str(Path(init_base_models_path) / subfolder) if subfolder else str(init_base_models_path)
    return model, model.classifier, source


def embedding_export_meta(model_name: str) -> tuple[str, int]:
    if model_name == "resnet18":
        return "ResNet18", 512
    if model_name == "vgg16":
        return "VGG16", 512
    if model_name == "densenet121":
        return "DenseNet121", 1024
    if model_name == "efficientnet_b0":
        return "EfficientNetB0", 1280
    if model_name == "vit_b16":
        return "ViT-B16", 768
    if model_name == "convnext_tiny":
        return "ConvNeXt-Tiny", 768
    raise ValueError(f"Unsupported model: {model_name}")


def build_empty_embedding_model(model_name: str, num_classes: int) -> EmbeddingModel:
    embedding_name, embedding_dim = embedding_export_meta(model_name)
    if model_name == "resnet18":
        base_model = models.resnet18(weights=None)
    elif model_name == "vgg16":
        base_model = models.vgg16(weights=None)
    elif model_name == "densenet121":
        base_model = models.densenet121(weights=None)
    elif model_name == "efficientnet_b0":
        base_model = models.efficientnet_b0(weights=None)
    elif model_name == "vit_b16":
        base_model = models.vit_b_16(weights=None)
        base_model.heads = nn.Identity()
    elif model_name == "convnext_tiny":
        base_model = models.convnext_tiny(weights=None)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return EmbeddingModel(
        base_model,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        name=embedding_name,
    )


def export_embedding_model_checkpoint(
    model_name: str,
    trained_model: nn.Module,
    trained_head: nn.Module,
    num_classes: int,
    export_path: Path,
) -> None:
    if isinstance(trained_model, EmbeddingModel):
        export_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trained_model.state_dict(), export_path)
        print(f"Exported {trained_model.name} EmbeddingModel checkpoint to: {export_path}")
        return

    embedding_name, embedding_dim = embedding_export_meta(model_name)
    if model_name == "vit_b16":
        # The EmbeddingModel wrapper expects ViT to emit embeddings; the classifier
        # is stored separately in wrapper.classifier.
        trained_model.heads = nn.Identity()
    export_model = EmbeddingModel(
        trained_model,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        name=embedding_name,
    )
    export_model.classifier.load_state_dict(trained_head.state_dict())
    export_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(export_model.state_dict(), export_path)
    print(f"Exported {embedding_name} EmbeddingModel checkpoint to: {export_path}")


def configure_trainable_params(model: nn.Module, head: nn.Module, mode: str) -> None:
    if mode == "full_finetune":
        for p in model.parameters():
            p.requires_grad = True
        return

    if mode == "head_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
        return

    raise ValueError(f"Unsupported mode: {mode}")


def make_dataloaders(args: argparse.Namespace, transform) -> DataBundle:
    manifest = _load_kshot_manifest(getattr(args, "kshot_manifest_path", None))
    kshot_selected_relpaths: Optional[set[str]] = None
    kshot_selected_filenames: Optional[set[str]] = None
    if manifest is not None:
        dataset_type = str(manifest.get("dataset_type", "")).strip().lower()
        selected_ids = [str(item["id"]) for item in manifest["selected"]]
        if dataset_type == "imagefolder":
            kshot_selected_relpaths = set(selected_ids)
        elif dataset_type == "deepweeds_csv":
            kshot_selected_filenames = set(selected_ids)
        else:
            raise ValueError(
                f"Unknown dataset_type='{manifest.get('dataset_type')}' in k-shot manifest "
                f"'{args.kshot_manifest_path}'. Use 'imagefolder' or 'deepweeds_csv'."
            )

    if args.dataset == "imagenet100":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='imagenet100' (ImageFolder)."
            )
        train_root = args.imagenet100_root / "train"
        test_root = args.imagenet100_root / "val"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing val/test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.imagenet100_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(train_ds.classes)
        if num_classes != 100:
            raise ValueError(
                f"imagenet100 expects 100 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "plant_vs_meadow":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='plant_vs_meadow' (ImageFolder)."
            )
        train_root = args.ssl_patches_root / "train"
        test_root = args.ssl_patches_root / "val"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing val/test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.ssl_patches_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(train_ds.classes)
        if num_classes != 2:
            raise ValueError(
                f"plant_vs_meadow expects 2 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "ninco_ood_split":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='ninco_ood_split' (ImageFolder)."
            )
        train_root = args.ninco_ood_root / "train"
        test_root = args.ninco_ood_root / "val"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing val/test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.ninco_ood_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(train_ds.classes)
        if num_classes != 64:
            raise ValueError(
                f"ninco_ood_split expects 64 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "pathmnist_224":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='pathmnist_224' (ImageFolder)."
            )
        train_root = args.pathmnist_root / "train"
        val_root = args.pathmnist_root / "val"
        test_root = args.pathmnist_root / "test"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not val_root.exists():
            raise FileNotFoundError(f"Missing val folder: {val_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if args.merge_val_into_train:
            val_ds_for_merge = datasets.ImageFolder(root=str(val_root), transform=transform)
            if train_ds.class_to_idx != val_ds_for_merge.class_to_idx:
                raise ValueError(
                    "Cannot merge pathmnist_224 train+val with different class mappings.\n"
                    f"train classes={train_ds.class_to_idx}\n"
                    f"val classes={val_ds_for_merge.class_to_idx}"
                )
            train_ds.samples = list(train_ds.samples) + list(val_ds_for_merge.samples)
            train_ds.targets = list(train_ds.targets) + list(val_ds_for_merge.targets)
            if hasattr(train_ds, "imgs"):
                train_ds.imgs = train_ds.samples
            print(f"pathmnist_224 merge enabled: appended val into train ({len(val_ds_for_merge)} samples).")
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.pathmnist_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(datasets.ImageFolder(root=str(train_root)).classes)
        if num_classes != 9:
            raise ValueError(
                f"pathmnist_224 expects 9 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "bloodmnist_224":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='bloodmnist_224' (ImageFolder)."
            )
        train_root = args.bloodmnist_root / "train"
        val_root = args.bloodmnist_root / "val"
        test_root = args.bloodmnist_root / "test"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not val_root.exists():
            raise FileNotFoundError(f"Missing val folder: {val_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if args.merge_val_into_train:
            val_ds_for_merge = datasets.ImageFolder(root=str(val_root), transform=transform)
            if train_ds.class_to_idx != val_ds_for_merge.class_to_idx:
                raise ValueError(
                    "Cannot merge bloodmnist_224 train+val with different class mappings.\n"
                    f"train classes={train_ds.class_to_idx}\n"
                    f"val classes={val_ds_for_merge.class_to_idx}"
                )
            train_ds.samples = list(train_ds.samples) + list(val_ds_for_merge.samples)
            train_ds.targets = list(train_ds.targets) + list(val_ds_for_merge.targets)
            if hasattr(train_ds, "imgs"):
                train_ds.imgs = train_ds.samples
            print(f"bloodmnist_224 merge enabled: appended val into train ({len(val_ds_for_merge)} samples).")
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.bloodmnist_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(datasets.ImageFolder(root=str(train_root)).classes)
        if num_classes != 8:
            raise ValueError(
                f"bloodmnist_224 expects 8 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "aptos2019":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='aptos2019' (ImageFolder)."
            )
        train_root = args.aptos_root / "train"
        test_root = args.aptos_root / "val"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing val/test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.aptos_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(train_ds.classes)
        if num_classes != 5:
            raise ValueError(
                f"aptos2019 expects 5 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "plantvillage":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='plantvillage' (ImageFolder)."
            )
        train_root = args.plantvillage_root / "train"
        test_root = args.plantvillage_root / "test"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.plantvillage_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(train_ds.classes)
        if num_classes != 38:
            raise ValueError(
                f"plantvillage expects 38 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "cassava_leaf_disease":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='cassava_leaf_disease' (ImageFolder)."
            )
        train_root = args.cassava_root / "train"
        test_root = args.cassava_root / "test"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.cassava_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(train_ds.classes)
        if num_classes != 5:
            raise ValueError(
                f"cassava_leaf_disease expects 5 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "leafsugar_disease":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='leafsugar_disease' (ImageFolder)."
            )
        train_root = args.leafsugar_root / "train"
        test_root = args.leafsugar_root / "test"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.leafsugar_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(train_ds.classes)
        if num_classes != 5:
            raise ValueError(
                f"leafsugar_disease expects 5 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "plant_seedlings_v2":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='plant_seedlings_v2' (ImageFolder)."
            )
        train_root = args.plant_seedlings_root / "train"
        test_root = args.plant_seedlings_root / "test"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.plant_seedlings_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(train_ds.classes)
        if num_classes != 12:
            raise ValueError(
                f"plant_seedlings_v2 expects 12 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "sar_cld_original":
        if kshot_selected_filenames is not None:
            raise ValueError(
                "k-shot manifest dataset_type='deepweeds_csv' cannot be used with "
                "dataset='sar_cld_original' (ImageFolder)."
            )
        train_root = args.sar_cld_root / "train"
        test_root = args.sar_cld_root / "test"
        if not train_root.exists():
            raise FileNotFoundError(f"Missing train folder: {train_root}")
        if not test_root.exists():
            raise FileNotFoundError(f"Missing test folder: {test_root}")

        train_ds = datasets.ImageFolder(root=str(train_root), transform=transform)
        test_ds = datasets.ImageFolder(root=str(test_root), transform=transform)
        if kshot_selected_relpaths is not None:
            _filter_imagefolder_train_dataset(
                train_ds=train_ds,
                dataset_root=args.sar_cld_root,
                selected_relpaths=kshot_selected_relpaths,
            )
        num_classes = len(train_ds.classes)
        if num_classes != 7:
            raise ValueError(
                f"sar_cld_original expects 7 classes, found {num_classes} in {train_root}."
            )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=None,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    if args.dataset == "deepweeds_no_bg":
        if kshot_selected_relpaths is not None:
            raise ValueError(
                "k-shot manifest dataset_type='imagefolder' cannot be used with "
                "dataset='deepweeds_no_bg' (CSV dataset)."
            )
        images_dir = args.deepweeds_root / "images"
        labels_dir = args.deepweeds_root / "labels"
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing DeepWeeds images dir: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Missing DeepWeeds labels dir: {labels_dir}")

        train_csv = resolve_deepweeds_csv(labels_dir, args.deepweeds_subset, "train")
        val_csv = resolve_deepweeds_csv(labels_dir, args.deepweeds_subset, "val")
        test_csv = resolve_deepweeds_csv(labels_dir, args.deepweeds_subset, "test")

        drop_labels = {8}
        train_ds = DeepWeedsCSVDataset(images_dir=images_dir, csv_path=train_csv, transform=transform, drop_labels=drop_labels)
        val_ds = DeepWeedsCSVDataset(images_dir=images_dir, csv_path=val_csv, transform=transform, drop_labels=drop_labels)
        test_ds = DeepWeedsCSVDataset(images_dir=images_dir, csv_path=test_csv, transform=transform, drop_labels=drop_labels)

        if args.merge_val_into_train:
            train_ds.samples = train_ds.samples + val_ds.samples
            val_ds = None
        if kshot_selected_filenames is not None:
            _filter_deepweeds_train_dataset(
                train_ds=train_ds,
                selected_filenames=kshot_selected_filenames,
            )

        num_classes = 8
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return DataBundle(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=num_classes,
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")


def compute_ece(confidences: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> Tuple[float, float]:
    if confidences.numel() == 0:
        return 0.0, 0.0

    boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = float(confidences.numel())

    for i in range(n_bins):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        acc_bin = correct[mask].float().mean().item()
        conf_bin = confidences[mask].mean().item()
        gap = abs(acc_bin - conf_bin)
        frac = mask.float().mean().item()
        ece += gap * frac
        mce = max(mce, gap)

    return float(ece), float(mce)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    return_raw: bool = False,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    total_correct = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    all_conf = []
    all_correct = []
    all_logits = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)

        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        correct = preds.eq(labels)

        total += labels.size(0)
        total_correct += correct.sum().item()

        idx = labels.long() * num_classes + preds.long()
        bincount = torch.bincount(idx.cpu(), minlength=num_classes * num_classes)
        cm += bincount.reshape(num_classes, num_classes)

        all_conf.append(conf.detach().cpu())
        all_correct.append(correct.detach().cpu())
        if return_raw:
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    if total == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "ece": 0.0,
            "mce": 0.0,
        }

    tp = cm.diag().float()
    pred_pos = cm.sum(dim=0).float().clamp(min=1.0)
    gt_pos = cm.sum(dim=1).float().clamp(min=1.0)
    precision_macro = (tp / pred_pos).mean().item()
    recall_macro = (tp / gt_pos).mean().item()

    conf_all = torch.cat(all_conf) if all_conf else torch.tensor([])
    corr_all = torch.cat(all_correct) if all_correct else torch.tensor([], dtype=torch.bool)
    ece, mce = compute_ece(conf_all, corr_all)

    out = {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "precision": precision_macro,
        "recall": recall_macro,
        "ece": ece,
        "mce": mce,
    }
    if return_raw:
        out["logits"] = torch.cat(all_logits, dim=0) if all_logits else torch.empty((0, num_classes))
        out["labels"] = torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,), dtype=torch.long)
    return out


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total = 0
    total_correct = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                with torch.amp.autocast("cuda", enabled=True):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                with torch.cuda.amp.autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def default_lr_for_mode(mode: str) -> float:
    if mode in {"head_only", "full_finetune"}:
        # Match your fusion pipeline default unless explicitly overridden via --lr.
        return 1e-3
    raise ValueError(f"Unsupported mode: {mode}")


def build_optimizer(name: str, params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    normalized = name.lower()
    if normalized == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if normalized == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def maybe_init_wandb(args: argparse.Namespace, run_name: str, config: Dict):
    if not args.use_wandb:
        return None

    import wandb

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
    )
    return run


def run_experiment(args: argparse.Namespace) -> dict:
    set_seed(args.seed)

    run_name = args.run_name or f"{args.dataset}_{args.model}_{args.mode}"
    lr = args.lr if args.lr is not None else default_lr_for_mode(args.mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(use_amp)

    weights = get_weights_for_model(args.model)
    transform = build_input_transform(args.preprocess, weights)
    data = make_dataloaders(args, transform)

    model, head_module, init_base_source = build_base_initialized_model(args, num_classes=data.num_classes)
    if model is None:
        model, head_module = build_model(args.model, num_classes=data.num_classes, weights=weights)
        init_base_source = None
    else:
        print(f"[TrainConfig] Initialized from base checkpoint source: {init_base_source}")
    configure_trainable_params(model, head_module, args.mode)
    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check mode/head setup.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in trainable_params)
    head_param_count = sum(p.numel() for p in head_module.parameters())
    print(
        "[TrainConfig] "
        f"model={args.model} mode={args.mode} optimizer={args.optimizer} "
        f"lr={lr} weight_decay={args.weight_decay} preprocess={args.preprocess}"
    )
    print(
        "[TrainConfig] "
        f"trainable_params={trainable_param_count:,}/{total_params:,} "
        f"head_params={head_param_count:,}"
    )

    optimizer = build_optimizer(args.optimizer, trainable_params, lr=lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / "best.pt"
    last_ckpt = output_dir / "last.pt"

    config_payload = vars(args).copy()
    for path_key in (
        "kshot_manifest_path",
        "export_base_model_path",
        "init_base_model_path",
        "init_base_models_path",
    ):
        if config_payload.get(path_key) is not None:
            config_payload[path_key] = str(config_payload[path_key])
    config_payload["resolved_lr"] = lr
    config_payload["resolved_device"] = str(device)
    config_payload["num_classes"] = data.num_classes
    config_payload["init_base_source"] = init_base_source

    wandb_run = maybe_init_wandb(args, run_name, config_payload)

    best_score = float("nan")
    best_epoch = args.epochs
    best_val_metrics = None
    history = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=data.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
        }
        history.append(row)

        log_line = (
            f"[{epoch:03d}/{args.epochs:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
        )
        print(log_line)

        if wandb_run is not None:
            wandb_run.log(row)

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "resolved_lr": lr,
            "num_classes": data.num_classes,
        },
        last_ckpt,
    )
    # Train-then-test only protocol: best checkpoint equals final epoch checkpoint.
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "resolved_lr": lr,
            "num_classes": data.num_classes,
        },
        best_ckpt,
    )

    # Strict train-then-test protocol: evaluate final model once on held-out test.
    eval_ckpt = safe_torch_load(last_ckpt, map_location="cpu")
    selection_source = "final_epoch_train_only"

    model.load_state_dict(eval_ckpt["model_state_dict"], strict=True)
    model = model.to(device)

    test_eval = evaluate_model(
        model=model,
        loader=data.test_loader,
        criterion=criterion,
        device=device,
        num_classes=data.num_classes,
        return_raw=True,
    )
    test_metrics = {
        "loss": float(test_eval["loss"]),
        "accuracy": float(test_eval["accuracy"]),
        "precision": float(test_eval["precision"]),
        "recall": float(test_eval["recall"]),
        "ece": float(test_eval["ece"]),
        "mce": float(test_eval["mce"]),
    }
    test_logits = test_eval["logits"]
    test_labels = test_eval["labels"]
    test_preds = test_logits.argmax(dim=1) if test_logits.numel() > 0 else torch.empty((0,), dtype=torch.long)
    test_probs = torch.softmax(test_logits, dim=1) if test_logits.numel() > 0 else torch.empty((0, data.num_classes))
    prediction_output_path = getattr(args, "prediction_output_path", None)
    if prediction_output_path is not None:
        prediction_output_path = Path(prediction_output_path)
        prediction_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            prediction_output_path,
            y_true=test_labels.detach().cpu().numpy(),
            y_prob=test_probs.detach().cpu().numpy(),
            logits=test_logits.detach().cpu().numpy(),
            num_classes=np.array([data.num_classes], dtype=np.int64),
            model_type=np.array(["single_backbone"]),
            model_name=np.array([args.model]),
            run_name=np.array([run_name]),
        )
        print(f"[PRED] Wrote prediction probabilities: {prediction_output_path}")
    incorrect = 1.0 - (test_preds == test_labels).float()
    u_entropy = uncertainty_predictive_entropy_from_logits(test_logits)
    test_metrics["auroc_err_pred_entropy"] = float(binary_auroc(u_entropy, incorrect))
    test_metrics["fpr@95_err_pred_entropy"] = float(fpr_at_tpr(u_entropy, incorrect, target_tpr=0.95))
    if getattr(args, "enable_uncertainty_metrics", False):
        test_metrics.update(
            avu_means_from_predictive_entropy_logits(
                logits=test_logits,
                labels=test_labels,
                n_thresholds=101,
            )
        )
    roc_ent = roc_curve_binary(u_entropy, incorrect, n_thresholds=400)
    roc_test_path = output_dir / "roc_err_pred_entropy_test.png"
    plot_roc_curve(
        roc_ent,
        str(roc_test_path),
        title="ROC: Error detection (Predictive Entropy)",
    )

    exported_base_model_path = None
    if getattr(args, "export_base_model_path", None) is not None:
        export_embedding_model_checkpoint(
            model_name=args.model,
            trained_model=model,
            trained_head=head_module,
            num_classes=data.num_classes,
            export_path=args.export_base_model_path,
        )
        exported_base_model_path = str(args.export_base_model_path)

    if wandb_run is not None:
        import wandb
        wandb_log = {
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_ece": test_metrics["ece"],
            "test_mce": test_metrics["mce"],
            "auroc_err_pred_entropy": test_metrics["auroc_err_pred_entropy"],
            "fpr@95_err_pred_entropy": test_metrics["fpr@95_err_pred_entropy"],
            "roc_err_pred_entropy_plot": wandb.Image(str(roc_test_path)),
        }
        if getattr(args, "enable_uncertainty_metrics", False):
            wandb_log.update(
                {
                    "uncertainty/p_a_given_c_mean": test_metrics.get("p_a_given_c_mean"),
                    "uncertainty/p_u_given_i_mean": test_metrics.get("p_u_given_i_mean"),
                    "uncertainty/p_i_given_u_mean": test_metrics.get("p_i_given_u_mean"),
                    "uncertainty/avu_mean": test_metrics.get("avu_mean"),
                }
            )
        wandb_run.log(wandb_log)

    elapsed = time.time() - start
    summary = {
        "run_name": run_name,
        "dataset": args.dataset,
        "model": args.model,
        "mode": args.mode,
        "num_classes": data.num_classes,
        "init_base_source": init_base_source,
        "resolved_lr": lr,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_metrics": best_val_metrics,
        "selection_source": selection_source,
        "test_metrics": test_metrics,
        "checkpoints": {
            "best": str(best_ckpt),
            "last": str(last_ckpt),
        },
        "exported_base_model_path": exported_base_model_path,
        "elapsed_seconds": elapsed,
        "history": history,
    }
    if getattr(args, "kshot_manifest_path", None) is not None:
        summary["kshot_manifest_path"] = str(args.kshot_manifest_path)
    if prediction_output_path is not None:
        summary["prediction_output_path"] = str(prediction_output_path)

    summary_path = output_dir / "metrics_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved summary to: {summary_path}")

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "best_epoch": best_epoch,
                "best_score": best_score,
                "selection_source": selection_source,
                "test_accuracy": test_metrics.get("accuracy", None),
                "test_precision": test_metrics.get("precision", None),
                "test_recall": test_metrics.get("recall", None),
                "test_ece": test_metrics.get("ece", None),
                "test_mce": test_metrics.get("mce", None),
                "auroc_err_pred_entropy": test_metrics.get("auroc_err_pred_entropy", None),
                "fpr@95_err_pred_entropy": test_metrics.get("fpr@95_err_pred_entropy", None),
                "uncertainty/p_a_given_c_mean": test_metrics.get("p_a_given_c_mean", None),
                "uncertainty/p_u_given_i_mean": test_metrics.get("p_u_given_i_mean", None),
                "uncertainty/p_i_given_u_mean": test_metrics.get("p_i_given_u_mean", None),
                "uncertainty/avu_mean": test_metrics.get("avu_mean", None),
            }
        )
        wandb_run.finish()
    return summary


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
