#!/usr/bin/env python3
"""Convert official BloodMNIST 224px .npz into ImageFolder splits.

Expected input is the official MedMNIST ``bloodmnist_224.npz`` file containing
``train_images``, ``train_labels``, ``val_images``, ``val_labels``,
``test_images``, and ``test_labels``. The output is compatible with torchvision
ImageFolder and this project's fusion/single-backbone loaders.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_DIRS = {
    0: "0_basophil",
    1: "1_eosinophil",
    2: "2_erythroblast",
    3: "3_immature_granulocytes",
    4: "4_lymphocyte",
    5: "5_monocyte",
    6: "6_neutrophil",
    7: "7_platelet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-npz",
        type=Path,
        default=Path("data/bloodmnist_raw/bloodmnist_224.npz"),
        help="Path to official bloodmnist_224.npz.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/bloodmnist_224_imagefolder"),
        help="Destination ImageFolder root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output root before writing.",
    )
    return parser.parse_args()


def _to_uint8_rgb(image: np.ndarray) -> Image.Image:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HxW, HxWx1, or HxWx3 image, got shape={arr.shape}")

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _write_split(
    *,
    split_name: str,
    images: np.ndarray,
    labels: np.ndarray,
    output_root: Path,
) -> dict[str, int]:
    split_root = output_root / split_name
    labels_flat = np.asarray(labels).reshape(-1).astype(int)
    if len(images) != len(labels_flat):
        raise ValueError(
            f"{split_name}: image/label count mismatch: {len(images)} images vs {len(labels_flat)} labels"
        )

    counts: Counter[str] = Counter()
    for idx, (image, label) in enumerate(zip(images, labels_flat)):
        if int(label) not in CLASS_DIRS:
            raise ValueError(f"{split_name}: unknown label {label} at index {idx}")
        class_dir = CLASS_DIRS[int(label)]
        dst_dir = split_root / class_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{split_name}_{idx:06d}.png"
        _to_uint8_rgb(image).save(dst)
        counts[class_dir] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    if not args.source_npz.is_file():
        raise FileNotFoundError(f"Missing source npz: {args.source_npz}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(args.output_root)

    data = np.load(args.source_npz)
    required = {
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise KeyError(f"Missing arrays in {args.source_npz}: {missing}")

    summary = {
        "source_npz": str(args.source_npz),
        "output_root": str(args.output_root),
        "classes": CLASS_DIRS,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        counts = _write_split(
            split_name=split,
            images=data[f"{split}_images"],
            labels=data[f"{split}_labels"],
            output_root=args.output_root,
        )
        summary["splits"][split] = {
            "num_images": int(sum(counts.values())),
            "class_counts": counts,
        }
        print(f"{split}: {sum(counts.values())} images")
        for class_name, count in counts.items():
            print(f"  {class_name}: {count}")

    summary_path = args.output_root / "bloodmnist_224_imagefolder_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
