#!/usr/bin/env python3
"""Prepare Cassava Leaf Disease Classification as an ImageFolder split.

Supported inputs:
  1. Kaggle competition layout:
       train.csv
       label_num_to_disease_map.json
       train_images/<image_id>.jpg
  2. Directory-per-class layout:
       Cassava___bacterial_blight/*.jpg
       Cassava___brown_streak_disease/*.jpg
       ...

Output layout:
    <output-root>/train/<class_name>/*.jpg
    <output-root>/test/<class_name>/*.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def safe_class_name(name: str) -> str:
    clean = name.replace("___", "_").replace("(", "").replace(")", "")
    clean = re.sub(r"[^A-Za-z0-9]+", "_", clean).strip("_")
    return clean or "unknown"


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_from_competition_layout(source_root: Path) -> dict[str, list[Path]] | None:
    train_csv = source_root / "train.csv"
    label_map_path = source_root / "label_num_to_disease_map.json"
    image_root = source_root / "train_images"
    if not (train_csv.is_file() and label_map_path.is_file() and image_root.is_dir()):
        return None

    label_map = json.loads(label_map_path.read_text())
    by_class: dict[str, list[Path]] = defaultdict(list)
    with train_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"].strip()
            label = row["label"].strip()
            class_name = safe_class_name(label_map[str(label)])
            image_path = image_root / image_id
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in train.csv not found: {image_path}")
            by_class[class_name].append(image_path)
    return dict(by_class)


def collect_from_class_folders(source_root: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        images = sorted(p for p in class_dir.rglob("*") if is_image(p))
        if not images:
            continue
        by_class[safe_class_name(class_dir.name)] = images
    if not by_class:
        raise ValueError(
            f"No class folders with images found under {source_root}. "
            "Expected either Kaggle competition layout or one folder per class."
        )
    return by_class


def collect_samples(source_root: Path) -> dict[str, list[Path]]:
    return collect_from_competition_layout(source_root) or collect_from_class_folders(source_root)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            # Cross-filesystem hardlinks are not always possible on clusters.
            os.symlink(src.resolve(), dst)
            return
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return
    raise ValueError(f"Unsupported mode: {mode}")


def unique_destination(root: Path, class_name: str, src: Path) -> Path:
    dst = root / class_name / src.name
    if not dst.exists():
        return dst
    stem = src.stem
    suffix = src.suffix
    counter = 1
    while True:
        candidate = root / class_name / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Raw Kaggle dataset folder.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/cassava_leaf_disease_split"),
        help="Prepared ImageFolder output root.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["symlink", "hardlink", "copy"],
        default="symlink",
        help="How to materialize files in the split.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {args.source_root}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output_root}. Use --overwrite to recreate it."
            )
        shutil.rmtree(args.output_root)

    by_class = collect_samples(args.source_root)
    rng = random.Random(args.seed)
    summary: dict[str, dict[str, int]] = {}

    for class_name, images in sorted(by_class.items()):
        images = list(images)
        rng.shuffle(images)
        train_count = int(len(images) * (1.0 - args.test_ratio))
        train_images = images[:train_count]
        test_images = images[train_count:]

        for split_name, split_images in (("train", train_images), ("test", test_images)):
            split_root = args.output_root / split_name
            for src in split_images:
                dst = unique_destination(split_root, class_name, src)
                link_or_copy(src, dst, args.mode)

        summary[class_name] = {"train": len(train_images), "test": len(test_images), "total": len(images)}

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "split_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote Cassava ImageFolder split to: {args.output_root}")
    print(f"Classes: {len(summary)}")
    print("class,train,test,total")
    for class_name, row in summary.items():
        print(f"{class_name},{row['train']},{row['test']},{row['total']}")
    print(
        "TOTAL,"
        f"{sum(v['train'] for v in summary.values())},"
        f"{sum(v['test'] for v in summary.values())},"
        f"{sum(v['total'] for v in summary.values())}"
    )


if __name__ == "__main__":
    main()
