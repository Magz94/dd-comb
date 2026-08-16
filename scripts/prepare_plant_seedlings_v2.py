#!/usr/bin/env python3
"""Prepare Plant Seedlings V2 as a fixed ImageFolder split.

Expected raw input is the official/re-hosted V2 nonsegmented single-plant
archive after extraction. The script accepts either:

    <source-root>/Black-grass/*.png
    <source-root>/Charlock/*.png
    ...

or a source root with one nested dataset directory containing those class
folders, for example:

    <source-root>/NonsegmentedV2/Black-grass/*.png

Output layout:

    <output-root>/train/<class_name>/*.png
    <output-root>/test/<class_name>/*.png
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

CLASS_ALIASES = {
    "blackgrass": "black_grass",
    "charlock": "charlock",
    "cleavers": "cleavers",
    "commonchickweed": "common_chickweed",
    "commonwheat": "common_wheat",
    "fathen": "fat_hen",
    "loosesilkybent": "loose_silky_bent",
    "maize": "maize",
    "scentlessmayweed": "scentless_mayweed",
    "shepherdspurse": "shepherds_purse",
    "smallfloweredcranesbill": "small_flowered_cranesbill",
    "sugarbeet": "sugar_beet",
}

EXPECTED_CLASSES = set(CLASS_ALIASES.values())


def compact_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def safe_class_name(name: str) -> str:
    compact = compact_name(name)
    if compact in CLASS_ALIASES:
        return CLASS_ALIASES[compact]
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def candidate_dataset_roots(source_root: Path) -> list[Path]:
    roots: list[Path] = []
    for root in [source_root, *sorted(p for p in source_root.rglob("*") if p.is_dir())]:
        # Keep the search shallow enough to avoid scanning every class image dir
        # as a candidate root while still handling one or two archive wrappers.
        try:
            depth = len(root.relative_to(source_root).parts)
        except ValueError:
            depth = 0
        if depth > 3:
            continue
        class_count = sum(1 for p in root.iterdir() if p.is_dir() and safe_class_name(p.name) in EXPECTED_CLASSES)
        if class_count >= 6:
            roots.append(root)
    return roots or [source_root]


def collect_from_class_folders(source_root: Path) -> dict[str, list[Path]]:
    for root in candidate_dataset_roots(source_root):
        by_class: dict[str, list[Path]] = {}
        for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            class_name = safe_class_name(class_dir.name)
            if class_name not in EXPECTED_CLASSES:
                continue
            images = sorted(p for p in class_dir.rglob("*") if is_image(p))
            if images:
                by_class[class_name] = images

        if by_class:
            missing = EXPECTED_CLASSES - set(by_class)
            extra = set(by_class) - EXPECTED_CLASSES
            if missing or extra:
                raise ValueError(
                    "Unexpected Plant Seedlings classes after normalization. "
                    f"Root={root}, missing={sorted(missing)}, extra={sorted(extra)}, "
                    f"found={sorted(by_class)}"
                )
            return by_class

    raise ValueError(
        f"No Plant Seedlings class folders with images found under {source_root}. "
        f"Expected classes: {sorted(EXPECTED_CLASSES)}"
    )


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
    parser.add_argument("--source-root", type=Path, required=True, help="Raw extracted V2 folder or its parent.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/plant_seedlings_v2_split"),
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
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError(f"--test-ratio must be in (0, 1), got {args.test_ratio}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {args.output_root}. Use --overwrite to recreate it."
            )
        shutil.rmtree(args.output_root)

    by_class = collect_from_class_folders(args.source_root)
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

    print(f"Wrote Plant Seedlings V2 ImageFolder split to: {args.output_root}")
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
