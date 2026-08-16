#!/usr/bin/env python3
"""Prepare SSL_Patches plant-vs-meadow as an ImageFolder split.

Expected source layout:

    <source-root>/meadow_cropped/*.{jpg,jpeg,png,...}
    <source-root>/plant_cropped/*.{jpg,jpeg,png,...}

Output layout:

    <output-root>/train/meadow_cropped/*
    <output-root>/train/plant_cropped/*
    <output-root>/val/meadow_cropped/*
    <output-root>/val/plant_cropped/*

The split is stratified per class. By default, files are symlinked so the
prepared dataset is cheap to recreate. Use --mode copy to materialize files.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_CLASSES = ("meadow_cropped", "plant_cropped")


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_class_images(source_root: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {}
    for class_name in EXPECTED_CLASSES:
        class_root = source_root / class_name
        if not class_root.is_dir():
            raise FileNotFoundError(f"Missing expected class folder: {class_root}")
        images = sorted(p for p in class_root.rglob("*") if is_image(p))
        if not images:
            raise ValueError(f"No image files found in: {class_root}")
        by_class[class_name] = images
    return by_class


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            os.symlink(src.resolve(), dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def unique_destination(split_root: Path, class_name: str, src: Path) -> Path:
    dst = split_root / class_name / src.name
    if not dst.exists():
        return dst
    stem, suffix = src.stem, src.suffix
    counter = 1
    while True:
        candidate = split_root / class_name / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/datasets/SSL_Patches/Cropped_VariableSize"),
        help="Folder containing meadow_cropped and plant_cropped.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/ssl_patches"),
        help="Prepared ImageFolder output root.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation fraction per class.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to materialize files in the output split.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete output-root before recreating it.")
    args = parser.parse_args()

    if not args.source_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {args.source_root}")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError(f"--val-ratio must be in (0, 1), got {args.val_ratio}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output_root}. Use --overwrite to recreate it.")
        shutil.rmtree(args.output_root)

    by_class = collect_class_images(args.source_root)
    rng = random.Random(args.seed)
    summary: dict[str, dict[str, int]] = {}

    for class_name, images in by_class.items():
        images = list(images)
        rng.shuffle(images)
        val_count = round(len(images) * args.val_ratio)
        val_images = images[:val_count]
        train_images = images[val_count:]

        for split_name, split_images in (("train", train_images), ("val", val_images)):
            split_root = args.output_root / split_name
            for src in split_images:
                dst = unique_destination(split_root, class_name, src)
                link_or_copy(src, dst, args.mode)

        summary[class_name] = {
            "train": len(train_images),
            "val": len(val_images),
            "total": len(images),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote SSL_Patches ImageFolder split to: {args.output_root.resolve()}")
    print(f"Source: {args.source_root.resolve()}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print("class,train,val,total")
    for class_name, row in summary.items():
        print(f"{class_name},{row['train']},{row['val']},{row['total']}")
    print(
        "TOTAL,"
        f"{sum(row['train'] for row in summary.values())},"
        f"{sum(row['val'] for row in summary.values())},"
        f"{sum(row['total'] for row in summary.values())}"
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
