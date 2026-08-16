#!/usr/bin/env python3
"""Prepare LeafSugar / Sugarcane leaf disease data as an ImageFolder split.

Expected raw input layout:
    <source-root>/healthy/*.jpg
    <source-root>/mosaic/*.jpg
    <source-root>/redrot/*.jpg
    <source-root>/rust/*.jpg
    <source-root>/yellow/*.jpg

Output layout:
    <output-root>/train/<class_name>/*.jpg
    <output-root>/test/<class_name>/*.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = {"healthy", "mosaic", "redrot", "rust", "yellow"}


def safe_class_name(name: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    compact = clean.replace("_", "")
    if "healthy" in compact:
        return "healthy"
    if "mosaic" in compact:
        return "mosaic"
    if "redrot" in compact or ("red" in compact and "rot" in compact):
        return "redrot"
    if "rust" in compact:
        return "rust"
    if "yellow" in compact:
        return "yellow"
    return clean or "unknown"


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_from_class_folders(source_root: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        class_name = safe_class_name(class_dir.name)
        images = sorted(p for p in class_dir.rglob("*") if is_image(p))
        if images:
            by_class[class_name] = images

    if not by_class:
        raise ValueError(
            f"No class folders with images found under {source_root}. "
            "Expected folders: healthy, mosaic, redrot, rust, yellow."
        )

    missing = EXPECTED_CLASSES - set(by_class)
    extra = set(by_class) - EXPECTED_CLASSES
    if missing or extra:
        raise ValueError(
            "Unexpected LeafSugar classes after normalization. "
            f"Missing={sorted(missing)}, extra={sorted(extra)}, found={sorted(by_class)}"
        )
    return by_class


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
    parser.add_argument("--source-root", type=Path, required=True, help="Raw folder containing class subfolders.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/leafsugar_disease_split"),
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

    print(f"Wrote LeafSugar ImageFolder split to: {args.output_root}")
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
