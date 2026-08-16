#!/usr/bin/env python3
"""Prepare the SAR-CLD original cotton leaf disease data as an ImageFolder split.

Expected raw input is the original, non-augmented dataset folder. The script
accepts either:

    <source-root>/<class_name>/*.jpg

or one/two archive wrapper directories above the class folders, for example:

    <source-root>/Original Dataset/<class_name>/*.jpg

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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def safe_class_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "unknown"


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def class_dirs_with_images(root: Path) -> list[Path]:
    class_dirs: list[Path] = []
    for candidate in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(is_image(p) for p in candidate.rglob("*")):
            class_dirs.append(candidate)
    return class_dirs


def candidate_dataset_roots(source_root: Path, expected_classes: int) -> list[Path]:
    roots: list[tuple[int, int, Path]] = []
    for root in [source_root, *sorted(p for p in source_root.rglob("*") if p.is_dir())]:
        try:
            depth = len(root.relative_to(source_root).parts)
        except ValueError:
            depth = 0
        # Handle archive wrappers without treating every nested image folder as
        # another candidate root.
        if depth > 4:
            continue
        class_count = len(class_dirs_with_images(root))
        if class_count:
            exact_bonus = 1 if class_count == expected_classes else 0
            roots.append((exact_bonus, -depth, root))

    roots.sort(reverse=True)
    return [root for _, _, root in roots]


def collect_from_class_folders(source_root: Path, expected_classes: int) -> tuple[Path, dict[str, list[Path]]]:
    for root in candidate_dataset_roots(source_root, expected_classes):
        by_class: dict[str, list[Path]] = {}
        raw_names: dict[str, str] = {}
        for class_dir in class_dirs_with_images(root):
            class_name = safe_class_name(class_dir.name)
            images = sorted(p for p in class_dir.rglob("*") if is_image(p))
            if not images:
                continue
            if class_name in by_class:
                by_class[class_name].extend(images)
                raw_names[class_name] = f"{raw_names[class_name]} | {class_dir.name}"
            else:
                by_class[class_name] = images
                raw_names[class_name] = class_dir.name

        if len(by_class) == expected_classes:
            return root, by_class

    found = []
    for root in candidate_dataset_roots(source_root, expected_classes)[:5]:
        found.append(
            {
                "root": str(root),
                "classes": [safe_class_name(p.name) for p in class_dirs_with_images(root)],
            }
        )
    raise ValueError(
        f"Could not find exactly {expected_classes} class folders with images under {source_root}. "
        f"Candidate roots/classes: {json.dumps(found, indent=2)}"
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


def split_images(images: list[Path], test_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    images = list(images)
    rng.shuffle(images)
    if len(images) < 2:
        raise ValueError("Each class needs at least two images to create train/test splits.")
    train_count = int(len(images) * (1.0 - test_ratio))
    train_count = max(1, min(train_count, len(images) - 1))
    return images[:train_count], images[train_count:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Raw original dataset folder.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/sar_cld_original_split"),
        help="Prepared ImageFolder output root.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-classes", type=int, default=7)
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

    detected_root, by_class = collect_from_class_folders(args.source_root, args.expected_classes)
    rng = random.Random(args.seed)
    summary: dict[str, dict[str, int | str]] = {}

    for class_name, images in sorted(by_class.items()):
        train_images, test_images = split_images(images, args.test_ratio, rng)

        for split_name, split_group in (("train", train_images), ("test", test_images)):
            split_root = args.output_root / split_name
            for src in split_group:
                dst = unique_destination(split_root, class_name, src)
                link_or_copy(src, dst, args.mode)

        summary[class_name] = {
            "train": len(train_images),
            "test": len(test_images),
            "total": len(images),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_root": str(args.source_root),
        "detected_dataset_root": str(detected_root),
        "output_root": str(args.output_root),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "classes": summary,
    }
    (args.output_root / "split_summary.json").write_text(json.dumps(payload, indent=2))

    print(f"Wrote SAR-CLD original ImageFolder split to: {args.output_root}")
    print(f"Detected class root: {detected_root}")
    print(f"Classes: {len(summary)}")
    print("class,train,test,total")
    for class_name, row in summary.items():
        print(f"{class_name},{row['train']},{row['test']},{row['total']}")
    print(
        "TOTAL,"
        f"{sum(int(v['train']) for v in summary.values())},"
        f"{sum(int(v['test']) for v in summary.values())},"
        f"{sum(int(v['total']) for v in summary.values())}"
    )


if __name__ == "__main__":
    main()
