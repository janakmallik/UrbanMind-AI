"""Inspect BDD100K segmentation data before training."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from urbanintel.config import dataset_path
from urbanintel.data.bdd100k_segmentation import (
    OUR_CLASSES,
    load_mask,
    prepare_segmentation,
)

SEG_ROOT = dataset_path("bdd100k", "seg")


def main() -> None:
    split = prepare_segmentation(SEG_ROOT)

    print(f"\nClasses: {split.label_names}")
    print(f"Num classes: {split.num_classes}\n")

    # Sample 50 train masks and count class distribution
    print("Sampling 50 train masks for class distribution ...")
    class_counts = Counter()
    total_pixels = 0
    for _, row in split.train_df.sample(50, random_state=42).iterrows():
        mask = load_mask(Path(row["mask_path"]))
        for c in range(split.num_classes):
            count = int((mask == c).sum())
            class_counts[c] += count
        total_pixels += mask.size

    print(f"\nClass distribution across 50 sampled masks ({total_pixels:,} pixels):")
    for c, name in enumerate(split.label_names):
        count = class_counts[c]
        pct = 100 * count / total_pixels
        print(f"  [{c}] {name:<12} {count:>10,} ({pct:5.2f}%)")

    # Specifically check sidewalk and person pixel share
    sidewalk_pixels = class_counts[1]
    person_pixels = class_counts[3]
    print(f"\nSpecific classes of interest:")
    print(f"  sidewalk: {sidewalk_pixels:,} pixels ({100 * sidewalk_pixels / total_pixels:.3f}%)")
    print(f"  person:   {person_pixels:,} pixels ({100 * person_pixels / total_pixels:.3f}%)")

    # Check mask shapes
    print("\nSample image/mask pairs (shape check):")
    from PIL import Image

    for _, row in split.train_df.head(3).iterrows():
        img = Image.open(row["image_path"])
        mask = Image.open(row["mask_path"])
        print(f"  {Path(row['image_path']).name}")
        print(f"    image: {img.size} {img.mode}")
        print(f"    mask:  {mask.size} {mask.mode}")


if __name__ == "__main__":
    main()
