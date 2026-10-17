"""Inspect BDD100K image dataset after downloading."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from urbanintel.config import dataset_path
from urbanintel.data.bdd100k import prepare_bdd100k

LABEL_DIR = dataset_path("bdd100k", "labels")
IMAGES_ROOT = dataset_path("bdd100k", "images", "100k")


def main() -> None:
    split = prepare_bdd100k(
        train_json=LABEL_DIR / "bdd100k_labels_images_train.json",
        val_json=LABEL_DIR / "bdd100k_labels_images_val.json",
        train_images_dir=IMAGES_ROOT / "train",
        val_images_dir=IMAGES_ROOT / "val",
    )

    print()
    print("=== Class distribution (train) ===")
    print(split.train_df["scene"].value_counts())

    print()
    print("=== Class distribution (val) ===")
    print(split.val_df["scene"].value_counts())

    print()
    print("=== Class distribution (test) ===")
    print(split.test_df["scene"].value_counts())

    print()
    print("=== Sample image paths ===")
    for _, row in split.train_df.head(3).iterrows():
        print(f"  [{row['scene']}] {row['image_path']}")


if __name__ == "__main__":
    main()
