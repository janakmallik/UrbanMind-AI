"""BDD100K image dataset loader for scene classification.

Handles the Kaggle mirror's nested subfolder structure
(train/test split into A/B subfolders) by recursively scanning.

Only the 3 viable scene classes are used: city street, highway, residential.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

VIABLE_CLASSES = ["city street", "highway", "residential"]
LABEL_TO_ID = {name: i for i, name in enumerate(VIABLE_CLASSES)}


@dataclass(frozen=True)
class BDDSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    label_names: list[str]


def _index_images(images_root: Path) -> dict[str, Path]:
    """Recursively find all .jpg files. Returns {filename: full_path}."""
    index: dict[str, Path] = {}
    for jpg in images_root.rglob("*.jpg"):
        index[jpg.name] = jpg
    return index


def load_labels(label_json: Path, image_index: dict[str, Path]) -> pd.DataFrame:
    """Load a BDD100K label JSON and join with image paths."""
    with open(label_json, "r") as f:
        records = json.load(f)

    rows = []
    missing = 0
    for record in records:
        scene = record.get("attributes", {}).get("scene")
        if scene not in VIABLE_CLASSES:
            continue
        name = record.get("name")
        path = image_index.get(name)
        if path is None:
            missing += 1
            continue
        rows.append(
            {
                "image_name": name,
                "image_path": str(path),
                "scene": scene,
                "label_id": LABEL_TO_ID[scene],
            }
        )

    if missing:
        print(f"WARNING: {missing} labeled images not found on disk")

    return pd.DataFrame(rows)


def prepare_bdd100k(
    train_json: Path,
    val_json: Path,
    train_images_dir: Path,
    val_images_dir: Path,
    val_test_split: float = 0.5,
    random_state: int = 42,
) -> BDDSplit:
    """Build the 3-class train/val/test split."""
    from sklearn.model_selection import train_test_split

    print(f"Indexing train images under {train_images_dir} ...")
    train_index = _index_images(train_images_dir)
    print(f"  Found {len(train_index):,} images")

    print(f"Indexing val images under {val_images_dir} ...")
    val_index = _index_images(val_images_dir)
    print(f"  Found {len(val_index):,} images")

    print("Loading train labels ...")
    train_df = load_labels(train_json, train_index)
    print(f"  Filtered to {len(train_df):,} rows in 3 classes")

    print("Loading val labels ...")
    val_pool_df = load_labels(val_json, val_index)
    print(f"  Filtered to {len(val_pool_df):,} rows in 3 classes")

    val_df, test_df = train_test_split(
        val_pool_df,
        test_size=val_test_split,
        random_state=random_state,
        stratify=val_pool_df["label_id"],
    )

    print()
    print("=== Split summary ===")
    print(f"Train: {len(train_df):,}")
    print(f"Val:   {len(val_df):,}")
    print(f"Test:  {len(test_df):,}")

    return BDDSplit(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        label_names=VIABLE_CLASSES,
    )
