"""BDD100K semantic segmentation dataset.

Loads image + train_id mask pairs, maps 19 BDD100K classes to 4:
  Class 0: road (0, 1)
  Class 1: vehicle (13, 14, 15)
  Class 2: person (11, 12)
  Class 3: background (everything else, and 255 -> 255 ignore)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# BDD100K train_id -> our 5-class mapping
# 255 stays 255 (ignore in loss)
BDD_TO_OURS = {
    0: 0,  # road
    1: 1,  # sidewalk  ← THIS MUST BE 1, not 0
    2: 4,  # building -> other
    3: 4,  # wall -> other
    4: 4,  # fence -> other
    5: 4,  # pole -> other
    6: 4,  # traffic light -> other
    7: 4,  # traffic sign -> other
    8: 4,  # vegetation -> other
    9: 4,  # terrain -> other
    10: 4,  # sky -> other
    11: 3,  # person
    12: 3,  # rider
    13: 2,  # car -> vehicle
    14: 2,  # truck -> vehicle
    15: 2,  # bus -> vehicle
    16: 2,  # train -> vehicle
    17: 2,  # motorcycle -> vehicle
    18: 4,  # bicycle -> other
    255: 255,  # ignore
}

OUR_CLASSES = ["road", "sidewalk", "vehicle", "person", "other"]
NUM_CLASSES = len(OUR_CLASSES)
IGNORE_INDEX = 255

# road=0, sidewalk=1, vehicle=2, person=3, other=4


@dataclass(frozen=True)
class SegSplit:
    train_df: pd.DataFrame  # columns: image_path, mask_path
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    label_names: list[str]
    num_classes: int


def _build_pairs(image_dir: Path, mask_dir: Path, suffix: str = "_train_id") -> pd.DataFrame:
    """Match images with masks by stem."""
    image_files = {p.stem: p for p in image_dir.glob("*.jpg")}
    mask_files = {p.stem.replace(suffix, ""): p for p in mask_dir.glob("*.png")}

    rows = []
    missing = 0
    for stem, img_path in image_files.items():
        mask_path = mask_files.get(stem)
        if mask_path is None:
            missing += 1
            continue
        rows.append(
            {
                "image_path": str(img_path),
                "mask_path": str(mask_path),
            }
        )

    if missing:
        print(f"WARNING: {missing} images without matching masks")

    return pd.DataFrame(rows)


def prepare_segmentation(
    seg_root: Path,
    test_from_train_size: int = 1000,
    random_state: int = 42,
) -> SegSplit:
    """Split:
    Train: 6000 (official train, minus 1000 held out for test)
    Val:   1000 (official val)
    Test:  1000 (held out from official train)
    """
    train_pool_df = _build_pairs(
        seg_root / "images" / "train",
        seg_root / "labels" / "train",
    )
    val_df = _build_pairs(
        seg_root / "images" / "val",
        seg_root / "labels" / "val",
    )

    train_df, test_df = train_test_split(
        train_pool_df,
        test_size=test_from_train_size,
        random_state=random_state,
    )

    print(f"Train: {len(train_df)}")
    print(f"Val:   {len(val_df)}")
    print(f"Test:  {len(test_df)} (from official train)")

    return SegSplit(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        label_names=OUR_CLASSES,
        num_classes=NUM_CLASSES,
    )


def load_mask(mask_path: Path) -> np.ndarray:
    """Load a mask and map BDD100K class IDs to our 4-class scheme."""
    arr = np.array(Image.open(mask_path), dtype=np.uint8)
    out = np.full_like(arr, IGNORE_INDEX)
    for bdd_id, our_id in BDD_TO_OURS.items():
        out[arr == bdd_id] = our_id
    return out
