"""HMDB-51 video dataset loader for 15-class urban behavior taxonomy.

Classes: stand, walk, sit, eat, drink, hug, run, ride_bike,
         ride_horse, fall_floor, push, hit, punch, shoot_gun, smoke

Categories for per-category evaluation:
  normal: stand, walk, sit, eat, drink, hug (6)
  mobility: run, ride_bike (2)
  anomaly: ride_horse, fall_floor (2)  [horse in modern urban = rare/anomalous]
  threat: push, hit, punch, shoot_gun (4)
  restricted: smoke (1)  [contextual — restricted in transit, healthcare, schools]

Split 1 verified: 892 train / 158 val / 450 test (1,500 total).
All 15 classes have split files and appear in all splits.
Per-class balance confirmed (59-60 train, 10-11 val, 30 test per class).

Flat 15-class model with per-category reporting.
Not hierarchical: time/power budget favors single model + category analysis.

Official HMDB-51 split 1 files use:
  0 = ignore (excluded from this split)
  1 = train
  2 = test

For training, we further split official train into train + val (85/15)
for checkpoint selection. Test set untouched until final eval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

HMDB_ROOT = Path(
    os.environ.get(
        "URBANINTEL_HMDB51_ROOT",
        "C:/Users/Janak Mallik/ml-data/hmdb51",
    )
)

CLASSES = [
    "stand",
    "walk",
    "sit",
    "eat",
    "drink",
    "hug",
    "run",
    "ride_bike",
    "ride_horse",
    "fall_floor",
    "push",
    "hit",
    "punch",
    "shoot_gun",
    "smoke",
]

CATEGORIES = {
    "normal": ["stand", "walk", "sit", "eat", "drink", "hug"],
    "mobility": ["run", "ride_bike"],
    "anomaly": ["ride_horse", "fall_floor"],
    "threat": ["push", "hit", "punch", "shoot_gun"],
    "restricted": ["smoke"],
}
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

VIDEO_DIR = HMDB_ROOT / "videos"
SPLIT_DIR = HMDB_ROOT / "annotations" / "testTrainMulti_7030_splits"

N_FRAMES = 16
RESOLUTION = 112
VAL_FRACTION = 0.15
SEED = 42


@dataclass(frozen=True)
class HMDB51Split:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    class_names: list[str]
    missing_files: list[str]


def parse_split_file(split_path: Path) -> dict[str, int]:
    result = {}
    invalid_lines = []
    with open(split_path) as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            filename = parts[0]
            try:
                label = int(parts[1])
            except ValueError:
                invalid_lines.append((line_no, line.strip()))
                continue
            if label not in (0, 1, 2):
                invalid_lines.append((line_no, line.strip()))
                continue
            if filename in result:
                invalid_lines.append((line_no, f"duplicate: {filename}"))
                continue
            result[filename] = label
    if invalid_lines:
        raise ValueError(
            f"Invalid lines in {split_path}:\n"
            + "\n".join(f"  line {n}: {s}" for n, s in invalid_lines[:10])
        )
    return result


def prepare_hmdb51(
    classes: list[str] = CLASSES,
    split_id: int = 1,
    val_fraction: float = VAL_FRACTION,
    seed: int = SEED,
) -> HMDB51Split:
    """Build train/val/test dataframes from official split files."""
    train_rows = []
    test_rows = []
    missing_files = []

    for cls in classes:
        split_file = SPLIT_DIR / f"{cls}_test_split{split_id}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        labels = parse_split_file(split_file)
        class_dir = VIDEO_DIR / cls
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class dir: {class_dir}")

        for filename, label in labels.items():
            if label == 0:
                continue
            video_path = class_dir / filename
            if not video_path.is_file():
                missing_files.append(str(video_path))
                continue

            row = {
                "video_path": str(video_path),
                "label": CLASS_TO_IDX[cls],
                "class_name": cls,
            }
            if label == 1:
                train_rows.append(row)
            elif label == 2:
                test_rows.append(row)

    train_df_full = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    if len(train_df_full) == 0:
        raise RuntimeError("No training videos found. Check split files and video paths.")

    if missing_files:
        print(
            f"\n[WARN] {len(missing_files)} videos referenced by split files are missing from disk:"
        )
        for p in missing_files[:10]:
            print(f"  {p}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    train_df, val_df = train_test_split(
        train_df_full,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_df_full["label"],
    )

    print(f"Split {split_id} totals:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")

    print(f"\nTrain class distribution:")
    for i, name in enumerate(classes):
        count = (train_df["label"] == i).sum()
        print(f"  [{i}] {name:<8} {count:>4}")

    print(f"\nTest class distribution:")
    for i, name in enumerate(classes):
        count = (test_df["label"] == i).sum()
        print(f"  [{i}] {name:<8} {count:>4}")

    return HMDB51Split(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        class_names=classes,
        missing_files=missing_files,
    )


def filter_corrupt_videos(df, n_frames: int = 16) -> pd.DataFrame:
    """Remove rows whose videos cannot be decoded.

    Returns filtered DataFrame.
    """
    valid_rows = []
    corrupt = []
    for _, row in df.iterrows():
        path = row["video_path"]
        # Try to open and read first frame
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            corrupt.append(path)
            cap.release()
            continue
        ok, _ = cap.read()
        cap.release()
        if not ok:
            corrupt.append(path)
            continue
        valid_rows.append(row)

    print(f"Filtered: {len(df)} -> {len(valid_rows)} (dropped {len(corrupt)} corrupt)")
    return pd.DataFrame(valid_rows).reset_index(drop=True)
