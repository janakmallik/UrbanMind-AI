"""UCF-Crime dataset: loader, category mapping, clip builder."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

UCF_ROOT = Path("C:/Users/Janak Mallik/ml-data/ucf-crime-extracted")

# 4-category grouping
CATEGORIES = {0: "normal", 1: "violence", 2: "criminal", 3: "public_hazard"}
NUM_CATEGORIES = len(CATEGORIES)

CLASS_TO_CATEGORY = {
    "NormalVideos": 0,
    # Violence
    "Abuse": 1,
    "Assault": 1,
    "Fighting": 1,
    "Shooting": 1,
    "Explosion": 1,
    # Criminal
    "Arrest": 2,
    "Burglary": 2,
    "Robbery": 2,
    "Shoplifting": 2,
    "Stealing": 2,
    # Public hazard
    "Arson": 3,
    "RoadAccidents": 3,
    "Vandalism": 3,
}

ALL_CLASSES = list(CLASS_TO_CATEGORY.keys())


def _video_id(filename: str) -> str:
    stem = Path(filename).stem
    m = re.match(r"(.+)_\d+$", stem)
    return m.group(1) if m else stem


def _frame_num(filename: str) -> int:
    stem = Path(filename).stem
    m = re.search(r"_(\d+)$", stem)
    return int(m.group(1)) if m else 0


def _scan_split(split_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in split_dir.iterdir():
        if not class_dir.is_dir() or class_dir.name not in CLASS_TO_CATEGORY:
            continue
        category = CLASS_TO_CATEGORY[class_dir.name]
        for png in class_dir.glob("*.png"):
            vid = _video_id(png.name)
            rows.append(
                {
                    "image_path": str(png),
                    "class": class_dir.name,
                    "category": category,
                    "video_id": vid,
                    "global_video_id": f"{class_dir.name}::{vid}",  # class-scoped
                    "frame_num": _frame_num(png.name),
                }
            )
    return pd.DataFrame(rows)


def load_ucf_crime(split: str = "Train") -> pd.DataFrame:
    """Load Train or Test split as DataFrame."""
    df = _scan_split(UCF_ROOT / split)
    print(f"{split}: {len(df):,} frames from {df['video_id'].nunique():,} videos")
    print(f"Category distribution:")
    for cat_id, cat_name in CATEGORIES.items():
        n = (df["category"] == cat_id).sum()
        print(f"  [{cat_id}] {cat_name:<15} {n:>8,} ({100 * n / len(df):.1f}%)")
    return df


def split_by_video(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by global_video_id, STRATIFIED by category.

    Ensures each split has videos from every category — prevents "test
    split has 0 normal videos" from random grouping.
    """
    import numpy as np

    # Step 1: get unique (global_video_id, category) pairs
    vids_df = df.groupby("global_video_id")["category"].first().reset_index()

    # Warn on categories with too few videos
    min_videos = vids_df.groupby("category").size().min()
    if min_videos < 3:
        print(
            f"WARNING: smallest category has only {min_videos} videos — "
            f"stratified split may leave it out of some splits"
        )

    # Step 2: for each category, split videos stratified
    train_ids, val_ids, test_ids = [], [], []
    rng = np.random.default_rng(seed)

    for cat in sorted(vids_df["category"].unique()):
        cat_vids = vids_df[vids_df["category"] == cat]["global_video_id"].to_numpy()
        rng.shuffle(cat_vids)
        n = len(cat_vids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Ensure at least 1 video per split per category (if enough videos)
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        train_ids.extend(cat_vids[:n_train])
        val_ids.extend(cat_vids[n_train : n_train + n_val])
        test_ids.extend(cat_vids[n_train + n_val :])

    train_df = df[df["global_video_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["global_video_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["global_video_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


def build_clips(
    df: pd.DataFrame,
    n_frames: int = 16,
    stride: int = 8,
    max_clips_per_video: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Group frames into temporal clips per video."""
    import random

    rng = random.Random(seed)

    clips = []
    n_skipped_gap = 0
    for global_video_id, group in df.groupby("global_video_id"):
        group = group.sort_values("frame_num").reset_index(drop=True)
        if len(group) < n_frames:
            continue

        # Validate: no duplicate or non-monotonic frames
        frame_nums = group["frame_num"].to_numpy()
        diffs = np.diff(frame_nums)
        if (diffs <= 0).any():
            # Non-monotonic after sort — data problem
            continue

        # Optional: warn on irregular gaps (>2× median)
        median_diff = np.median(diffs)
        if (diffs > 2 * median_diff).any():
            n_skipped_gap += 1
            # Skip this video — irregular sampling makes clips unreliable
            continue

        starts = list(range(0, len(group) - n_frames + 1, stride))
        if len(starts) > max_clips_per_video:
            starts = rng.sample(starts, max_clips_per_video)

        for start in starts:
            window = group.iloc[start : start + n_frames]
            clips.append(
                {
                    "video_id": group["video_id"].iloc[0],
                    "global_video_id": global_video_id,
                    "frame_paths": window["image_path"].tolist(),
                    "class": group["class"].iloc[0],
                    "category": int(group["category"].iloc[0]),
                }
            )

    if n_skipped_gap > 0:
        print(f"  Skipped {n_skipped_gap} videos with irregular frame sampling")

    return pd.DataFrame(clips)
