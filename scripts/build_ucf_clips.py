"""Build train/val/test clip DataFrames from UCF-Crime frames.

Saves to data/ucf_crime_clips/*.parquet for reuse by training.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from urbanintel.data.ucf_crime import (
    CATEGORIES,
    build_clips,
    load_ucf_crime,
    split_by_video,
)

OUT_DIR = _REPO_ROOT / "data" / "ucf_crime_clips"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-videos-per-class",
        type=int,
        default=None,
        help="For smoke test: cap videos per class",
    )
    args = parser.parse_args()

    print("Scanning UCF-Crime Train split (frame paths only) ...")
    train_full = load_ucf_crime("Train")

    if args.max_videos_per_class:
        capped = []
        rng = random.Random(42)
        for cls, grp in train_full.groupby("class"):
            vids = grp["global_video_id"].unique()
            if len(vids) > args.max_videos_per_class:
                vids = rng.sample(list(vids), args.max_videos_per_class)
            capped.append(grp[grp["global_video_id"].isin(vids)])
        train_full = pd.concat(capped, ignore_index=True)
        print(
            f"Capped to {args.max_videos_per_class} videos/class: "
            f"{len(train_full):,} frames, "
            f"{train_full['global_video_id'].nunique()} videos"
        )

    gc.collect()

    print("\nSplitting by (class, video_id) ...")
    train_df, val_df, test_df = split_by_video(train_full)

    print(f"\nFrame-level split (using global_video_id):")
    print(f"  Train: {len(train_df):,} frames / {train_df['global_video_id'].nunique()} videos")
    print(f"  Val:   {len(val_df):,} frames / {val_df['global_video_id'].nunique()} videos")
    print(f"  Test:  {len(test_df):,} frames / {test_df['global_video_id'].nunique()} videos")

    print("\nBuilding 16-frame clips ...")
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        clips = build_clips(df, n_frames=16, stride=8, max_clips_per_video=5)
        path = OUT_DIR / f"{name}_clips.parquet"
        clips.to_parquet(path)
        if len(clips) == 0:
            print(
                f"\n{name}: 0 clips — WARNING: split is empty. "
                f"Cause: too few frames per video or too aggressive gap filtering."
            )
            continue
        print(f"\n{name}: {len(clips)} clips saved to {path.name}")
        print(f"  Category distribution:")
        for cat_id, cat_name in CATEGORIES.items():
            n = (clips["category"] == cat_id).sum()
            pct = 100 * n / len(clips)
            print(f"    [{cat_id}] {cat_name:<15} {n:>6} ({pct:.1f}%)")

    print("\nLoading official UCF-Crime Test split ...")
    official_test = load_ucf_crime("Test")
    official_clips = build_clips(official_test, n_frames=16, stride=8, max_clips_per_video=5)
    official_clips.to_parquet(OUT_DIR / "official_test_clips.parquet")
    print(f"\nOfficial test: {len(official_clips)} clips")
    print("Saved to official_test_clips.parquet")


if __name__ == "__main__":
    main()
