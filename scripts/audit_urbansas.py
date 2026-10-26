"""Urbansas label + split audit.

Answers:
    1. How many unique location_id / city groups?
    2. Offscreen label distribution per location.
    3. Are there enough offscreen clips per location for reliable CV?
    4. Subset (TAU vs. MAVD) and night distribution.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\Janak Mallik\ml-data\urbansas")


def main() -> None:
    audio_ann = pd.read_csv(ROOT / "annotations" / "audio_annotations.csv")
    video_ann = pd.read_csv(ROOT / "annotations" / "video_annotations.csv")

    # Per-clip metadata from video annotations
    clip_meta = video_ann.drop_duplicates(subset=["filename"])[
        ["filename", "location_id", "city", "subset", "night"]
    ]

    # Per-clip offscreen label from audio annotations
    labels = (
        audio_ann.groupby("filename")["label"]
        .apply(lambda x: "offscreen" in set(x.astype(str)))
        .rename("has_offscreen")
    )

    df = clip_meta.merge(labels, left_on="filename", right_index=True)

    print("=" * 70)
    print("1. CLIP-LEVEL STATS")
    print("=" * 70)
    print(f"Total annotated clips: {len(df)}")
    n_pos = int(df["has_offscreen"].sum())
    print(f"Has offscreen: {n_pos} ({100 * n_pos / len(df):.1f}%)")
    print(f"Has no offscreen: {len(df) - n_pos} ({100 * (len(df) - n_pos) / len(df):.1f}%)")
    print()

    print("=" * 70)
    print("2. LOCATIONS + CITIES")
    print("=" * 70)
    print(f"Unique location_id: {df['location_id'].nunique()}")
    print(f"Unique cities: {df['city'].nunique()}")
    print()

    print("=" * 70)
    print("3. CLIPS PER LOCATION")
    print("=" * 70)
    per_loc = (
        df.groupby(["city", "location_id"])
        .agg(
            clips=("filename", "count"),
            offscreen=("has_offscreen", "sum"),
        )
        .reset_index()
    )
    per_loc["offscreen_pct"] = (100 * per_loc["offscreen"] / per_loc["clips"]).round(1)
    print(per_loc.to_string(index=False))
    print()

    print("=" * 70)
    print("4. SUBSET DISTRIBUTION (TAU vs. MAVD)")
    print("=" * 70)
    per_subset = df.groupby("subset").agg(
        clips=("filename", "count"),
        locations=("location_id", "nunique"),
        offscreen=("has_offscreen", "sum"),
    )
    per_subset["offscreen_pct"] = (100 * per_subset["offscreen"] / per_subset["clips"]).round(1)
    print(per_subset.to_string())
    print()

    print("=" * 70)
    print("5. NIGHT DISTRIBUTION")
    print("=" * 70)
    per_night = df.groupby("night").agg(
        clips=("filename", "count"),
        locations=("location_id", "nunique"),
        offscreen=("has_offscreen", "sum"),
    )
    per_night["offscreen_pct"] = (100 * per_night["offscreen"] / per_night["clips"]).round(1)
    print(per_night.to_string())
    print()

    print("=" * 70)
    print("6. OFFScreen CLIPS PER LOCATION (if any location has 0)")
    print("=" * 70)
    zero_offscreen = per_loc[per_loc["offscreen"] == 0]
    if len(zero_offscreen) > 0:
        print(f"WARNING: {len(zero_offscreen)} location(s) have 0 offscreen clips")
        print(zero_offscreen.to_string(index=False))
        print("These locations contribute nothing to positive-class recall.")
        print("Consider LeaveOneGroupOut CV instead of GroupKFold.")
    else:
        print("All locations have at least 1 offscreen clip.")
    print()

    print("=" * 70)
    print("7. RECOMMENDATION")
    print("=" * 70)
    n_locs = df["location_id"].nunique()
    if n_locs >= 5:
        print(f"GroupKFold(n_splits=5) is viable ({n_locs} locations).")
    elif n_locs >= 3:
        print(f"Use LeaveOneGroupOut CV ({n_locs} locations).")
    else:
        print(f"Only {n_locs} locations — use a single location-based train/test split.")


if __name__ == "__main__":
    main()
