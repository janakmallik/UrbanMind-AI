"""Inspect HMDB-51 dataset structure, class counts, and splits."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

HMDB_ROOT = Path("C:/Users/Janak Mallik/ml-data/hmdb51")
VIDEO_DIR = HMDB_ROOT / "videos"
SPLIT_DIR = HMDB_ROOT / "annotations" / "testTrainMulti_7030_splits"

# Candidate classes (as suggested — verify counts before committing)
CANDIDATE_CLASSES = ["walk", "run", "sit", "wave", "clap"]


def main() -> None:
    if not VIDEO_DIR.exists():
        print(f"ERROR: {VIDEO_DIR} not found.")
        return

    print(f"Scanning {VIDEO_DIR} ...")
    class_dirs = sorted([d for d in VIDEO_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} class folders\n")

    class_counts = {}
    for cls_dir in class_dirs:
        videos = list(cls_dir.glob("*.avi")) + list(cls_dir.glob("*.mp4"))
        class_counts[cls_dir.name] = len(videos)

    print("=== Class counts (top 20 by size) ===")
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {name:<20} {count:>4} videos")

    print("\n=== Smallest 10 classes ===")
    for name, count in sorted(class_counts.items(), key=lambda x: x[1])[:10]:
        print(f"  {name:<20} {count:>4} videos")

    print("\n=== Candidate classes for Week 7 ===")
    total_candidate = 0
    for name in CANDIDATE_CLASSES:
        count = class_counts.get(name, 0)
        total_candidate += count
        print(f"  {name:<20} {count:>4} videos")

    print(f"\nTotal candidate videos: {total_candidate}")

    # Check splits
    if SPLIT_DIR.exists():
        split_files = sorted(SPLIT_DIR.glob("*_test_split1.txt"))
        print(f"\nSplit files (split 1): {len(split_files)} found")
        # Inspect one split file to see format
        if split_files:
            sample = split_files[0]
            print(f"\nSample split file: {sample.name}")
            with open(sample) as f:
                lines = f.readlines()[:5]
            for line in lines:
                print(f"  {line.strip()}")
    else:
        print(f"\nWARNING: Split directory not found at {SPLIT_DIR}")


if __name__ == "__main__":
    main()
