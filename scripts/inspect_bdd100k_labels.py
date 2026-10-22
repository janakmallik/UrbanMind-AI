"""Inspect BDD100K scene labels before downloading images.

Reads the official label JSON and reports scene distribution. Determines
which classes are viable for a classification experiment.

If registration is blocked, this script also works with a Kaggle mirror
that provides the same JSON format.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

LABEL_DIR = Path("C:/Users/Janak Mallik/ml-data/bdd100k/labels")


def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def count_scenes(records: list[dict]) -> Counter:
    """Count scene values across records. Missing -> 'undefined'."""
    counts = Counter()
    for record in records:
        scene = record.get("attributes", {}).get("scene")
        counts[scene if scene is not None else "undefined"] += 1
    return counts


def main() -> None:
    train_path = LABEL_DIR / "bdd100k_labels_images_train.json"
    val_path = LABEL_DIR / "bdd100k_labels_images_val.json"

    for name, path in [("train", train_path), ("val", val_path)]:
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        records = load_json(path)
        counts = count_scenes(records)
        total = sum(counts.values())
        print(f"\n=== {name} ({total:,} images) ===")
        for scene, count in counts.most_common():
            pct = 100 * count / total
            print(f"  {scene:<20} {count:>7,}  ({pct:5.2f}%)")


if __name__ == "__main__":
    main()
