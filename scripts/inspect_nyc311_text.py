"""Inspect NYC 311 descriptor text before training.

Prints class distribution, descriptor vocabulary size, and train/test
overlap. Verifies the pipeline before we spend time training DistilBERT.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanintel.config import dataset_path
from urbanintel.data.loaders import load_nyc311
from urbanintel.data.nyc311_text import prepare_text_data


def main() -> None:
    csv_path = dataset_path("nyc311", "nyc311_sample.csv")
    print(f"Loading {csv_path} ...")
    df = load_nyc311(csv_path, validate=False)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

    split = prepare_text_data(df)

    print()
    print("=== Label distribution ===")
    print(f"Classes: {split.label_names}")
    print(f"Train: {len(split.train_texts):,} examples")
    print(f"Val:   {len(split.val_texts):,} examples")
    print(f"Test:  {len(split.test_texts):,} examples")

    print()
    print("=== Train class counts ===")
    train_counter = Counter(split.train_labels)
    for i, name in enumerate(split.label_names):
        count = train_counter.get(i, 0)
        print(f"  [{i}] {name:<30} {count:>6,}")

    print()
    print("=== Descriptor statistics ===")
    train_descs = split.train_texts
    lengths = [len(d.split()) for d in train_descs]
    print(f"Unique train descriptors: {len(set(train_descs)):,}")
    print(f"Unique test descriptors:  {len(set(split.test_texts)):,}")
    print(f"Mean descriptor word count: {sum(lengths) / len(lengths):.2f}")
    print(f"Max descriptor word count:  {max(lengths)}")
    print(f"Min descriptor word count:  {min(lengths)}")

    print()
    print("=== Top 10 train descriptors ===")
    for desc, count in Counter(train_descs).most_common(10):
        print(f"  {count:>6,}  {desc}")

    print()
    print("=== Train/test overlap ===")
    for k, v in split.overlap_report.items():
        if isinstance(v, float):
            print(f"  {k:<40} {v:.4f}")
        else:
            print(f"  {k:<40} {v}")


if __name__ == "__main__":
    main()
