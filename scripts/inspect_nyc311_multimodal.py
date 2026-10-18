"""Inspect the multimodal NYC 311 data before modeling."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanintel.config import dataset_path
from urbanintel.data.loaders import load_nyc311
from urbanintel.data.nyc311_multimodal import prepare_multimodal


def main() -> None:
    csv_path = dataset_path("nyc311", "nyc311_sample.csv")
    print(f"Loading {csv_path} ...")
    df = load_nyc311(csv_path, validate=False)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns\n")

    split = prepare_multimodal(df, text_field="additional_details")

    print(f"Text field: {split.text_field}")
    print(f"Text missing fraction: {split.text_missing_fraction:.3f}\n")

    print("=== Split sizes ===")
    print(f"Train: {len(split.train_df):,}")
    print(f"Val:   {len(split.val_df):,}")
    print(f"Test:  {len(split.test_df):,}\n")

    print("=== Class distribution (train) ===")
    for i, name in enumerate(split.label_names):
        count = (split.train_df["label_id"] == i).sum()
        pct = 100 * count / len(split.train_df)
        print(f"  [{i}] {name:<30} {count:>6,} ({pct:.1f}%)")

    print("\n=== Text statistics (train) ===")
    texts = split.train_df["text"]
    non_empty = texts[texts != "<EMPTY>"]
    lengths = non_empty.str.split().str.len()
    print(
        f"Non-empty texts: {len(non_empty):,} / {len(texts):,} ({100 * len(non_empty) / len(texts):.1f}%)"
    )
    print(f"Mean word count (non-empty): {lengths.mean():.2f}")
    print(f"Median word count: {lengths.median():.0f}")
    print(f"Max word count: {lengths.max()}")
    print(f"Unique non-empty texts: {non_empty.nunique():,}")

    print("\n=== Sample non-empty texts (train) ===")
    for _, row in non_empty.sample(5, random_state=42).iterrows():
        label = split.label_names[row["label_id"]]
        print(f"  [{label[:20]:<20}] {row['text'][:100]}")


if __name__ == "__main__":
    main()
