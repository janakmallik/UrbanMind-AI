"""Inspect UrbanSound8K loaded from local parquet files."""

from __future__ import annotations

from collections import Counter

import numpy as np
from datasets import load_dataset


def main() -> None:
    print("Loading UrbanSound8K from local parquet ...")
    ds = load_dataset(
        "parquet",
        data_files="C:/Users/Janak Mallik/ml-data/urbansound8k/data/train-*.parquet",
        split="train",
    )

    print(f"\nRows: {len(ds)}")
    print(f"Features: {ds.features}")

    print("\n=== First row ===")
    sample = ds[0]
    for k, v in sample.items():
        if k == "audio":
            print(
                f"  {k}: array shape={np.array(v['array']).shape}, "
                f"sr={v['sampling_rate']}, dtype={np.array(v['array']).dtype}"
            )
        else:
            print(f"  {k}: {v}")

    print("\n=== Class distribution ===")
    label_col = "class" if "class" in ds.features else "classID"
    counts = Counter([row[label_col] for row in ds])
    for lbl, count in sorted(counts.items(), key=lambda x: str(x[0])):
        print(f"  {lbl}: {count}")

    print("\n=== Fold distribution ===")
    if "fold" in ds.features:
        fold_counts = Counter([row["fold"] for row in ds])
        for fold, count in sorted(fold_counts.items()):
            print(f"  fold {fold}: {count}")


if __name__ == "__main__":
    main()
