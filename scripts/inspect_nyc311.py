"""Inspect the NYC 311 dataset before modeling.

Prints row count, columns, label distribution, missing values, and size.
Does not modify anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanintel.config import dataset_path


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    csv_path = dataset_path("nyc311", "nyc311_sample.csv")
    print(f"File: {csv_path}")
    if not csv_path.exists():
        print("ERROR: file not found. Set URBANINTEL_DATA_ROOT or download the dataset.")
        raise SystemExit(1)

    size = csv_path.stat().st_size
    print(f"Size: {human_bytes(size)}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print()

    print("=== First 15 columns ===")
    for i, col in enumerate(df.columns[:15], 1):
        print(f"  {i:2d}. {col}")
    print()

    # Label column
    label_col = None
    for candidate in ["Problem (formerly Complaint Type)", "Complaint Type"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        print("ERROR: label column not found")
        raise SystemExit(1)

    print(f"=== Label column: {label_col} ===")
    counts = df[label_col].value_counts()
    print(f"Distinct labels: {counts.size}")
    print()
    print("Top 15 labels:")
    for label, count in counts.head(15).items():
        pct = 100 * count / len(df)
        print(f"  {count:>8,}  {pct:5.2f}%  {label}")
    print()

    print("=== Missing values (top 15 by fraction) ===")
    missing = df.isna().mean().sort_values(ascending=False).head(15)
    for col, frac in missing.items():
        print(f"  {frac * 100:5.2f}%  {col}")
    print()

    # Key columns for our baseline
    print("=== Key columns for baseline ===")
    for col in [
        "Problem (formerly Complaint Type)",
        "Problem Detail (formerly Descriptor)",
        "Borough",
        "Incident Zip",
        "Latitude",
        "Longitude",
        "Created Date",
    ]:
        if col in df.columns:
            frac_null = df[col].isna().mean()
            n_unique = df[col].nunique(dropna=True)
            print(f"  {col:<45} null={frac_null * 100:5.2f}%  unique={n_unique:,}")
        else:
            print(f"  {col:<45} NOT PRESENT")


if __name__ == "__main__":
    main()
