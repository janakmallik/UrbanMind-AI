"""Week 9.5 gate — is NYC 311 tabular+text fusion viable?

Usage:
    uv run python scripts/audit_nyc311.py --csv path/to/nyc311.csv
    NYC311_CSV=path/to/nyc311.csv uv run python scripts/audit_nyc311.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

load_dotenv()


def resolve_csv_path(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    if args.csv is not None:
        return args.csv
    env = os.environ.get("NYC311_CSV")
    if env:
        return Path(env)
    parser.error("Provide --csv <path> or set NYC311_CSV environment variable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=None, help="Path to NYC 311 CSV (or set NYC311_CSV env var)"
    )
    args = parser.parse_args()

    csv_path = resolve_csv_path(args, parser)
    print(f"Loading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    # --- Resolve column names (raw CSV uses NYC's official names) ---
    def find_col(*candidates: str) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    closed_col = find_col("closed_date", "Closed Date")
    created_col = find_col("created_date", "Created Date")
    descriptor_col = find_col("descriptor", "Problem Detail (formerly Descriptor)")
    type_col = find_col("complaint_type", "Problem (formerly Complaint Type)")

    if closed_col is None or created_col is None or descriptor_col is None:
        print("\nFAIL: could not resolve required columns")
        print(f"  closed_date: {closed_col}")
        print(f"  created_date: {created_col}")
        print(f"  descriptor: {descriptor_col}")
        sys.exit(1)

    print(f"\n=== Resolved columns ===")
    print(f"  created_date:  {created_col}")
    print(f"  closed_date:   {closed_col}")
    print(f"  descriptor:    {descriptor_col}")
    if type_col:
        print(f"  complaint_type: {type_col}")

    print(f"\n=== Shape ===")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Columns: {df.columns.tolist()}")

    descriptor_col = find_col("descriptor", "Problem Detail (formerly Descriptor)")
    type_col = find_col("complaint_type", "Problem (formerly Complaint Type)")

    if closed_col is None:
        print("\nFAIL: no closed_date / Closed Date column")
        sys.exit(1)
    if created_col is None:
        print("\nFAIL: no created_date / Created Date column")
        sys.exit(1)
    if descriptor_col is None:
        print("\nFAIL: no descriptor / 'Problem Detail (formerly Descriptor)' column")
        sys.exit(1)

    print(f"\n=== Resolved columns ===")
    print(f"  created_date:  {created_col}")
    print(f"  closed_date:   {closed_col}")
    print(f"  descriptor:    {descriptor_col}")
    if type_col:
        print(f"  complaint_type: {type_col}")

    closed_col = "closed_date" if "closed_date" in df.columns else "Closed Date"
    created_col = "created_date" if "created_date" in df.columns else "Created Date"

    df[closed_col] = pd.to_datetime(df[closed_col], errors="coerce")
    df[created_col] = pd.to_datetime(df[created_col], errors="coerce")

    # --- Timestamp validation ---
    print(f"\n=== Timestamp validation ===")
    print(
        f"Missing created_date: {df[created_col].isna().sum()} ({100 * df[created_col].isna().sum() / len(df):.1f}%)"
    )
    print(
        f"Missing closed_date:  {df[closed_col].isna().sum()} ({100 * df[closed_col].isna().sum() / len(df):.1f}%)"
    )
    print(f"Date range (created): {df[created_col].min()} to {df[created_col].max()}")

    valid = df.dropna(subset=[closed_col, created_col]).copy()
    print(f"Valid (both dates): {len(valid)} ({100 * len(valid) / len(df):.1f}%)")

    # --- Resolution time distribution ---
    print(f"\n=== Resolution-time distribution ===")
    days = (valid[closed_col] - valid[created_col]).dt.total_seconds() / 86400
    print(
        f"min={days.min():.2f}, median={days.median():.1f}, "
        f"mean={days.mean():.1f}, max={days.max():.1f}"
    )

    negative = (days < 0).sum()
    if negative > 0:
        print(f"\n⚠️  {negative} rows with negative resolution time — dropping")
        mask = days >= 0
        days = days[mask]
        valid = valid[mask].copy()

    for t in [1, 3, 7, 14, 30]:
        pct = 100 * (days <= t).sum() / len(days)
        print(f"  resolved within {t:2d} days: {pct:5.1f}%")

    # --- Target definition ---
    valid["resolved_fast"] = (days <= 7).astype(int)
    y = valid["resolved_fast"].values
    print(f"\n=== Target: resolved within 7 days ===")
    print(f"Fast (<=7 days): {y.sum()} ({100 * y.mean():.1f}%)")
    print(f"Slow (>7 days):  {(1 - y).sum()} ({100 * (1 - y).mean():.1f}%)")

    # --- Majority baseline ---
    majority_f1 = f1_score(y, np.zeros_like(y), average="macro", zero_division=0)
    print(f"Majority-class baseline macro-F1: {majority_f1:.3f}")

    # --- Text evaluation (leak-free via Pipeline) ---
    print(f"\n=== Test: does descriptor predict resolution? ===")

    # TF-IDF pipeline (leak-free: vectorizer fits inside CV folds)
    text_pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    text_series = valid[descriptor_col].fillna("").astype(str)
    scores_tfidf = cross_val_score(
        text_pipe,
        text_series,
        y,
        cv=5,
        scoring="f1_macro",
    )
    print(
        f"TF-IDF + LogReg 5-fold CV macro-F1: {scores_tfidf.mean():.3f} ± {scores_tfidf.std():.3f}"
    )

    # Categorical (one-hot) inside pipeline
    cat_pipe = Pipeline(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    cat_series = valid[[descriptor_col]].fillna("")
    scores_cat = cross_val_score(
        cat_pipe,
        cat_series,
        y,
        cv=5,
        scoring="f1_macro",
    )
    print(
        f"Descriptor one-hot 5-fold CV macro-F1: {scores_cat.mean():.3f} ± {scores_cat.std():.3f}"
    )

    # --- Interpretation ---
    print(f"\n=== Interpretation ===")
    print(f"Majority baseline: {majority_f1:.3f}")
    print(f"Text (TF-IDF):     {scores_tfidf.mean():.3f}")
    print(f"Text (one-hot):    {scores_cat.mean():.3f}")
    print()
    print("Decision rules:")
    print(f"  - TF-IDF - majority = {scores_tfidf.mean() - majority_f1:+.3f}")
    print(f"  - One-hot - majority = {scores_cat.mean() - majority_f1:+.3f}")
    print(f"  - TF-IDF - one-hot = {scores_tfidf.mean() - scores_cat.mean():+.3f}")
    print()
    print("If TF-IDF > majority by ≥0.10 and TF-IDF > one-hot by ≥0.03:")
    print("  → text has real signal, fusion is worth pursuing")
    print("Otherwise:")
    print("  → text adds nothing beyond categorical lookup; fusion not justified")


if __name__ == "__main__":
    main()
