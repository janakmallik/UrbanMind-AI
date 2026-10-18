"""NYC 311 multimodal dataset: text + tabular fusion.

Combines:
  - Text: `additional_details` (operator free text, ~53% null)
  - Tabular: borough, ZIP, calendar features, lat/lon

Target: complaint_type (top-5 classes).

Uses the same split methodology as Weeks 1 and 4
(stratified 70/15/15, random_state=42).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TOP_N_CLASSES = 5
RANDOM_STATE = 42


@dataclass(frozen=True)
class MultimodalSplit:
    train_df: pd.DataFrame  # columns: text, borough, zip_binned, hour, dow, lat, lon, label_id
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    label_names: list[str]
    text_field: str  # 'additional_details' or 'descriptor'
    text_missing_fraction: float


def prepare_multimodal(
    df: pd.DataFrame,
    top_n: int = TOP_N_CLASSES,
    text_field: str = "additional_details",
    test_size: float = 0.30,
    val_size: float = 0.50,
    random_state: int = RANDOM_STATE,
) -> MultimodalSplit:
    """Filter to top-N classes, prep text + tabular, split.

    Text field must exist in the DataFrame. Rows with null or empty text
    are assigned a placeholder token so the fusion model sees them.
    """
    if text_field not in df.columns:
        raise ValueError(f"text_field '{text_field}' not found in DataFrame")

    df = df.copy()

    # Step 1: select top-N classes from full dataset (matches Week 1/4)
    top = df["complaint_type"].value_counts().head(top_n).index.tolist()
    df = df[df["complaint_type"].isin(top)].copy()

    # Step 2: encode labels
    label_names = sorted(top)
    label_to_id = {name: i for i, name in enumerate(label_names)}
    df["label_id"] = df["complaint_type"].map(label_to_id)

    # Step 3: prepare text
    df["text"] = df[text_field].fillna("").astype(str).str.strip()
    df.loc[df["text"] == "", "text"] = "<EMPTY>"
    text_missing = (df[text_field].isna() | (df["text"] == "<EMPTY>")).mean()

    # Step 4: prepare tabular
    df["borough"] = df["borough"].fillna("UNKNOWN").astype(str)
    df["incident_zip"] = df["incident_zip"].fillna("UNKNOWN").astype(str)

    # Top-50 ZIPs
    top_zips = df["incident_zip"].value_counts().head(50).index.tolist()
    df["zip_binned"] = df["incident_zip"].where(df["incident_zip"].isin(top_zips), "OTHER")

    # Calendar
    df["created_at"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["hour"] = df["created_at"].dt.hour.fillna(-1).astype(int)
    df["dow"] = df["created_at"].dt.dayofweek.fillna(-1).astype(int)
    df["month"] = df["created_at"].dt.month.fillna(-1).astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Keep only what the model needs
    keep = [
        "text",
        "borough",
        "zip_binned",
        "hour",
        "dow",
        "month",
        "is_weekend",
        "latitude",
        "longitude",
        "label_id",
        "complaint_type",
    ]
    df = df[keep].reset_index(drop=True)

    # Step 5: stratified split
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_id"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size,
        random_state=random_state,
        stratify=temp_df["label_id"],
    )

    return MultimodalSplit(
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        label_names=label_names,
        text_field=text_field,
        text_missing_fraction=float(text_missing),
    )
