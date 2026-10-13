"""Feature engineering for NYC 311 complaint-type classification.

CRITICAL: Only features that exist BEFORE the complaint is classified
are used. The `descriptor` field is a downstream refinement of the
complaint type and MUST NOT be used as a model feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_ZIP_COUNT = 50


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert created_date to datetime and extract calendar features."""
    out = df.copy()
    out["created_at"] = pd.to_datetime(out["created_date"], errors="coerce")
    out["hour"] = out["created_at"].dt.hour
    out["dow"] = out["created_at"].dt.dayofweek
    out["month"] = out["created_at"].dt.month
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    return out


def add_cyclical(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour, dow, month as sin/cos pairs."""
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def bin_zip(df: pd.DataFrame, top_zips: list[str] | None = None) -> pd.DataFrame:
    """Reduce ZIP cardinality by keeping top-N zips and folding the rest."""
    out = df.copy()
    out["incident_zip"] = out["incident_zip"].fillna("UNKNOWN").astype(str)
    if top_zips is None:
        top_zips = out["incident_zip"].value_counts().head(TOP_ZIP_COUNT).index.tolist()
    out["zip_binned"] = out["incident_zip"].where(out["incident_zip"].isin(top_zips), "OTHER")
    return out


def clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing categorical features with a clear placeholder."""
    out = df.copy()
    if "borough" in out.columns:
        out["borough"] = out["borough"].fillna("UNKNOWN").astype(str)
    if "agency" in out.columns:
        out["agency"] = out["agency"].fillna("UNKNOWN").astype(str)
    if "location_type" in out.columns:
        out["location_type"] = out["location_type"].fillna("UNKNOWN").astype(str)
    return out


def build_features(df: pd.DataFrame, top_zips: list[str] | None = None) -> pd.DataFrame:
    """Apply all feature transformations in one call."""
    df = parse_datetime(df)
    df = add_cyclical(df)
    df = bin_zip(df, top_zips=top_zips)
    df = clean_categoricals(df)
    return df


def numeric_feature_columns() -> list[str]:
    """Numeric feature columns fed to the model."""
    return [
        "hour",
        "dow",
        "month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "latitude",
        "longitude",
    ]


def categorical_feature_columns() -> list[str]:
    """Categorical feature columns fed to the model.

    NOTE: 'descriptor' is intentionally excluded — it is a downstream
    refinement of the target and would cause leakage.
    """
    cols = ["borough", "zip_binned"]
    # agency / location_type included only if present in the dataframe
    return cols
