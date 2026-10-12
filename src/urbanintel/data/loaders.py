"""Dataset loaders for UrbanMind-AI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urbanintel.data.schemas import NYC311Schema

DATA_ROOT = Path("data/raw")


def load_nyc311(
    path: Path | str = DATA_ROOT / "nyc311" / "nyc311_sample.csv",
    validate: bool = True,
) -> pd.DataFrame:
    """Load NYC 311 service requests.

    Parameters
    ----------
    path:
        Path to the CSV file.
    validate:
        If True, validate against NYC311Schema.

    Returns
    -------
    A cleaned DataFrame with normalized column names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"NYC 311 dataset not found at {path}. Run scripts/download_nyc311.ps1 first."
        )

    df = pd.read_csv(path, low_memory=False)

    # Normalize column names: lowercase, spaces -> underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)

    # NYC Open Data renamed "Complaint Type" -> "Problem (formerly Complaint Type)"
    # and "Descriptor" -> "Problem Detail (formerly Descriptor)".
    # Map these aliases to canonical names used by the schema.
    column_aliases = {
        "problem_(formerly_complaint_type)": "complaint_type",
        "problem_detail_(formerly_descriptor)": "descriptor",
    }
    df = df.rename(columns=column_aliases)

    # Keep only the columns we declared in the schema
    keep = [
        "unique_key",
        "created_date",
        "complaint_type",
        "descriptor",
        "borough",
        "incident_zip",
        "latitude",
        "longitude",
    ]

    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {missing}. "
            f"Available columns: {sorted(df.columns)[:15]}..."
        )

    df = df[keep].copy()

    if validate:
        NYC311Schema.validate(df)

    return df


def summarize(df: pd.DataFrame) -> dict:
    """Return a summary of the DataFrame for logging."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_complaint_types": df["complaint_type"].nunique(),
        "n_boroughs": df["borough"].nunique() if "borough" in df else 0,
        "null_fraction": float(df.isna().mean().mean()),
    }
