"""Dataset loaders for UrbanMind-AI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from urbanintel.config import dataset_path
from urbanintel.data.schemas import NYC311Schema

# Canonical name map: raw CSV column -> our schema name
_COLUMN_ALIASES = {
    "unique key": "unique_key",
    "created date": "created_date",
    "closed date": "closed_date",
    "agency": "agency",
    "agency name": "agency_name",
    "problem (formerly complaint type)": "complaint_type",
    "problem detail (formerly descriptor)": "descriptor",
    "incident zip": "incident_zip",
    "borough": "borough",
    "latitude": "latitude",
    "longitude": "longitude",
    "city": "city",
    "status": "status",
}

REQUIRED_COLUMNS = [
    "unique_key",
    "created_date",
    "complaint_type",
    "descriptor",
    "borough",
    "incident_zip",
    "latitude",
    "longitude",
]

# dtypes we care about — zip must be string (preserve leading zeros)
_DTYPES = {
    "Incident Zip": str,
    "Unique Key": str,
}


class DataContractError(ValueError):
    """Raised when a dataset violates its declared contract."""


def load_nyc311(
    path: Path | str | None = None,
    *,
    validate: bool = True,
) -> pd.DataFrame:
    """Load NYC 311 service requests.

    Parameters
    ----------
    path:
        Path to the CSV file. If None, resolves via config.dataset_path.
    validate:
        If True, validate the result against NYC311Schema.

    Returns
    -------
    A DataFrame with canonical column names, filtered to required columns.

    Raises
    ------
    FileNotFoundError:
        If the CSV does not exist.
    DataContractError:
        If required columns are missing or validation fails.
    """
    if path is None:
        path = dataset_path("nyc311", "nyc311_sample.csv")
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"NYC 311 dataset not found at {path}. "
            "Set URBANINTEL_DATA_ROOT or download the dataset."
        )

    df = pd.read_csv(path, low_memory=False, dtype=_DTYPES)

    # Normalize: strip, lowercase, but preserve the original for alias lookup
    renamed = {
        col: _COLUMN_ALIASES.get(col.strip().lower(), col.strip().lower()) for col in df.columns
    }
    df = df.rename(columns=renamed)

    # Check required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataContractError(
            f"Missing required columns: {missing}. Available (first 20): {sorted(df.columns)[:20]}"
        )

    # Keep only required columns
    df = df[REQUIRED_COLUMNS].copy()

    if validate:
        try:
            NYC311Schema.validate(df)
        except Exception as e:
            raise DataContractError(f"Schema validation failed: {e}") from e

    return df


def summarize(df: pd.DataFrame) -> dict:
    """Return a summary of the DataFrame for logging."""
    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "n_complaint_types": int(df["complaint_type"].nunique()),
        "n_boroughs": int(df["borough"].nunique()) if "borough" in df else 0,
        "null_fraction": float(df.isna().mean().mean()),
    }
