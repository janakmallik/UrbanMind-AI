"""Data schemas for UrbanMind-AI datasets."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class NYC311Schema(pa.DataFrameModel):
    """Schema for NYC 311 service requests.

    Only the columns we actually use are validated. Real-world NYC 311
    files have ~40 columns; we keep the subset we care about.
    """

    unique_key: Series[str] = pa.Field(unique=True, nullable=False)
    created_date: Series[str] = pa.Field(nullable=False)
    complaint_type: Series[str] = pa.Field(nullable=False)
    descriptor: Series[str] = pa.Field(nullable=True)
    borough: Series[str] = pa.Field(nullable=True)
    incident_zip: Series[str] = pa.Field(nullable=True)
    latitude: Series[float] = pa.Field(nullable=True, ge=40.0, le=41.5)
    longitude: Series[float] = pa.Field(nullable=True, ge=-74.5, le=-73.5)

    class Config:
        strict = False  # allow extra columns
        coerce = True  # attempt type coercion
