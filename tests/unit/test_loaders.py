"""Tests for dataset loaders."""

from pathlib import Path

import pandas as pd
import pytest

from urbanintel.data.loaders import load_nyc311, summarize


@pytest.fixture
def tiny_nyc311(tmp_path: Path) -> Path:
    """Create a tiny fake NYC 311 CSV for testing."""
    csv = tmp_path / "nyc311_sample.csv"
    df = pd.DataFrame(
        {
            "Unique Key": ["1", "2", "3", "4", "5"],
            "Created Date": [
                "2024-01-01 10:00:00",
                "2024-01-01 11:00:00",
                "2024-01-01 12:00:00",
                "2024-01-02 09:00:00",
                "2024-01-02 10:00:00",
            ],
            "Complaint Type": [
                "Noise - Residential",
                "Noise - Residential",
                "Illegal Parking",
                "Illegal Parking",
                "HEAT/HOT WATER",
            ],
            "Descriptor": [
                "Loud Music",
                "Loud Music",
                "Blocked Driveway",
                "Blocked Driveway",
                "No Heat",
            ],
            "Borough": ["BROOKLYN", "BROOKLYN", "MANHATTAN", "MANHATTAN", "BRONX"],
            "Incident Zip": ["11201", "11201", "10001", "10001", "10451"],
            "Latitude": [40.69, 40.69, 40.75, 40.75, 40.81],
            "Longitude": [-73.99, -73.99, -73.99, -73.99, -73.92],
        }
    )
    df.to_csv(csv, index=False)
    return csv


def test_load_nyc311_returns_dataframe(tiny_nyc311: Path) -> None:
    df = load_nyc311(tiny_nyc311)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5


def test_load_nyc311_normalizes_columns(tiny_nyc311: Path) -> None:
    df = load_nyc311(tiny_nyc311)
    assert "complaint_type" in df.columns
    assert "Complaint Type" not in df.columns


def test_load_nyc311_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_nyc311(tmp_path / "does_not_exist.csv")


def test_summarize_returns_expected_keys(tiny_nyc311: Path) -> None:
    df = load_nyc311(tiny_nyc311)
    summary = summarize(df)
    assert "n_rows" in summary
    assert summary["n_rows"] == 5
    assert summary["n_complaint_types"] == 3
