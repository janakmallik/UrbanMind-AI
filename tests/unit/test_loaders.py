"""Tests for the NYC 311 loader — contract enforcement."""

from pathlib import Path

import pandas as pd
import pytest

from urbanintel.data.loaders import DataContractError, load_nyc311, summarize


def _write_csv(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


BASE_ROW = {
    "Unique Key": "1",
    "Created Date": "2024-01-01 10:00:00",
    "Problem (formerly Complaint Type)": "Illegal Parking",
    "Problem Detail (formerly Descriptor)": "Blocked Driveway",
    "Borough": "BROOKLYN",
    "Incident Zip": "11201",
    "Latitude": 40.69,
    "Longitude": -73.99,
}


@pytest.fixture
def tiny_csv(tmp_path: Path) -> Path:
    rows = [
        {**BASE_ROW, "Unique Key": "1"},
        {**BASE_ROW, "Unique Key": "2", "Problem (formerly Complaint Type)": "Noise - Residential"},
        {**BASE_ROW, "Unique Key": "3", "Problem (formerly Complaint Type)": "HEAT/HOT WATER"},
    ]
    return _write_csv(tmp_path / "nyc311.csv", rows)


def test_loads_valid_file(tiny_csv: Path) -> None:
    df = load_nyc311(tiny_csv)
    assert len(df) == 3
    assert "complaint_type" in df.columns
    assert "descriptor" in df.columns


def test_raises_when_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_nyc311(tmp_path / "nope.csv")


def test_raises_when_required_column_missing(tmp_path: Path) -> None:
    row = {k: v for k, v in BASE_ROW.items() if k != "Borough"}
    bad = _write_csv(tmp_path / "bad.csv", [row])
    with pytest.raises(DataContractError):
        load_nyc311(bad)


def test_raises_when_latitude_out_of_bounds(tmp_path: Path) -> None:
    row = {**BASE_ROW, "Latitude": 99.0}  # not NYC
    bad = _write_csv(tmp_path / "bad_lat.csv", [row])
    with pytest.raises(DataContractError):
        load_nyc311(bad)


def test_raises_when_label_missing(tmp_path: Path) -> None:
    row = {**BASE_ROW, "Problem (formerly Complaint Type)": None}
    bad = _write_csv(tmp_path / "bad_label.csv", [row])
    with pytest.raises(DataContractError):
        load_nyc311(bad)


def test_zip_codes_stay_as_strings(tmp_path: Path) -> None:
    row = {**BASE_ROW, "Incident Zip": "01234"}  # leading zero
    good = _write_csv(tmp_path / "zip.csv", [row])
    df = load_nyc311(good)
    assert df["incident_zip"].iloc[0] == "01234"


def test_validate_false_skips_schema(tmp_path: Path) -> None:
    row = {**BASE_ROW, "Latitude": 99.0}
    bad = _write_csv(tmp_path / "bad.csv", [row])
    # Should not raise when validate=False
    df = load_nyc311(bad, validate=False)
    assert len(df) == 1


def test_summarize_shape(tiny_csv: Path) -> None:
    df = load_nyc311(tiny_csv, validate=False)
    summary = summarize(df)
    assert summary["n_rows"] == 3
    assert summary["n_complaint_types"] == 3
