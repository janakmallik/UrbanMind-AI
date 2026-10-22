"""Tests for configuration resolution."""

from pathlib import Path

from urbanintel.config import DEFAULT_DATA_ROOT, dataset_path, get_data_root


def test_default_data_root_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("URBANINTEL_DATA_ROOT", raising=False)
    assert get_data_root() == DEFAULT_DATA_ROOT


def test_env_overrides_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("URBANINTEL_DATA_ROOT", str(tmp_path))
    assert get_data_root() == tmp_path.resolve()


def test_dataset_path_joins_correctly(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("URBANINTEL_DATA_ROOT", str(tmp_path))
    p = dataset_path("nyc311", "sample.csv")
    assert p == tmp_path.resolve() / "nyc311" / "sample.csv"
