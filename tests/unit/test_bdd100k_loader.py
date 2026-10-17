"""Tests for the BDD100K loader."""

import json
from pathlib import Path

import pandas as pd
import pytest

from urbanintel.data.bdd100k import (
    LABEL_TO_ID,
    VIABLE_CLASSES,
    _index_images,
    load_labels,
    prepare_bdd100k,
)


@pytest.fixture
def tiny_annotation(tmp_path: Path) -> Path:
    records = [
        {"name": "a.jpg", "attributes": {"scene": "city street"}},
        {"name": "b.jpg", "attributes": {"scene": "highway"}},
        {"name": "c.jpg", "attributes": {"scene": "residential"}},
        {"name": "d.jpg", "attributes": {"scene": "parking lot"}},  # excluded
        {"name": "e.jpg", "attributes": {}},  # missing scene
    ]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(records))
    return path


@pytest.fixture
def tiny_images(tmp_path: Path) -> Path:
    root = tmp_path / "images"
    root.mkdir()
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        (root / name).touch()
    return root


def test_index_images_recursive(tmp_path: Path) -> None:
    root = tmp_path / "images"
    (root / "sub1").mkdir(parents=True)
    (root / "sub2").mkdir()
    (root / "x.jpg").touch()
    (root / "sub1" / "y.jpg").touch()
    (root / "sub2" / "z.jpg").touch()
    index = _index_images(root)
    assert set(index.keys()) == {"x.jpg", "y.jpg", "z.jpg"}


def test_load_labels_filters_classes(tiny_annotation: Path, tiny_images: Path) -> None:
    index = _index_images(tiny_images)
    df = load_labels(tiny_annotation, index)
    # Only 3 rows: a, b, c (parking lot and missing-scene excluded)
    assert len(df) == 3
    assert set(df["scene"]) == set(VIABLE_CLASSES)
    assert df["label_id"].isin([0, 1, 2]).all()


def test_load_labels_skips_missing_images(tiny_annotation: Path, tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    df = load_labels(tiny_annotation, {})
    assert len(df) == 0


def test_label_to_id_mapping() -> None:
    assert LABEL_TO_ID == {"city street": 0, "highway": 1, "residential": 2}
