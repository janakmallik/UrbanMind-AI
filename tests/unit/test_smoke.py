"""Smoke test - verifies the package imports and version is defined."""

import urbanintel


def test_version_exists():
    assert urbanintel.__version__ == "0.1.0"


def test_package_imports():
    from urbanintel import data, evaluation, features, models, serving, training

    assert data is not None
    assert features is not None
    assert models is not None
    assert training is not None
    assert evaluation is not None
    assert serving is not None
