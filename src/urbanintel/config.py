"""Configuration for UrbanMind-AI.

Data root resolution order:
1. URBANINTEL_DATA_ROOT environment variable, if set.
2. .env file at the repo root, if present.
3. Fallback: ./data/raw (relative to the repo).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw"

# Load .env once on import
load_dotenv(REPO_ROOT / ".env", override=False)


def get_data_root() -> Path:
    """Return the resolved data root directory.

    Reads URBANINTEL_DATA_ROOT from the environment (populated by .env if present).
    Falls back to <repo>/data/raw.
    """
    env_value = os.environ.get("URBANINTEL_DATA_ROOT")
    if env_value:
        root = Path(env_value).expanduser().resolve()
    else:
        root = DEFAULT_DATA_ROOT
    return root


def dataset_path(name: str, *parts: str) -> Path:
    """Return the path to a dataset file or directory.

    Example
    -------
    dataset_path("nyc311", "nyc311_sample.csv")
        -> <data_root>/nyc311/nyc311_sample.csv
    """
    return get_data_root().joinpath(name, *parts)
