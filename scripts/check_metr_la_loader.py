"""Manual smoke test: load METR-LA, split, window, print shapes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanintel.data.metr_la import make_datasets


def main() -> None:
    datasets = make_datasets()  # uses URBANINTEL_DATA_ROOT
    for name, ds in datasets.items():
        print(
            f"{name:5s}  X={ds.X.shape}  y={ds.y.shape}  "
            f"valid_mask={ds.valid_mask.mean():.3f}  "
            f"time range={ds.timestamps[0]} -> {ds.timestamps[-1]}"
        )


if __name__ == "__main__":
    main()
