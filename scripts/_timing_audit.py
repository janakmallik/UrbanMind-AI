"""Throwaway timing audit for nested-CV cost estimation.

Runs ONE audio inner-fold training (3 locations train, 1 test) to measure
real per-fold cost, then extrapolates to the full nested CV (25 folds).

Delete this file after use — it's not part of the project.
"""

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from train_urbansas_audio import train_and_eval  # same directory

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import prepare_urbansas


def main():
    root = urbansas_root()
    split = prepare_urbansas(root, val_fraction=0.0, test_fraction=0.0)
    mavd = split.train_df[split.train_df["subset"] == "mavd"].reset_index(drop=True)

    locs = sorted(mavd["location_id"].unique())
    train_df = mavd[mavd["location_id"].isin(locs[:3])].reset_index(drop=True)
    test_df = mavd[mavd["location_id"] == locs[3]].reset_index(drop=True)

    print(f"Train: {len(train_df)} clips (3 locations)")
    print(f"Test:  {len(test_df)} clips (1 location, held out)")
    print(f"Timing one inner-fold training (15 epochs, batch 16, workers 4)...")
    print()

    t0 = time.time()
    m, *_ = train_and_eval(
        train_df,
        test_df,
        epochs=15,
        batch_size=16,
        lr=1e-3,
        num_workers=4,
        seed=42,
    )
    dt = time.time() - t0

    print()
    print(f"Audio inner-fold time:  {dt:.0f}s = {dt / 60:.2f} min")
    print(f"Extrapolated to 25 inner folds: {25 * dt / 60:.1f} min = {25 * dt / 3600:.2f} hours")
    print()
    print(f"Sanity: this fold's F1 was {m['f1']:.4f} (n_pos={m['n_pos_test']})")


if __name__ == "__main__":
    main()
