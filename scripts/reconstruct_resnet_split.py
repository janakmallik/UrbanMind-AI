"""Reproduce ResNet's BDD100K split using seed=42.

Writes:
    data/splits/resnet_train_ids.json
    data/splits/resnet_val_ids.json
    data/splits/resnet_test_ids.json

Verifies against MLflow's dataset.json:
    val_rows should be 4932
    test_rows should be 4932
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from urbanintel.config import dataset_path
from urbanintel.data.bdd100k import prepare_bdd100k

# Same paths as train_bdd100k_resnet50.py
LABEL_DIR = dataset_path("bdd100k", "labels/100k")
IMAGES_DIR = dataset_path("bdd100k", "images/100k")
OUT_DIR = _REPO_ROOT / "data" / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Reconstructing ResNet's BDD100K split (seed=42)...")
    split = prepare_bdd100k(
        train_json=LABEL_DIR / "bdd100k_labels_images_train.json",
        val_json=LABEL_DIR / "bdd100k_labels_images_val.json",
        train_images_dir=IMAGES_DIR / "train",
        val_images_dir=IMAGES_DIR / "val",
        val_test_split=0.5,
        random_state=42,
    )

    # Save image IDs
    for name, df in [
        ("train", split.train_df),
        ("val", split.val_df),
        ("test", split.test_df),
    ]:
        ids = df["image_name"].tolist()
        out_path = OUT_DIR / f"resnet_{name}_ids.json"
        out_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
        print(f"  {name}: {len(ids)} images → {out_path.name}")

    # Verify against MLflow
    expected = {"train": 68969, "val": 4932, "test": 4932}
    actual = {
        "train": len(split.train_df),
        "val": len(split.val_df),
        "test": len(split.test_df),
    }
    print("\n=== Verification vs MLflow dataset.json ===")
    for k in expected:
        match = "✓" if actual[k] == expected[k] else "✗"
        print(f"  {match} {k}: expected={expected[k]}, actual={actual[k]}")

    if actual["val"] != 4932 or actual["test"] != 4932:
        print("\n⚠️  COUNT MISMATCH — split logic may have changed. STOP.")
        sys.exit(1)

    print("\n✓ Split reconstruction matches MLflow.")


if __name__ == "__main__":
    main()
