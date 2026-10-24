"""Find SegFormer training contamination in ResNet's split.

Reads:
    data/splits/resnet_{train,val,test}_ids.json

Writes:
    data/splits/contaminated_test_ids.json    (in ResNet test AND SegFormer train)
    data/splits/contaminated_val_ids.json     (in ResNet val AND SegFormer train)
    data/splits/clean_test_ids.json           (ResNet test minus contaminated)
    data/splits/clean_val_ids.json            (ResNet val minus contaminated)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = _REPO_ROOT / "data" / "splits"
SEG_TRAIN_DIR = Path("C:/Users/Janak Mallik/ml-data/bdd100k/seg/images/train")


def main() -> None:
    print("Loading saved splits...")
    resnet_test = set(json.loads((SPLITS_DIR / "resnet_test_ids.json").read_text()))
    resnet_val = set(json.loads((SPLITS_DIR / "resnet_val_ids.json").read_text()))
    print(f"  ResNet test: {len(resnet_test)}")
    print(f"  ResNet val: {len(resnet_val)}")

    print("Loading SegFormer train image names...")
    seg_train = {p.name for p in SEG_TRAIN_DIR.iterdir() if p.suffix.lower() == ".jpg"}
    print(f"  SegFormer train: {len(seg_train)}")

    # Contamination in test
    contaminated_test = resnet_test & seg_train
    clean_test = resnet_test - seg_train
    print(
        f"\n  Contaminated test: {len(contaminated_test)} "
        f"({100 * len(contaminated_test) / len(resnet_test):.2f}%)"
    )
    print(f"  Clean test: {len(clean_test)}")

    # Contamination in val (fusion training pool)
    contaminated_val = resnet_val & seg_train
    clean_val = resnet_val - seg_train
    print(
        f"\n  Contaminated val: {len(contaminated_val)} "
        f"({100 * len(contaminated_val) / len(resnet_val):.2f}%)"
    )
    print(f"  Clean val: {len(clean_val)}")

    # Save
    for name, ids in [
        ("contaminated_test", contaminated_test),
        ("clean_test", clean_test),
        ("contaminated_val", contaminated_val),
        ("clean_val", clean_val),
    ]:
        out = SPLITS_DIR / f"{name}_ids.json"
        out.write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")
        print(f"  → {out.name}")

    print("\n✓ Contamination analysis complete.")


if __name__ == "__main__":
    main()
