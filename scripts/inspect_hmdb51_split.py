import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from urbanintel.data.hmdb51 import CLASSES, prepare_hmdb51


def main() -> None:
    split = prepare_hmdb51(CLASSES, split_id=1)
    if hasattr(split, "missing_files") and split.missing_files:
        print(f"\n⚠️  {len(split.missing_files)} missing files")

    print(f"\n=== Summary ===")
    print(f"Total train:  {len(split.train_df):,}")
    print(f"Total val:    {len(split.val_df):,}")
    print(f"Total test:   {len(split.test_df):,}")
    print(f"Total:        {len(split.train_df) + len(split.val_df) + len(split.test_df):,}")

    train_paths = set(split.train_df["video_path"])
    val_paths = set(split.val_df["video_path"])
    test_paths = set(split.test_df["video_path"])

    assert train_paths & val_paths == set(), "train/val overlap!"
    assert train_paths & test_paths == set(), "train/test overlap!"
    assert val_paths & test_paths == set(), "val/test overlap!"
    print("\n✅ No train/val/test overlap")

    # Verify every class present in each split
    for split_name, df in [
        ("train", split.train_df),
        ("val", split.val_df),
        ("test", split.test_df),
    ]:
        present = set(df["class_name"].unique())
        missing = set(CLASSES) - present
        if missing:
            print(f"⚠️  {split_name} missing classes: {missing}")

    print("\n=== Per-class counts ===")
    print(f"{'Class':<8} {'Train':>6} {'Val':>6} {'Test':>6}")
    for i, name in enumerate(CLASSES):
        t = (split.train_df["label"] == i).sum()
        v = (split.val_df["label"] == i).sum()
        s = (split.test_df["label"] == i).sum()
        print(f"{name:<8} {t:>6} {v:>6} {s:>6}")


if __name__ == "__main__":
    main()
