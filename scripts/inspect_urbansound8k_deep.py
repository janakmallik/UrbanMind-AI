"""Deep inspection of UrbanSound8K parquet — bypasses HF audio decoder.

Reads the audio column as raw bytes and decodes with soundfile.
No torchcodec required.
"""

from __future__ import annotations

import io
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import soundfile as sf

PARQUET_DIR = "C:/Users/Janak Mallik/ml-data/urbansound8k/data"


def load_raw():
    """Load all parquet shards into a single DataFrame with raw audio bytes."""
    import glob

    files = sorted(glob.glob(f"{PARQUET_DIR}/train-*.parquet"))
    print(f"Loading {len(files)} parquet shards ...")

    dfs = []
    for f in files:
        df = pd.read_parquet(f, columns=["audio", "slice_file_name", "fold", "classID", "class"])
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    return full


def decode_audio(raw):
    """Decode HF audio column (dict with 'bytes' and 'path') to numpy array."""
    if isinstance(raw, dict):
        # HF returns {'bytes': ..., 'path': ...} when decode=False
        if raw.get("bytes") is not None:
            array, sr = sf.read(io.BytesIO(raw["bytes"]))
            return array, sr
    raise ValueError(f"Unexpected audio format: {type(raw)}")


def main() -> None:
    # Load with decode disabled — read raw bytes
    import glob

    from datasets import Audio, load_dataset

    files = sorted(glob.glob(f"{PARQUET_DIR}/train-*.parquet"))
    print("Loading parquet (decode=False) ...")
    ds = load_dataset(
        "parquet",
        data_files=files,
        split="train",
    )
    # Disable auto-decode
    ds = ds.cast_column("audio", Audio(decode=False))

    print(f"Rows: {len(ds)}")
    print(f"Features: {ds.features}")

    # --- Audio sample check ---
    print("\n=== Audio samples (3 random) ===")
    rng = np.random.default_rng(42)
    for idx in rng.choice(len(ds), size=3, replace=False):
        row = ds[int(idx)]
        raw = row["audio"]
        print(f"Row {idx}: classID={row['classID']}, fold={row['fold']}")
        print(f"  raw keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw)}")
        try:
            array, sr = decode_audio(raw)
            print(f"  decoded: shape={array.shape}, sr={sr}")
            print(f"  duration: {len(array) / sr:.2f}s")
            print(f"  amplitude: min={array.min():.3f}, max={array.max():.3f}")
        except Exception as e:
            print(f"  decode failed: {e}")

    # --- Class distribution ---
    print("\n=== Class distribution ===")
    class_counts = Counter([int(r["classID"]) for r in ds])
    for c in sorted(class_counts):
        print(f"  classID {c}: {class_counts[c]}")

    # --- Duration distribution (200 clips) ---
    print("\n=== Duration distribution (200 random) ===")
    durations = []
    for idx in rng.choice(len(ds), size=200, replace=False):
        row = ds[int(idx)]
        try:
            array, sr = decode_audio(row["audio"])
            durations.append(len(array) / sr)
        except Exception:
            pass
    if durations:
        d = np.array(durations)
        print(f"  min={d.min():.2f}s  median={np.median(d):.2f}s  max={d.max():.2f}s")
        print(f"  >4s: {(d > 4).sum()}/200")

    # --- Fold distribution per class ---
    print("\n=== Folds per class ===")
    fold_class = defaultdict(lambda: defaultdict(int))
    for r in ds:
        fold_class[int(r["classID"])][int(r["fold"])] += 1
    print(f"{'classID':<8} " + " ".join(f"f{f:<3}" for f in range(1, 11)))
    for c in sorted(fold_class):
        row_str = f"{c:<8} "
        for f in range(1, 11):
            row_str += f"{fold_class[c][f]:<4}"
        print(row_str)


if __name__ == "__main__":
    main()
