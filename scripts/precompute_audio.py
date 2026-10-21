"""Pre-decode UrbanSound8K audio to memory-mapped numpy.

Uses soundfile directly (no torchcodec required).
"""

from __future__ import annotations

import glob
import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset

PARQUET_GLOB = "C:/Users/Janak Mallik/ml-data/urbansound8k/data/train-*.parquet"
OUT_DIR = Path("C:/Users/Janak Mallik/ml-data/urbansound8k-cache")

SAMPLE_RATE = 22050
DURATION = 4.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)


def resample(array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return array
    import librosa

    return librosa.resample(array, orig_sr=orig_sr, target_sr=target_sr)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ready = OUT_DIR / "ready.txt"
    if ready.exists():
        print(f"Cache already exists at {OUT_DIR}")
        return

    files = sorted(glob.glob(PARQUET_GLOB))
    print(f"Loading {len(files)} parquet shards (decode=False) ...")
    ds = load_dataset("parquet", data_files=files, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    meta = pd.DataFrame(
        {
            "classID": [int(r["classID"]) for r in ds],
            "fold": [int(r["fold"]) for r in ds],
            "slice_file_name": [r["slice_file_name"] for r in ds],
        }
    )
    print(f"Total clips: {len(meta)}")

    arr = np.zeros((len(meta), N_SAMPLES), dtype=np.float32)

    print(f"Decoding {len(ds)} clips ...")
    for i in range(len(ds)):
        raw = ds[i]["audio"]
        if not isinstance(raw, dict) or raw.get("bytes") is None:
            raise ValueError(f"Row {i}: unexpected audio format {type(raw)}")
        array, sr = sf.read(io.BytesIO(raw["bytes"]))
        if array.ndim > 1:
            array = array.mean(axis=1)  # mono
        array = resample(array.astype(np.float32), sr, SAMPLE_RATE)
        if len(array) < N_SAMPLES:
            array = np.pad(array, (0, N_SAMPLES - len(array)))
        else:
            array = array[:N_SAMPLES]
        arr[i] = array
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(ds)} done")

    npy_path = OUT_DIR / "audio.npy"
    print(f"Saving {arr.shape} to {npy_path} (~{arr.nbytes / 1e9:.2f} GB) ...")
    np.save(npy_path, arr)
    meta.to_csv(OUT_DIR / "meta.csv", index=False)
    ready.write_text("ok")
    print(f"Done.")


if __name__ == "__main__":
    main()
