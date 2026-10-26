"""Precompute log-mel spectrograms for all 1,080 Urbansas clips.

Saves to a single .npy file for fast loading during training.
~3-5 min on CPU (parallelized with multiprocessing).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from multiprocessing import Pool

import numpy as np

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import (
    AUDIO_SAMPLE_RATE,
    N_MELS,
    audio_to_mel,
    load_audio,
)

OUT_DIR = _REPO_ROOT / "data" / "urbansas_mels"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_one(args):
    """Worker: load one wav, compute mel, return (filename, mel)."""
    filename, wav_path = args
    audio = load_audio(wav_path)
    mel = audio_to_mel(audio, AUDIO_SAMPLE_RATE)
    return filename, mel


def main() -> None:
    root = urbansas_root()
    audio_dir = root / "audio"
    wav_files = sorted(audio_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} wav files")

    tasks = [(p.stem, p) for p in wav_files]

    print(f"Computing mels with multiprocessing.Pool(8)...")
    import time

    t0 = time.time()
    with Pool(8) as pool:
        results = pool.map(compute_one, tasks, chunksize=20)
    print(f"Done in {time.time() - t0:.1f}s")

    # Determine mel shape (all should be same: 128, 173)
    first_mel = results[0][1]
    print(f"Mel shape: {first_mel.shape}")

    # Build arrays
    filenames = [r[0] for r in results]
    mels = np.stack([r[1] for r in results], axis=0).astype(np.float32)
    print(f"Mels array: {mels.shape} ({mels.nbytes / 1e6:.1f} MB)")

    np.save(OUT_DIR / "mels.npy", mels)
    with open(OUT_DIR / "filenames.txt", "w") as f:
        f.write("\n".join(filenames))
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
