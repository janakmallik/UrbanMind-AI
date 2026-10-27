"""Precompute Urbansas video frames to a memmap cache.

Decoding video per-sample is slow on these MP4s (~800ms/clip via decord,
~2s/clip via cv2). Precompute once to a numpy memmap; read at ~5-15ms/clip.

Cache shape: (1080, 16, 112, 112, 3) float32, Kinetics-normalized.
Cache size:  ~2.6 GB on disk.

Preprocessing per clip:
    1. Decode 16 uniformly-spaced frames (via decord, fallback to cv2)
    2. Resize each to 112x112 (cv2.INTER_AREA)
    3. Convert to float32, divide by 255
    4. Apply Kinetics mean/std normalization
    5. Store as (16, 112, 112, 3) float32

Augmentation is NOT applied here — it is applied at training time
by CachedUrbansasVideoDataset, so each epoch sees a different random crop.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from multiprocessing import Pool

import cv2
import numpy as np

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import _decode_frames_decord

OUT_DIR = _REPO_ROOT / "data" / "urbansas_video_frames"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FRAMES = 16
RESOLUTION = 112
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def _find_video_path(root: Path, stem: str) -> Path | None:
    for folder in ["video_24fps_1", "video_24fps_2", "video_24fps_3"]:
        p = root / folder / f"{stem}.mp4"
        if p.exists():
            return p
    return None


def _process_one(args):
    stem, video_path = args
    frames = _decode_frames_decord(video_path, n_frames=N_FRAMES)
    if frames is None:
        return stem, None

    # Resize
    resized = np.empty((N_FRAMES, RESOLUTION, RESOLUTION, 3), dtype=np.uint8)
    for i in range(N_FRAMES):
        resized[i] = cv2.resize(frames[i], (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)

    # Float + normalize
    x = resized.astype(np.float32) / 255.0
    x = (x - KINETICS_MEAN) / KINETICS_STD
    return stem, x.astype(np.float32)  # (16, 112, 112, 3)


def main() -> None:
    root = urbansas_root()

    # Enumerate clips via audio_annotations.csv (matches what prepare_urbansas uses)
    import pandas as pd

    ann = pd.read_csv(root / "annotations" / "audio_annotations.csv")
    stems = sorted(ann["filename"].unique().tolist())
    print(f"Found {len(stems)} unique clip stems")

    tasks = []
    missing = []
    for stem in stems:
        p = _find_video_path(root, stem)
        if p is None:
            missing.append(stem)
        else:
            tasks.append((stem, p))

    if missing:
        print(f"WARNING: {len(missing)} clips have no video; they will be excluded from cache")
        for m in missing[:5]:
            print(f"  {m}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")

    print(f"Processing {len(tasks)} clips with multiprocessing.Pool(8)...")
    import time

    t0 = time.time()
    with Pool(8) as pool:
        results = pool.map(_process_one, tasks, chunksize=8)
    print(f"Done in {time.time() - t0:.0f}s")

    # Check for decode failures
    failed = [stem for stem, x in results if x is None]
    if failed:
        print(f"WARNING: {len(failed)} clips failed to decode")
        for s in failed[:5]:
            print(f"  {s}")

    # Build arrays, dropping failed
    kept = [(stem, x) for stem, x in results if x is not None]
    print(f"Keeping {len(kept)}/{len(results)} clips")

    stems_kept = [s for s, _ in kept]
    frames_arr = np.stack([x for _, x in kept], axis=0).astype(np.float32)
    print(f"Cache array: {frames_arr.shape}  ({frames_arr.nbytes / 1e9:.2f} GB)")

    np.save(OUT_DIR / "frames.npy", frames_arr)
    with open(OUT_DIR / "filenames.txt", "w") as f:
        f.write("\n".join(stems_kept))
    print(f"Saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
