"""Precompute 16 kHz mono audio cache for AST pretraining.

AST expects 16 kHz audio input. The existing 48 kHz mel cache is not
compatible with AST's preprocessing pipeline.

Cache shape: (1080, 160000) float32, 16 kHz mono.
Cache size:  ~691 MB on disk.

Reference: ast-finetuned-audioset uses the standard AST preprocessing:
    16 kHz mono → 128-mel log-spectrogram → 1024 frames (10.24s)
This cache stores the raw waveform; AST's feature extractor handles the mel.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from multiprocessing import Pool

import numpy as np
import soundfile as sf

from urbanintel.config import urbansas_root

OUT_DIR = _REPO_ROOT / "data" / "urbansas_audio16k"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16_000
DURATION_S = 10.0
TARGET_SAMPLES = int(TARGET_SR * DURATION_S)  # 160,000


def _load_and_resample(args):
    stem, wav_path = args
    try:
        data, sr = sf.read(str(wav_path), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            import librosa

            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
        # Pad or trim to exactly 10s
        if len(data) < TARGET_SAMPLES:
            data = np.pad(data, (0, TARGET_SAMPLES - len(data)))
        else:
            data = data[:TARGET_SAMPLES]
        return stem, data.astype(np.float32)
    except Exception as e:
        print(f"  FAILED: {stem} — {type(e).__name__}: {e}")
        return stem, None


def main() -> None:
    root = urbansas_root()
    audio_dir = root / "audio"

    # --- Verify source completeness ---
    wav_files = sorted(audio_dir.glob("*.wav"))
    if len(wav_files) != 1080:
        raise RuntimeError(
            f"Expected 1080 wav files in {audio_dir}, found {len(wav_files)}. "
            "Urbansas dataset is incomplete — refusing to build partial cache."
        )
    stems_expected = {p.stem for p in wav_files}
    if len(stems_expected) != 1080:
        raise RuntimeError(
            f"Duplicate stems in audio dir: {len(wav_files)} files, "
            f"{len(stems_expected)} unique stems"
        )
    print(f"Found {len(wav_files)} unique wav files (expected 1080). OK.")

    tasks = [(p.stem, p) for p in wav_files]

    print("Computing 16 kHz audio cache with multiprocessing.Pool(8)...")
    import time

    t0 = time.time()
    with Pool(8) as pool:
        results = pool.map(_load_and_resample, tasks, chunksize=20)
    print(f"Done in {time.time() - t0:.1f}s")

    # --- Verify all loaded successfully, fail otherwise ---
    failed = [stem for stem, x in results if x is None]
    if failed:
        raise RuntimeError(
            f"{len(failed)} files failed to load. First 5: {failed[:5]}. "
            "Refusing to save partial cache. Investigate the failures and re-run."
        )

    # --- Verify no rows were dropped ---
    if len(results) != len(tasks):
        raise RuntimeError(
            f"Result count {len(results)} != task count {len(tasks)}. "
            "Multiprocessing dropped rows — investigate before proceeding."
        )

    stems = [s for s, _ in results]
    audio_arr = np.stack([x for _, x in results], axis=0).astype(np.float32)

    # --- Verify shape ---
    expected_shape = (1080, TARGET_SAMPLES)
    if audio_arr.shape != expected_shape:
        raise RuntimeError(f"Audio array shape {audio_arr.shape} != expected {expected_shape}")

    # --- Verify no NaNs ---
    if np.isnan(audio_arr).any():
        raise RuntimeError(
            f"Audio array contains {int(np.isnan(audio_arr).sum())} NaN values — "
            "investigate the failed clips before proceeding."
        )

    # --- Verify per-clip non-silence (RMS > 1e-6) ---
    rms = np.sqrt(np.mean(audio_arr**2, axis=1))
    silent = (rms < 1e-6).sum()
    if silent > 0:
        raise RuntimeError(
            f"{silent} clips have RMS < 1e-6 (essentially silent). Investigate before proceeding."
        )

    print(f"Cache array: {audio_arr.shape}  ({audio_arr.nbytes / 1e6:.1f} MB)")
    print(f"Global RMS: {float(np.sqrt(np.mean(audio_arr**2))):.4f}")
    print(f"Per-clip RMS: min={rms.min():.4f}  median={np.median(rms):.4f}  max={rms.max():.4f}")

    np.save(OUT_DIR / "audio.npy", audio_arr)
    with open(OUT_DIR / "filenames.txt", "w") as f:
        f.write("\n".join(stems))

    # --- Verify saved files ---
    if not (OUT_DIR / "audio.npy").exists():
        raise RuntimeError(f"Failed to save {OUT_DIR / 'audio.npy'}")
    if not (OUT_DIR / "filenames.txt").exists():
        raise RuntimeError(f"Failed to save {OUT_DIR / 'filenames.txt'}")

    print(f"Saved to: {OUT_DIR}")
    print(f"  audio.npy: {(OUT_DIR / 'audio.npy').stat().st_size / 1e6:.1f} MB")
    print(f"  filenames.txt: {(OUT_DIR / 'filenames.txt').stat().st_size / 1e3:.1f} KB")
    print("Cache build complete and verified.")


if __name__ == "__main__":
    main()
