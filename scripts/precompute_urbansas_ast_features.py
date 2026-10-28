"""Precompute AST features (input_values) for all 1,080 Urbansas clips.

AST's feature extractor runs on CPU and takes ~900ms per 10s clip. Precomputing
once is required to make training tractable.

Cache shape: (1080, 1024, 128) float32, matching AST's input_values spec.
Cache size:  ~565 MB on disk.

The dataset (CachedASTFeatureDataset) reads from this cache — no per-batch
feature extraction during training.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from multiprocessing import Pool

import numpy as np
from transformers import ASTFeatureExtractor

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
OUT_DIR = _REPO_ROOT / "data" / "urbansas_ast_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO16K_DIR = _REPO_ROOT / "data" / "urbansas_audio16k"


def _process_one(args):
    stem, audio_array = args
    fe = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    feats = fe(audio_array, sampling_rate=16_000, return_tensors="pt")
    iv = feats["input_values"].squeeze(0).numpy()  # (1024, 128)
    return stem, iv.astype(np.float32)


def main() -> None:
    if not (AUDIO16K_DIR / "audio.npy").exists():
        raise FileNotFoundError(
            f"16 kHz audio cache not found at {AUDIO16K_DIR}. "
            "Run scripts/precompute_urbansas_audio16k.py first."
        )

    audio_cache = np.load(AUDIO16K_DIR / "audio.npy", mmap_mode="r")
    with open(AUDIO16K_DIR / "filenames.txt") as f:
        stems = f.read().strip().split("\n")

    assert len(stems) == 1080, f"expected 1080 stems, got {len(stems)}"
    assert audio_cache.shape == (1080, 160000), f"unexpected audio cache shape {audio_cache.shape}"

    print(f"Loaded {len(stems)} clips from 16 kHz audio cache")

    tasks = [(stem, np.asarray(audio_cache[i])) for i, stem in enumerate(stems)]

    print("Extracting AST features with multiprocessing.Pool(8)...")
    import time

    t0 = time.time()
    with Pool(8) as pool:
        results = pool.map(_process_one, tasks, chunksize=4)
    print(f"Done in {time.time() - t0:.0f}s")

    # Verify
    assert len(results) == 1080, f"result count {len(results)} != 1080"
    failed = [stem for stem, x in results if x is None]
    if failed:
        raise RuntimeError(f"{len(failed)} clips failed: {failed[:5]}")

    feature_stems = [s for s, _ in results]
    features = np.stack([x for _, x in results], axis=0).astype(np.float32)
    print(f"Features array: {features.shape}  ({features.nbytes / 1e6:.1f} MB)")

    # Verify shape matches AST spec
    expected_shape = (1080, 1024, 128)
    if features.shape != expected_shape:
        raise RuntimeError(f"features shape {features.shape} != expected {expected_shape}")

    if np.isnan(features).any():
        raise RuntimeError(f"features contain {int(np.isnan(features).sum())} NaNs")

    np.save(OUT_DIR / "ast_features.npy", features)
    with open(OUT_DIR / "filenames.txt", "w") as f:
        f.write("\n".join(feature_stems))

    print(f"Saved to: {OUT_DIR}")
    print(f"  ast_features.npy: {(OUT_DIR / 'ast_features.npy').stat().st_size / 1e6:.1f} MB")
    print(f"  filenames.txt: {(OUT_DIR / 'filenames.txt').stat().st_size / 1e3:.1f} KB")
    print("AST feature cache build complete and verified.")


if __name__ == "__main__":
    main()
