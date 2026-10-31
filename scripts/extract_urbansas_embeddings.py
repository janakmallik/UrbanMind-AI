"""Extract frozen AST audio + R(2+1)D-18 video embeddings for Urbansas.

Both encoders are loaded from their Week 9c training checkpoints and
frozen. Extraction is reproducible: checkpoint identity, preprocessing
config, and dataset alignment are recorded in embedding_metadata.json.

Audio embeddings: 768-dim, mean-pooled over AST last_hidden_state tokens
Video embeddings: 512-dim, R(2+1)D-18 pooled features (fc replaced by Identity)

Both encoders were trained on Urbansas for their respective Week 9c
baselines. Using the training checkpoints (not fresh pretrained weights)
ensures the embeddings match the models that produced the Week 9c
probabilities used in the late-fusion comparison.

Outputs:
    data/urbansas_results/audio_embeddings.npy  (1080, 768) float32
    data/urbansas_results/video_embeddings.npy  (1080, 512) float32
    data/urbansas_results/embedding_filenames.txt
    data/urbansas_results/embedding_metadata.json

Runtime: ~3 min audio, ~12 min video on RTX 4050.
"""

from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.video import r2plus1d_18

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import (
    CachedASTFeatureDataset,
    CachedUrbansasVideoDataset,
    prepare_urbansas,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = _REPO_ROOT / "data" / "urbansas_results"
AUDIO_CACHE = _REPO_ROOT / "data" / "urbansas_ast_features"
VIDEO_CACHE = _REPO_ROOT / "data" / "urbansas_video_frames"
AUDIO_BATCH = 32
VIDEO_BATCH = 8


def extract_audio_embeddings(df) -> tuple[np.ndarray, list[str]]:
    from urbanintel.models.ast_classifier import ASTClassifier

    ckpt_path = RESULTS_DIR / "audio_pretrained_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"AST training checkpoint not found at {ckpt_path}. "
            "Run scripts/train_urbansas_audio_pretrained.py first."
        )
    print(f"Loading AST encoder from {ckpt_path.name} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classifier = ASTClassifier()
    classifier.load_state_dict(ckpt["model"])
    encoder = classifier.encoder.to(DEVICE).eval()
    print(f"  encoder frozen, {sum(p.numel() for p in encoder.parameters()):,} params")

    ds = CachedASTFeatureDataset(df, cache_dir=AUDIO_CACHE)
    loader = DataLoader(ds, batch_size=AUDIO_BATCH, shuffle=False, num_workers=0)

    all_emb = []
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(loader):
            features = features.to(DEVICE, non_blocking=True)
            out = encoder(input_values=features)
            pooled = out.last_hidden_state.mean(dim=1)  # (B, 768)
            all_emb.append(pooled.cpu().numpy())
            if (batch_idx + 1) % 5 == 0:
                print(f"  audio: {batch_idx + 1}/{len(loader)} batches  ({time.time() - t0:.0f}s)")

    emb = np.concatenate(all_emb, axis=0).astype(np.float32)
    print(f"Audio embeddings: {emb.shape}  ({time.time() - t0:.0f}s)")
    return emb, df["filename"].tolist()


def extract_video_embeddings(df) -> tuple[np.ndarray, list[str]]:
    ckpt_path = RESULTS_DIR / "urbansas_video_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Video training checkpoint not found at {ckpt_path}. "
            "Run scripts/train_urbansas_video.py first."
        )
    print(f"Loading R(2+1)D-18 from {ckpt_path.name} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = r2plus1d_18(weights=None)
    # The saved checkpoint has fc = Linear(512, 2); load with that architecture
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["model"])
    # Replace fc with Identity AFTER loading — extracts 512-dim pooled features
    model.fc = nn.Identity()
    model = model.to(DEVICE).eval()
    print(f"  encoder frozen, {sum(p.numel() for p in model.parameters()):,} params")

    ds = CachedUrbansasVideoDataset(df, resolution=112, is_train=False)
    loader = DataLoader(ds, batch_size=VIDEO_BATCH, shuffle=False, num_workers=4)

    all_emb = []
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (frames, _) in enumerate(loader):
            frames = frames.to(DEVICE, non_blocking=True)
            pooled = model(frames)  # (B, 512)
            all_emb.append(pooled.cpu().numpy())
            if (batch_idx + 1) % 10 == 0:
                print(f"  video: {batch_idx + 1}/{len(loader)} batches  ({time.time() - t0:.0f}s)")

    emb = np.concatenate(all_emb, axis=0).astype(np.float32)
    print(f"Video embeddings: {emb.shape}  ({time.time() - t0:.0f}s)")
    return emb, df["filename"].tolist()


def main():
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not (AUDIO_CACHE / "ast_features.npy").exists():
        raise FileNotFoundError(f"AST feature cache missing at {AUDIO_CACHE}")
    if not (VIDEO_CACHE / "frames.npy").exists():
        raise FileNotFoundError(f"Video frame cache missing at {VIDEO_CACHE}")

    root = urbansas_root()
    split = prepare_urbansas(root, val_fraction=0.0, test_fraction=0.0)
    all_df = split.train_df.reset_index(drop=True).copy()
    print(f"Total clips: {len(all_df)}")

    # Verify composition before extraction
    mavd_df = all_df[all_df["subset"] == "mavd"]
    tau_df = all_df[all_df["subset"] == "tau"]
    print(f"  MAVD: {len(mavd_df)}  ({mavd_df['has_offscreen'].sum()} pos)")
    print(f"  TAU:  {len(tau_df)}  ({tau_df['has_offscreen'].sum()} pos)")
    assert len(mavd_df) == 540, f"expected 540 MAVD, got {len(mavd_df)}"
    assert len(tau_df) == 540, f"expected 540 TAU, got {len(tau_df)}"
    assert all_df["filename"].is_unique, "duplicate filenames in dataset"

    print("\n" + "=" * 70)
    print("Extracting audio embeddings (AST)")
    print("=" * 70)
    audio_emb, audio_fns = extract_audio_embeddings(all_df)

    print("\n" + "=" * 70)
    print("Extracting video embeddings (R(2+1)D-18)")
    print("=" * 70)
    video_emb, video_fns = extract_video_embeddings(all_df)

    # Strict alignment checks
    assert audio_fns == video_fns, "audio and video filename orders differ"
    assert len(audio_fns) == 1080, f"expected 1080, got {len(audio_fns)}"
    assert len(audio_fns) == len(set(audio_fns)), "duplicate filenames"
    assert audio_emb.shape == (1080, 768), f"audio shape {audio_emb.shape}"
    assert video_emb.shape == (1080, 512), f"video shape {video_emb.shape}"
    assert np.isfinite(audio_emb).all(), "audio has inf or nan"
    assert np.isfinite(video_emb).all(), "video has inf or nan"

    # Alignment with Week 9c prediction files
    audio_oof = pd.read_csv(RESULTS_DIR / "audio_pretrained_oof.csv")
    video_oof = pd.read_csv(RESULTS_DIR / "urbansas_video_oof.csv")
    mavd_set = set(audio_oof["filename"])
    assert mavd_set == set(video_oof["filename"]), "OOF files don't align"
    assert mavd_set.issubset(set(audio_fns)), "MAVD OOF fns missing from embeddings"
    assert len(mavd_set) == 540
    print(f"\nAlignment with Week 9c prediction files: OK")
    print(f"  540 MAVD clips present in both embeddings and OOF")
    print(f"  540 TAU clips present in embeddings (not in OOF)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "audio_embeddings.npy", audio_emb)
    np.save(RESULTS_DIR / "video_embeddings.npy", video_emb)
    with open(RESULTS_DIR / "embedding_filenames.txt", "w") as f:
        f.write("\n".join(audio_fns))

    # Metadata
    metadata = {
        "extraction_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "device": DEVICE,
        "audio_encoder": {
            "checkpoint": "data/urbansas_results/audio_pretrained_final.pt",
            "base_architecture": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "finetuned_on": "Urbansas MAVD (Week 9c)",
            "head_discarded": True,
            "embedding_dim": 768,
            "pooling": "mean over last_hidden_state tokens",
        },
        "video_encoder": {
            "checkpoint": "data/urbansas_results/urbansas_video_final.pt",
            "base_architecture": "torchvision.models.video.r2plus1d_18 (Kinetics-400)",
            "finetuned_on": "Urbansas MAVD (Week 9c)",
            "head_discarded": True,
            "embedding_dim": 512,
            "pooling": "built-in global avg pool before fc",
        },
        "dataset": {
            "n_clips": 1080,
            "mavd_clips": 540,
            "tau_clips": 540,
            "filename_order": "matches embedding_filenames.txt",
        },
        "preprocessing": {
            "audio": "AST feature cache from precompute_urbansas_ast_features.py",
            "video": "16 frames @ 112×112, Kinetics mean [0.43216, 0.394666, 0.37645], std [0.22803, 0.22145, 0.216989]",
        },
    }
    with open(RESULTS_DIR / "embedding_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved:")
    print(f"  audio_embeddings.npy:     {audio_emb.nbytes / 1e6:.1f} MB")
    print(f"  video_embeddings.npy:     {video_emb.nbytes / 1e6:.1f} MB")
    print(f"  embedding_filenames.txt:  {len(audio_fns)} filenames")
    print(f"  embedding_metadata.json:  provenance + preprocessing")
    print(f"\nEmbedding stats:")
    print(f"  audio: mean={audio_emb.mean():.4f}  std={audio_emb.std():.4f}")
    print(f"  video: mean={video_emb.mean():.4f}  std={video_emb.std():.4f}")


if __name__ == "__main__":
    main()
