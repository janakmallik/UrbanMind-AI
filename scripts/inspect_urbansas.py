"""Sanity check — load 5 random Urbansas clips and save sample outputs.

For each clip:
    - Save the first video frame
    - Save the middle video frame
    - Save the audio log-mel spectrogram (as PNG)
    - Write the audio annotations to a text file

Open the sample folder and verify visually that:
    - Video frame shows a real urban traffic scene
    - Mel spectrogram shows real audio signal (not noise)
    - Labels match what you see/hear
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import (
    audio_to_mel,
    load_audio,
    sample_video_frames,
)

OUT_DIR = _REPO_ROOT / "data" / "urbansas_samples"


def main() -> None:
    root = urbansas_root()
    print(f"Urbansas root: {root}")

    annotations_csv = root / "annotations" / "audio_annotations.csv"
    if not annotations_csv.exists():
        raise FileNotFoundError(f"Missing: {annotations_csv}")

    audio_ann = pd.read_csv(annotations_csv)
    unique_files = audio_ann["filename"].unique()
    print(f"Total annotated clips: {len(unique_files)}")

    # Sample 5 clips deterministically
    rng = np.random.default_rng(42)
    sample_files = rng.choice(unique_files, size=5, replace=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(sample_files, 1):
        print(f"\n=== Sample {i}: {filename} ===")

        audio_path = root / "audio" / f"{filename}.wav"
        video_path = None
        for folder in ["video_24fps_1", "video_24fps_2", "video_24fps_3"]:
            candidate = root / folder / f"{filename}.mp4"
            if candidate.exists():
                video_path = candidate
                break

        if audio_path is None or not audio_path.exists():
            print(f"  MISSING audio: {audio_path}")
            continue
        if video_path is None:
            print(f"  MISSING video")
            continue

        clip_dir = OUT_DIR / filename
        clip_dir.mkdir(exist_ok=True)

        # 1. Video frames
        print("  Sampling 16 video frames...")
        frames = sample_video_frames(video_path, n_frames=16)
        cv2.imwrite(str(clip_dir / "frame_00.png"), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(clip_dir / "frame_15.png"), cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))

        # 2. Audio mel
        print("  Computing audio mel spectrogram...")
        audio = load_audio(audio_path)
        mel = audio_to_mel(audio, 48_000)
        plt.figure(figsize=(10, 4))
        plt.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(label="dB")
        plt.title(f"Log-mel — {filename}")
        plt.xlabel("Time frames")
        plt.ylabel("Mel bins")
        plt.tight_layout()
        plt.savefig(clip_dir / "mel.png", dpi=80)
        plt.close()

        # 3. Annotations
        clip_ann = audio_ann[audio_ann["filename"] == filename]
        labels = clip_ann["label"].unique()
        has_offscreen = "offscreen" in labels
        with open(clip_dir / "annotations.txt", "w") as f:
            f.write(f"Filename: {filename}\n")
            f.write(f"Has offscreen vehicle: {has_offscreen}\n")
            f.write(f"\nAudio annotations:\n")
            f.write(clip_ann.to_string())
        print(f"  Has offscreen: {has_offscreen}")
        print(f"  Labels present: {sorted(labels)}")

    print(f"\n=== Done ===")
    print(f"Sample outputs in: {OUT_DIR}")
    print(f"\nOpen the folders and verify visually:")
    print(f"  - frame_00.png, frame_15.png: should show real urban traffic")
    print(f"  - mel.png: should show spectral structure, not noise")
    print(f"  - annotations.txt: should list vehicles/offscreen/empty")


if __name__ == "__main__":
    main()
