"""Tests for HMDB-51 video Dataset.

Uses a synthetic video generated on-the-fly — no real dataset needed.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from urbanintel.data.hmdb51_video import (
    HMDB51VideoDataset,
    decode_frames,
    sample_frame_indices,
)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a 20-frame synthetic video with a checkerboard pattern."""
    video_path = tmp_path / "test_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (160, 120))
    assert out.isOpened(), "VideoWriter failed to open — codec issue"
    for i in range(20):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        # Checkerboard pattern
        for y in range(0, 120, 20):
            for x in range(0, 160, 20):
                if (x // 20 + y // 20) % 2 == 0:
                    frame[y : y + 20, x : x + 20] = 255
        frame += i * 5  # slight temporal variation
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        out.write(frame)
    out.release()
    assert video_path.exists() and video_path.stat().st_size > 0, "Video file is empty"
    return video_path


def test_sample_frame_indices_length():
    indices = sample_frame_indices(100, n_frames=16)
    assert len(indices) == 16
    assert indices[0] == 0
    assert indices[-1] == 99


def test_sample_frame_indices_short_video():
    """Short video: still returns n_frames (repeated last frame)."""
    indices = sample_frame_indices(5, n_frames=16)
    assert len(indices) == 16
    assert indices[0] == 0
    assert indices[-1] == 4


def test_sample_frame_indices_empty():
    with pytest.raises(ValueError):
        sample_frame_indices(0)


def test_decode_frames_shape(synthetic_video: Path):
    frames = decode_frames(str(synthetic_video), n_frames=16)
    assert frames.shape == (16, 120, 160, 3)
    assert frames.dtype == np.uint8


def test_decode_frames_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        decode_frames(str(tmp_path / "nonexistent.avi"))


def test_dataset_output_shape(synthetic_video: Path):
    df = pd.DataFrame({"video_path": [str(synthetic_video)], "label": [3]})
    ds = HMDB51VideoDataset(df, is_train=False, n_frames=16, resolution=112)
    assert len(ds) == 1

    frames, label = ds[0]
    assert isinstance(frames, torch.Tensor)
    assert frames.shape == (16, 3, 112, 112)
    assert frames.dtype == torch.float32
    assert label == 3


def test_dataset_train_augmentation(synthetic_video: Path):
    """Train mode should produce different tensors on repeated calls."""
    df = pd.DataFrame({"video_path": [str(synthetic_video)], "label": [0]})
    ds = HMDB51VideoDataset(df, is_train=True, n_frames=8, resolution=64)

    torch.manual_seed(0)
    np.random.seed(0)
    frames1, _ = ds[0]

    torch.manual_seed(1)
    np.random.seed(1)
    frames2, _ = ds[0]

    # Different seeds → different augmentation → different tensors
    assert not torch.equal(frames1, frames2)


def test_dataset_val_deterministic(synthetic_video: Path):
    """Val mode: repeated calls produce identical tensors."""
    df = pd.DataFrame({"video_path": [str(synthetic_video)], "label": [0]})
    ds = HMDB51VideoDataset(df, is_train=False, n_frames=8, resolution=64)
    frames1, _ = ds[0]
    frames2, _ = ds[0]
    assert torch.equal(frames1, frames2)
