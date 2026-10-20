"""PyTorch Dataset for HMDB-51 video clips.

Each sample: (frames_tensor, label)
  frames_tensor: (16, 3, 112, 112) float32
  label: int in [0, NUM_CLASSES)

Decoding uses OpenCV. Frame sampling is uniform across clip duration.
Train augmentations: random crop + horizontal flip.
Val/test: deterministic center crop, no augmentation.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

N_FRAMES = 16
RESOLUTION = 112
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def sample_frame_indices(total_frames: int, n_frames: int = N_FRAMES) -> np.ndarray:
    """Uniformly sample n_frames indices from [0, total_frames)."""
    if total_frames <= 0:
        raise ValueError(f"Cannot sample from empty video (total_frames={total_frames})")
    if total_frames < n_frames:
        # Repeat last frame to fill
        indices = np.linspace(0, total_frames - 1, n_frames).astype(np.int64)
    else:
        indices = np.linspace(0, total_frames - 1, n_frames).astype(np.int64)
    return indices


import decord


def decode_frames(video_path: str, n_frames: int = N_FRAMES) -> np.ndarray | None:
    """Decode n_frames uniformly using decord (reliable + fast).

    Returns (n_frames, H, W, 3) uint8 RGB array, or None if unreadable.
    """
    if not Path(video_path).is_file():
        return None

    try:
        vr = decord.VideoReader(str(video_path), num_threads=1)
    except Exception:
        return None

    total = len(vr)
    if total == 0:
        return None

    indices = np.linspace(0, total - 1, n_frames).astype(int).tolist()
    try:
        frames = vr.get_batch(indices).asnumpy()
    except Exception:
        return None

    return frames.astype(np.uint8)  # already RGB, (T, H, W, 3)


class HMDB51VideoDataset(Dataset):
    """HMDB-51 video dataset returning (frames, label) tensors.

    Args:
        df: DataFrame with columns 'video_path' and 'label'.
        is_train: if True, apply augmentation.
        n_frames: number of frames per clip.
        resolution: spatial resolution after resize.
        normalization: "imagenet" (default) or "kinetics".
            Kinetics-400 normalization matches the input distribution
            expected by torchvision's R(2+1)D-18 pretrained weights.
    """

    def __init__(
        self,
        df,
        is_train: bool = False,
        n_frames: int = N_FRAMES,
        resolution: int = RESOLUTION,
        normalization: str = "imagenet",
    ):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.n_frames = n_frames
        self.resolution = resolution

        if normalization == "imagenet":
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        elif normalization == "kinetics":
            self.mean, self.std = KINETICS_MEAN, KINETICS_STD
        else:
            raise ValueError(
                f"Unknown normalization '{normalization}'. Choose 'imagenet' or 'kinetics'."
            )

    def __len__(self) -> int:
        return len(self.df)

    def _process_frames(self, frames: np.ndarray) -> torch.Tensor:
        """frames: (T, H, W, 3) uint8 -> (T, 3, R, R) float32, normalized."""
        T, H, W, C = frames.shape
        R = self.resolution

        # Resize all frames first
        resized = np.empty((T, R, R, C), dtype=np.uint8)
        for i in range(T):
            resized[i] = cv2.resize(frames[i], (R, R), interpolation=cv2.INTER_AREA)

        if self.is_train:
            # Random crop
            pad = 8
            r2 = R + pad
            padded = np.empty((T, r2, r2, C), dtype=np.uint8)
            for i in range(T):
                padded[i] = cv2.resize(resized[i], (r2, r2), interpolation=cv2.INTER_AREA)
            top = np.random.randint(0, pad + 1)
            left = np.random.randint(0, pad + 1)
            resized = padded[:, top : top + R, left : left + R]

            # Random horizontal flip
            if np.random.rand() < 0.5:
                resized = np.ascontiguousarray(resized[:, :, ::-1, :])

        # Convert to float and normalize using instance attributes
        x = resized.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        # (T, H, W, C) -> (T, C, H, W)
        x = x.transpose(0, 3, 1, 2)
        return torch.from_numpy(x).float()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        frames = decode_frames(row["video_path"], n_frames=self.n_frames)
        if frames is None:
            raise RuntimeError(
                f"Corrupt video at index {idx}: {row['video_path']}. "
                "This should have been filtered out during preprocessing."
            )
        tensor = self._process_frames(frames)
        return tensor, int(row["label"])


def filter_corrupt_videos(df, n_frames: int = N_FRAMES, verbose: bool = True):
    """Remove rows whose videos cannot be decoded.

    Uses the same decoder as the Dataset (decode_frames), so filtering
    and training see the same set of usable videos.
    """
    import pandas as pd

    valid_rows = []
    corrupt = []

    for _, row in df.iterrows():
        frames = decode_frames(row["video_path"], n_frames=n_frames)
        if frames is None:
            corrupt.append(row["video_path"])
        else:
            valid_rows.append(row)

    if verbose:
        print(f"Filtered: {len(df)} -> {len(valid_rows)} (dropped {len(corrupt)} corrupt)")
        for p in corrupt[:5]:
            print(f"  {p}")
        if len(corrupt) > 5:
            print(f"  ... and {len(corrupt) - 5} more")

    return pd.DataFrame(valid_rows).reset_index(drop=True) if valid_rows else pd.DataFrame()
