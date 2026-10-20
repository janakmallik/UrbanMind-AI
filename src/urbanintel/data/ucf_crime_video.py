"""UCF-Crime clip Dataset for R(2+1)D-18 training."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from urbanintel.data.hmdb51_video import KINETICS_MEAN, KINETICS_STD

N_FRAMES = 16
RESOLUTION = 112


class UCFCrimeClipDataset(Dataset):
    """Returns (frames_tensor, category) for each clip.

    frames_tensor shape: (16, 3, 112, 112), Kinetics-normalized.
    """

    def __init__(
        self,
        clips_df,
        is_train: bool = False,
        resolution: int = RESOLUTION,
    ):
        self.clips = clips_df.reset_index(drop=True)
        self.is_train = is_train
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.clips)

    def _process(self, frame_paths: list[str]) -> torch.Tensor:
        R = self.resolution
        frames = []
        for p in frame_paths:
            img = cv2.imread(p)
            if img is None:
                raise RuntimeError(f"Cannot read {p}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (R, R), interpolation=cv2.INTER_AREA)
            frames.append(img)
        x = np.stack(frames).astype(np.float32) / 255.0

        if self.is_train:
            # Random crop
            pad = 8
            r2 = R + pad
            padded = (
                np.stack(
                    [cv2.resize(f, (r2, r2), interpolation=cv2.INTER_AREA) for f in frames]
                ).astype(np.float32)
                / 255.0
            )
            top = np.random.randint(0, pad + 1)
            left = np.random.randint(0, pad + 1)
            x = padded[:, top : top + R, left : left + R]
            # Random flip
            if np.random.rand() < 0.5:
                x = np.ascontiguousarray(x[:, :, ::-1, :])

        x = (x - KINETICS_MEAN) / KINETICS_STD
        x = x.transpose(0, 3, 1, 2)  # (T, C, H, W)
        return torch.from_numpy(x).float()

    def __getitem__(self, idx: int):
        row = self.clips.iloc[idx]
        frames = self._process(row["frame_paths"])
        return frames, int(row["category"])
