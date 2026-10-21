"""UrbanSound8K audio dataset loader.

Two datasets:
  - AudioDataset: loads from parquet (slower, for exploration)
  - CachedAudioDataset: loads from pre-decoded memmap (fast, for training)

Run scripts/precompute_audio.py once before using CachedAudioDataset.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

URBANSOUND_CLASSES = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]
NUM_CLASSES = len(URBANSOUND_CLASSES)

SAMPLE_RATE = 22050
DURATION = 4.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 1024
FMIN = 20
FMAX = 8000

PARQUET_GLOB = "C:/Users/Janak Mallik/ml-data/urbansound8k/data/train-*.parquet"
CACHE_DIR = Path("C:/Users/Janak Mallik/ml-data/urbansound8k-cache")


def audio_to_mel(array: np.ndarray, sr: int) -> np.ndarray:
    """Raw audio (mono or stereo) -> log-mel spectrogram (N_MELS, T) float32."""
    # Stereo -> mono
    if array.ndim > 1:
        array = array.mean(axis=1)

    # Resample to SAMPLE_RATE
    if sr != SAMPLE_RATE:
        array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Pad or trim
    if len(array) < N_SAMPLES:
        array = np.pad(array, (0, N_SAMPLES - len(array)))
    else:
        array = array[:N_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=array,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def load_dataframe() -> pd.DataFrame:
    """Load metadata as a DataFrame. Uses cache meta.csv if available for row alignment."""
    if (CACHE_DIR / "meta.csv").exists():
        return pd.read_csv(CACHE_DIR / "meta.csv")
    import glob

    files = sorted(glob.glob(PARQUET_GLOB))
    if not files:
        raise FileNotFoundError(f"No parquet at {PARQUET_GLOB}")
    dfs = [pd.read_parquet(f, columns=["classID", "fold", "slice_file_name"]) for f in files]
    return pd.concat(dfs, ignore_index=True)


def _augment(mel: np.ndarray) -> np.ndarray:
    """SpecAugment: random time and frequency masks."""
    if np.random.rand() < 0.5:
        w = np.random.randint(1, max(2, mel.shape[1] // 10))
        s = np.random.randint(0, mel.shape[1] - w)
        mel[:, s : s + w] = mel.min()
    if np.random.rand() < 0.3:
        w = np.random.randint(1, max(2, N_MELS // 10))
        s = np.random.randint(0, N_MELS - w)
        mel[s : s + w, :] = mel.min()
    return mel


class CachedAudioDataset(Dataset):
    """Fast dataset that loads pre-decoded audio via memory-mapped numpy.

    Requires scripts/precompute_audio.py to have been run first.

    Note: memmap is loaded lazily per worker because numpy memmap objects
    cannot be pickled by PyTorch DataLoader workers on Windows.
    """

    def __init__(self, indices: list[int], is_train: bool = False):
        self.indices = indices
        self.is_train = is_train

        npy_path = CACHE_DIR / "audio.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"{npy_path} not found. Run scripts/precompute_audio.py first.")
        self._npy_path = str(npy_path)
        self._audio = None
        self._meta = None

    def _get_audio(self):
        if self._audio is None:
            self._audio = np.load(self._npy_path, mmap_mode="r")
        return self._audio

    def _get_meta(self):
        if self._meta is None:
            self._meta = pd.read_csv(CACHE_DIR / "meta.csv")
        return self._meta

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        row_idx = self.indices[i]
        array = np.asarray(self._get_audio()[row_idx])
        mel = audio_to_mel(array, SAMPLE_RATE)
        if self.is_train:
            mel = _augment(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        label = int(self._get_meta().iloc[row_idx]["classID"])
        return torch.from_numpy(mel).unsqueeze(0), label
