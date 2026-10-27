"""Urbansas dataset loader for offscreen vehicle detection.

Task: Given a 10-second clip (video + audio), predict whether it contains
an **offscreen vehicle** — a vehicle that is heard in the audio but not
visible in any video frame.

This is the cross-modal test:
- Video alone CANNOT detect offscreen vehicles (no visual signal)
- Audio alone CAN detect them (but may confuse with visible-vehicle sounds)
- Fusion should outperform either

Dataset structure (assumes `urbansas_root` from config):
    urbansas/
    ├── annotations/
    │   ├── audio_annotations.csv
    │   └── video_annotations.csv
    ├── audio/            1,080 .wav files (48kHz stereo 10s)
    ├── video_24fps_1/    ~300 .mp4 files
    ├── video_24fps_2/    ~240 .mp4 files
    └── video_24fps_3/    ~540 .mp4 files

Video files are split across three folders. This loader builds a filename
→ path index across all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from urbanintel.config import urbansas_root

# --- Constants ---

AUDIO_SAMPLE_RATE = 48_000
CLIP_DURATION_S = 10.0
N_SAMPLES = int(AUDIO_SAMPLE_RATE * CLIP_DURATION_S)  # 480,000
N_MELS = 128
HOP_LENGTH = 2048  # For 48kHz audio; produces ~234 frames (vs. 938 at hop=512)
N_FFT = 2048
FMIN = 20
FMAX = 24_000  # Nyquist for 48kHz

VIDEO_FPS = 24
VIDEO_FRAMES_PER_CLIP = 240  # 10s @ 24fps
NUM_VIDEO_FRAMES = 16  # sampled frames per clip


@dataclass(frozen=True)
class UrbansasSplit:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame


# --- Annotation loading ---


def _build_video_index(root: Path) -> dict[str, Path]:
    """Index all .mp4 files by filename stem (without extension)."""
    index: dict[str, Path] = {}
    for folder in ["video_24fps_1", "video_24fps_2", "video_24fps_3"]:
        folder_path = root / folder
        if not folder_path.exists():
            continue
        for p in folder_path.glob("*.mp4"):
            index[p.stem] = p
    return index


def _build_audio_index(root: Path) -> dict[str, Path]:
    """Index all .wav files by filename stem."""
    audio_dir = root / "audio"
    return {p.stem: p for p in audio_dir.glob("*.wav")}


def _build_clip_labels(audio_annotations: pd.DataFrame) -> pd.DataFrame:
    """Build per-clip labels from audio annotations.

    Each clip is labeled:
        has_offscreen: bool — any annotation with label == "offscreen"
        has_vehicle: bool — any annotation with a vehicle label
        num_vehicles: int — count of distinct vehicle annotations
        is_empty: bool — only label == "-1" (no objects)
    """
    rows = []
    for filename, group in audio_annotations.groupby("filename"):
        labels = set(group["label"].astype(str).unique())
        rows.append(
            {
                "filename": filename,
                "has_offscreen": "offscreen" in labels,
                "has_vehicle": bool(labels & {"car", "bus", "motorbike", "truck"}),
                "num_vehicle_events": int(
                    group[group["label"].isin(["car", "bus", "motorbike", "truck"])].shape[0]
                ),
                "is_empty": labels == {"-1"},
            }
        )
    return pd.DataFrame(rows)


def _load_clip_metadata(root: Path) -> pd.DataFrame:
    """Build a DataFrame with one row per annotated clip.

    Columns:
        filename: str — clip ID (matches .wav and .mp4 stems)
        audio_path: str
        video_path: str
        has_offscreen: bool — the task label
        has_vehicle, num_vehicle_events, is_empty
        location_id: str — from video annotations (used for splitting)
        city: str
        subset: str — "mavd" or "tau"
        night: int — 0 or 1
    """
    audio_csv = root / "annotations" / "audio_annotations.csv"
    video_csv = root / "annotations" / "video_annotations.csv"

    if not audio_csv.exists() or not video_csv.exists():
        raise FileNotFoundError(f"Annotations not found. Expected:\n  {audio_csv}\n  {video_csv}")

    audio_ann = pd.read_csv(audio_csv)
    video_ann = pd.read_csv(video_csv)

    clip_labels = _build_clip_labels(audio_ann)

    # Extract per-clip metadata from video annotations (one row per clip is enough)
    clip_meta = video_ann.drop_duplicates(subset=["filename"]).set_index("filename")[
        ["location_id", "city", "subset", "night"]
    ]

    audio_index = _build_audio_index(root)
    video_index = _build_video_index(root)

    rows = []
    for _, row in clip_labels.iterrows():
        filename = row["filename"]
        audio_path = audio_index.get(filename)
        video_path = video_index.get(filename)
        if audio_path is None or video_path is None:
            continue

        meta_row = clip_meta.loc[filename] if filename in clip_meta.index else None
        rows.append(
            {
                "filename": filename,
                "audio_path": str(audio_path),
                "video_path": str(video_path),
                "has_offscreen": bool(row["has_offscreen"]),
                "has_vehicle": bool(row["has_vehicle"]),
                "num_vehicle_events": int(row["num_vehicle_events"]),
                "is_empty": bool(row["is_empty"]),
                "location_id": str(meta_row["location_id"]) if meta_row is not None else "unknown",
                "city": str(meta_row["city"]) if meta_row is not None else "unknown",
                "subset": str(meta_row["subset"]) if meta_row is not None else "unknown",
                "night": int(meta_row["night"]) if meta_row is not None else 0,
            }
        )

    return pd.DataFrame(rows).reset_index(drop=True)


def prepare_urbansas(
    root: Path | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> UrbansasSplit:
    """Load and split the 1,080-clip annotated subset by location.

    Splitting is by `location_id` (not by clip) to prevent scene-level
    leakage. All clips from the same camera location end up in the same
    split.
    """
    from sklearn.model_selection import train_test_split

    if root is None:
        root = urbansas_root()

    df = _load_clip_metadata(root)
    print(f"Loaded {len(df)} annotated clips")
    print(
        f"  With offscreen vehicle: {df['has_offscreen'].sum()} "
        f"({100 * df['has_offscreen'].mean():.1f}%)"
    )
    print(f"  Unique locations: {df['location_id'].nunique()}")

    # Skip splitting when caller wants the full dataset (val/test handled externally)
    if val_fraction == 0.0 and test_fraction == 0.0:
        print(f"  Returning all {len(df)} clips as train_df (no internal split)")
        empty = df.iloc[0:0].copy()
        return UrbansasSplit(train_df=df, val_df=empty, test_df=empty)

    # Split by location_id
    unique_locs = df["location_id"].unique()
    if len(unique_locs) < 3:
        raise ValueError(f"Need at least 3 locations for split, got {len(unique_locs)}")

    loc_train, loc_holdout = train_test_split(
        unique_locs,
        test_size=val_fraction + test_fraction,
        random_state=random_state,
    )
    loc_val, loc_test = train_test_split(
        loc_holdout,
        test_size=test_fraction / (val_fraction + test_fraction),
        random_state=random_state,
    )

    train_df = df[df["location_id"].isin(loc_train)].reset_index(drop=True)
    val_df = df[df["location_id"].isin(loc_val)].reset_index(drop=True)
    test_df = df[df["location_id"].isin(loc_test)].reset_index(drop=True)

    print(f"  Train: {len(train_df)} clips ({train_df['has_offscreen'].sum()} offscreen)")
    print(f"  Val:   {len(val_df)} clips ({val_df['has_offscreen'].sum()} offscreen)")
    print(f"  Test:  {len(test_df)} clips ({test_df['has_offscreen'].sum()} offscreen)")

    return UrbansasSplit(train_df=train_df, val_df=val_df, test_df=test_df)


# --- Feature extraction ---


def audio_to_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    """Convert raw audio to log-mel spectrogram (N_MELS, T)."""
    import librosa

    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    if sr != AUDIO_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)

    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32),
        sr=AUDIO_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    """Load a 10s wav as mono float32."""
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != AUDIO_SAMPLE_RATE:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=AUDIO_SAMPLE_RATE)
    if len(data) < N_SAMPLES:
        data = np.pad(data, (0, N_SAMPLES - len(data)))
    else:
        data = data[:N_SAMPLES]
    return data.astype(np.float32)


def sample_video_frames(path: str | Path, n_frames: int = NUM_VIDEO_FRAMES) -> np.ndarray:
    """Sample n_frames uniformly from the video.

    Returns (n_frames, H, W, 3) uint8 RGB.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise ValueError(f"Empty video: {path}")

    indices = np.linspace(0, total - 1, n_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Repeat last valid frame
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    return np.stack(frames, axis=0)


# --- Dataset classes ---


class UrbansasAudioDataset(Dataset):
    """Audio-only dataset for the offscreen detection task."""

    def __init__(self, df: pd.DataFrame, is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio = load_audio(row["audio_path"])
        mel = audio_to_mel(audio, AUDIO_SAMPLE_RATE)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        label = int(row["has_offscreen"])
        return torch.from_numpy(mel).unsqueeze(0), label


def _decode_frames_decord(video_path: str | Path, n_frames: int = 16) -> np.ndarray | None:
    """Decode n_frames uniformly via decord. Returns (T, H, W, 3) uint8 RGB or None.

    Falls back to cv2-based sample_video_frames if decord fails (e.g., codec
    unsupported). decord is ~5-10x faster than cv2 seek-per-frame.
    """
    try:
        import decord
    except ImportError:
        return sample_video_frames(video_path, n_frames)

    p = Path(video_path)
    if not p.is_file():
        return None

    try:
        vr = decord.VideoReader(str(p), num_threads=1)
        total = len(vr)
        if total == 0:
            return None
        indices = np.linspace(0, total - 1, n_frames).astype(int).tolist()
        frames = vr.get_batch(indices).asnumpy()
        return frames.astype(np.uint8)
    except Exception:
        # decord failed — fall back to cv2
        try:
            return sample_video_frames(video_path, n_frames)
        except Exception:
            return None


class UrbansasVideoDataset(Dataset):
    """Video-only dataset for the offscreen detection task.

    Returns (3, 16, 112, 112) float32 frames + label.
    Train: random crop (pad=8) + horizontal flip 50%.
    Eval: deterministic center crop, no augmentation.
    """

    def __init__(self, df: pd.DataFrame, resolution: int = 112, is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.resolution = resolution
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _process_frames(self, frames: np.ndarray) -> torch.Tensor:
        """frames: (T, H, W, 3) uint8 -> (C, T, R, R) float32, Kinetics-normalized."""
        import cv2

        T, _, _, _ = frames.shape
        R = self.resolution

        resized = np.empty((T, R, R, 3), dtype=np.uint8)
        for i in range(T):
            resized[i] = cv2.resize(frames[i], (R, R), interpolation=cv2.INTER_AREA)

        if self.is_train:
            # Random crop (pad=8 then crop to R)
            pad = 8
            r2 = R + pad
            padded = np.empty((T, r2, r2, 3), dtype=np.uint8)
            for i in range(T):
                padded[i] = cv2.resize(resized[i], (r2, r2), interpolation=cv2.INTER_AREA)
            top = np.random.randint(0, pad + 1)
            left = np.random.randint(0, pad + 1)
            resized = padded[:, top : top + R, left : left + R]

            # Random horizontal flip
            if np.random.rand() < 0.5:
                resized = np.ascontiguousarray(resized[:, :, ::-1, :])

        frames_np = resized.astype(np.float32) / 255.0

        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        frames_np = (frames_np - mean) / std

        # (T, H, W, C) -> (C, T, H, W)  [torchvision R(2+1)D-18 expects this]
        return torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        frames = sample_video_frames(row["video_path"], NUM_VIDEO_FRAMES)
        frames_t = self._process_frames(frames)
        label = int(row["has_offscreen"])
        return frames_t, label


class UrbansasFusionDataset(Dataset):
    """Fusion dataset returning (video_frames, audio_mel, label)."""

    def __init__(self, df: pd.DataFrame, resolution: int = 112, is_train: bool = False):
        self.df = df.reset_index(drop=True)
        self.resolution = resolution
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import cv2

        row = self.df.iloc[idx]

        # Audio
        audio = load_audio(row["audio_path"])
        mel = audio_to_mel(audio, AUDIO_SAMPLE_RATE)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel_t = torch.from_numpy(mel).unsqueeze(0)

        # Video
        frames = _decode_frames_decord(row["video_path"], NUM_VIDEO_FRAMES)
        if frames is None:
            raise RuntimeError(f"Could not decode video at index {idx}: {row['video_path']}")
        resized = []
        for frame in frames:
            r = cv2.resize(frame, (self.resolution, self.resolution))
            resized.append(r)
        frames_np = np.stack(resized, axis=0).astype(np.float32) / 255.0

        mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
        std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
        frames_np = (frames_np - mean) / std
        frames_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()

        label = int(row["has_offscreen"])
        return frames_t, mel_t, label


class CachedUrbansasAudioDataset(Dataset):
    """Fast dataset using precomputed mel cache."""

    def __init__(self, df: pd.DataFrame, cache_dir: Path | None = None, is_train: bool = False):
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[3] / "data" / "urbansas_mels"
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self._cache_dir = cache_dir
        self._mels = None
        self._filename_to_idx = None

    def _ensure_loaded(self):
        if self._mels is None:
            self._mels = np.load(self._cache_dir / "mels.npy", mmap_mode="r")
            with open(self._cache_dir / "filenames.txt") as f:
                filenames = f.read().strip().split("\n")
            self._filename_to_idx = {name: i for i, name in enumerate(filenames)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self._ensure_loaded()
        row = self.df.iloc[idx]
        mel = np.asarray(self._mels[self._filename_to_idx[row["filename"]]])
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        label = int(row["has_offscreen"])
        return torch.from_numpy(mel).unsqueeze(0), label


class CachedUrbansasVideoDataset(Dataset):
    """Video dataset backed by the precomputed frame cache.

    Expects data/urbansas_video_frames/frames.npy and filenames.txt
    (created by scripts/precompute_urbansas_video_frames.py).

    Frames are stored pre-normalized (Kinetics mean/std applied).
    Augmentation (random crop, hflip) is applied at load time.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Path | None = None,
        resolution: int = 112,
        is_train: bool = False,
    ):
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parents[3] / "data" / "urbansas_video_frames"
        self.df = df.reset_index(drop=True)
        self.resolution = resolution
        self.is_train = is_train
        self._cache_dir = cache_dir
        self._frames = None
        self._stem_to_idx = None

    def _ensure_loaded(self):
        if self._frames is None:
            self._frames = np.load(self._cache_dir / "frames.npy", mmap_mode="r")
            with open(self._cache_dir / "filenames.txt") as f:
                stems = f.read().strip().split("\n")
            self._stem_to_idx = {s: i for i, s in enumerate(stems)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        import cv2

        self._ensure_loaded()
        row = self.df.iloc[idx]
        stem = row["filename"]
        if stem not in self._stem_to_idx:
            raise KeyError(f"Clip {stem} not in video frame cache")
        # (16, 112, 112, 3) float32, already Kinetics-normalized
        frames = np.asarray(self._frames[self._stem_to_idx[stem]])

        R = self.resolution
        if R != frames.shape[1]:
            # Resize if resolution differs
            resized = np.empty((frames.shape[0], R, R, 3), dtype=np.float32)
            for i in range(frames.shape[0]):
                resized[i] = cv2.resize(frames[i], (R, R), interpolation=cv2.INTER_AREA)
            frames = resized

        if self.is_train:
            # Random crop: pad=8, crop to R
            pad = 8
            r2 = R + pad
            padded = np.empty((frames.shape[0], r2, r2, 3), dtype=np.float32)
            for i in range(frames.shape[0]):
                padded[i] = cv2.resize(frames[i], (r2, r2), interpolation=cv2.INTER_AREA)
            top = np.random.randint(0, pad + 1)
            left = np.random.randint(0, pad + 1)
            frames = padded[:, top : top + R, left : left + R]

            if np.random.rand() < 0.5:
                frames = np.ascontiguousarray(frames[:, :, ::-1, :])

        # (T, H, W, C) -> (C, T, H, W)
        frames_t = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).float()
        label = int(row["has_offscreen"])
        return frames_t, label
