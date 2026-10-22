"""UrbanMind-AI quickstart — run one trained model end-to-end.

Loads the Week 7.6 UrbanSound8K audio classifier and predicts on a
bundled 3-second sample. Prints the predicted class and confidence.
Total runtime: <30s on first run, <2s after (model is cached by HF).

Usage:
    uv run python scripts/quickstart.py

If you don't have a pretrained checkpoint at
`checkpoints/urbansound8k_audio_cnn_best.pt`, the script will tell you
where to get it (retrain via scripts/train_audio_events.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from urbanintel.data.urbansound8k import (  # noqa: E402
    N_MELS,
    NUM_CLASSES,
    SAMPLE_RATE,
    URBANSOUND_CLASSES,
    audio_to_mel,
)

CHECKPOINT = _REPO_ROOT / "checkpoints" / "urbansound8k_audio_cnn_best.pt"
SAMPLE_WAV = _REPO_ROOT / "assets" / "quickstart_sample.wav"


class AudioCNN(torch.nn.Module):
    """Must match the training definition in scripts/train_audio_events.py."""

    def __init__(self, n_mels: int = N_MELS, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(n_mels, 64, 3, padding=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(64, 128, 3, padding=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(128, 256, 3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(256, 256, 3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = torch.nn.Dropout(0.5)
        self.head = torch.nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        h = self.conv(x).squeeze(-1)
        h = self.dropout(h)
        return self.head(h)


def load_audio(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read a wav file, mono, resampled to target_sr, padded/trimmed to 4s."""
    import soundfile as sf

    array, sr = sf.read(str(path), always_2d=False)
    if array.ndim > 1:
        array = array.mean(axis=1)
    if sr != target_sr:
        import librosa

        array = librosa.resample(array, orig_sr=sr, target_sr=target_sr)
    n_samples = int(target_sr * 4.0)
    if len(array) < n_samples:
        array = np.pad(array, (0, n_samples - len(array)))
    else:
        array = array[:n_samples]
    return array.astype(np.float32)


def predict(model: torch.nn.Module, mel: np.ndarray) -> tuple[str, float]:
    """Return (predicted class name, confidence)."""
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 173)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax().item())
    return URBANSOUND_CLASSES[idx], float(probs[idx].item())


def main() -> None:
    if not CHECKPOINT.exists():
        print(f"ERROR: checkpoint not found at {CHECKPOINT}")
        print("Retrain via: uv run python scripts/train_audio_events.py --single-fold 1")
        print("Then move the best checkpoint to:")
        print(f"  {CHECKPOINT}")
        sys.exit(1)

    if not SAMPLE_WAV.exists():
        print(f"ERROR: sample audio not found at {SAMPLE_WAV}")
        print("This file is bundled in the repo (a 3-second siren clip).")
        sys.exit(1)

    print("Loading model...")
    model = AudioCNN()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)

    print(f"Reading sample: {SAMPLE_WAV.name}")
    audio = load_audio(SAMPLE_WAV)

    print("Running inference...")
    mel = audio_to_mel(audio, SAMPLE_RATE)
    label, conf = predict(model, mel)

    print()
    print("=" * 50)
    print(f"  Predicted class: {label}")
    print(f"  Confidence:      {conf:.3f}")
    print("=" * 50)
    print()
    print("This is one of 10 urban sound classes the Week 7.6 model was")
    print("trained on (UrbanSound8K, 10-fold CV test accuracy 0.6779).")
    print("See docs/model_cards/urbansound8k_audio_cnn_v1.md for what it")
    print("can and cannot do.")


if __name__ == "__main__":
    main()
