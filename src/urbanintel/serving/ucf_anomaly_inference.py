"""Inference-time anomaly detection with confidence-based fallback.

For each clip:
  1. R(2+1)D-18 predicts 4-class probabilities
  2. Top-1 confidence determines action:
     - >0.70: accept, no flag
     - 0.40-0.70: accept but flag as "low confidence"
     - <0.40: return "uncertain — manual review required"
  3. Optionally, if uncertain, defer to a secondary model (e.g., audio)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

from urbanintel.data.ucf_crime import CATEGORIES

CONFIDENCE_HIGH = 0.70
CONFIDENCE_LOW = 0.40


@dataclass
class Detection:
    category: str  # "normal" / "violence" / "criminal" / "public_hazard"
    confidence: float
    action: str  # "alert", "flag_for_review", "uncertain"
    requires_review: bool
    raw_probs: dict  # {category_name: prob}


class UCFAnomalyDetector:
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = r2plus1d_18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CATEGORIES))
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, frames_tensor: torch.Tensor) -> Detection:
        """frames_tensor: (T, 3, H, W) or (1, T, 3, H, W)."""
        if frames_tensor.dim() == 4:
            frames_tensor = frames_tensor.unsqueeze(0)
        frames_tensor = frames_tensor.to(self.device)

        logits = self.model(frames_tensor.permute(0, 2, 1, 3, 4))
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        top_idx = int(probs.argmax())
        top_conf = float(probs[top_idx])
        category = CATEGORIES[top_idx]

        raw_probs = {CATEGORIES[i]: float(probs[i]) for i in range(len(CATEGORIES))}

        # Fallback logic
        if top_conf >= CONFIDENCE_HIGH:
            action = "alert" if category != "normal" else "no_alert"
            requires_review = False
        elif top_conf >= CONFIDENCE_LOW:
            action = "flag_for_review"
            requires_review = True
        else:
            # Uncertain — could be any of several categories
            action = "uncertain"
            requires_review = True

        return Detection(
            category=category,
            confidence=top_conf,
            action=action,
            requires_review=requires_review,
            raw_probs=raw_probs,
        )


def format_detection(d: Detection) -> str:
    """Human-readable output for a detection."""
    if d.action == "uncertain":
        top3 = sorted(d.raw_probs.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{c}={p:.2f}" for c, p in top3)
        return (
            f"[UNCERTAIN] No single category dominates. Top 3: {top3_str}. Manual review required."
        )
    elif d.action == "flag_for_review":
        return (
            f"[LOW CONFIDENCE] Predicted: {d.category} ({d.confidence:.2f}). "
            f"Manual review recommended."
        )
    elif d.action == "alert":
        return (
            f"[ALERT] {d.category.upper()} detected ({d.confidence:.2f}). "
            f"Route to appropriate responder."
        )
    else:
        return f"[OK] Normal activity ({d.confidence:.2f})."
