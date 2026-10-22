# How a city system would use UrbanMind-AI

This is a **conceptual integration guide**, not a working API. It
describes the shape of the interface a real city system would consume
if it wanted to use UrbanMind-AI's trained models. Building the actual
service layer is Week 12.

---

## The pattern

Every UrbanMind-AI model follows the same input/output contract:

```
┌──────────────────────────┐
│  City system (consumer)  │
└──────────────────────────┘
            │
            ▼  one audio clip / one image / one 16-frame video clip
┌──────────────────────────┐
│  UrbanMind model         │  ← loads once at startup
│  (per-modality inference)│
└──────────────────────────┘
            │
            ▼  structured prediction
┌──────────────────────────┐
│  City system (decision)  │
└──────────────────────────┘
```

The city system is responsible for:
- Capturing the input (microphone, camera, sensor)
- Deciding *when* to send a clip (e.g., every 4 seconds, or on motion trigger)
- Routing the prediction to a human reviewer or an automated action

UrbanMind-AI is responsible for:
- Returning a labeled prediction with confidence
- Documenting what the model can and cannot do

---

## Example 1 — Audio event detection

```python
# City system code (hypothetical)
from urbanmind_client import AudioEventClassifier  # would be built in Week 12

classifier = AudioEventClassifier.from_checkpoint(
    "checkpoints/urbansound8k_audio_cnn_best.pt"
)

# Every 4 seconds, grab a buffered audio clip and classify it
while True:
    clip = microphone.capture(duration_s=4.0)
    result = classifier.predict(clip)
    # → {"label": "siren", "confidence": 0.89}

    if result["label"] in {"gun_shot", "siren"} and result["confidence"] > 0.75:
        alert_queue.put({
            "time": now(),
            "source": "mic_14th_street",
            "category": result["label"],
            "confidence": result["confidence"],
            "audio_path": clip.save(),
        })
```

**What the model guarantees:**
- Returns a class label from the 10 UrbanSound8K categories
- Returns a confidence score in [0, 1]
- Does not crash on malformed input (returns an error, not a prediction)

**What the model does not guarantee:**
- Zero false positives
- Detection of sounds not in UrbanSound8K (e.g., "scream", "crash")
- Correct behavior on overlapping audio (siren + traffic simultaneously)
- Latency < any threshold (batch inference only)

---

## Example 2 — Video anomaly detection

```python
from urbanmind_client import VideoAnomalyClassifier

classifier = VideoAnomalyClassifier.from_checkpoint(
    "checkpoints/ucf_crime_4class_best.pt"
)

# Every 16 frames, run anomaly classification
for clip in video_stream.iter_clips(frame_count=16, stride=8):
    result = classifier.predict(clip)
    # → {"label": "violence", "confidence": 0.68}

    if result["label"] != "normal" and result["confidence"] > 0.70:
        alert_queue.put(
            {
                "time": now(),
                "source": "cam_broadway_42nd",
                "category": result["label"],
                "confidence": result["confidence"],
                "video_path": clip.save(),
            }
        )
```

**Honest caveat:** the Week 7.5 model has **violence F1 = 0.27 and recall
= 0.17**. A city system using this model would miss ~83% of violent
events. This is documented in the model card as a known limitation,
not fixed in Week 7.5. Any production use would require retraining on
real surveillance footage (not UCF-Crime movies/web videos).

---

## Example 3 — Combined alert (video + audio, rule-based fusion)

This is what Week 9 will formalize as a **trained cross-attention
fusion model**. Until Week 9, the pattern is rule-based:

```python
def compute_alert(video_result, audio_result):
    """Combine two modality predictions into a severity-tiered alert."""
    signals = []
    if video_result["label"] != "normal" and video_result["confidence"] > 0.70:
        signals.append(("video", video_result))
    if (
        audio_result["label"] in {"gun_shot", "siren"}
        and audio_result["confidence"] > 0.75
    ):
        signals.append(("audio", audio_result))

    if not signals:
        return {"severity": 0, "action": "log"}

    # Critical: gun_shot OR both modalities agree
    is_critical = any(s[1]["label"] == "gun_shot" for s in signals) or len(signals) >= 2

    return {
        "severity": 3 if is_critical else 1,
        "action": "escalate" if is_critical else "review",
        "signals": signals,
    }
```

**This is not a trained model.** It's a hand-written rule engine. Week 9
replaces it with a cross-attention fusion network trained on a paired
dataset (video + audio aligned in time). Until then, this is the honest
"how you'd combine two modalities" pattern.

---

## What's missing to make this production-real

| Missing piece | Where it would go |
|---|---|
| Actual service layer (FastAPI, gRPC) | Week 12 |
| Model loading + versioning | Week 12 |
| Latency benchmarks | Week 12 |
| Failure mode handling | Week 12 |
| Monitoring for drift | Week 12 |
| Human review queue | Week 12 |
| Multi-tenant security | Week 12 |
| Real training data (not benchmarks) | Out of scope |

**The point of this document is not that these models are ready to
deploy.** The point is that **the interface they expose is standard**,
and a city system can consume their outputs in a predictable way. The
gap between "predict on a clip" and "operate at city scale" is Week 12
and beyond.