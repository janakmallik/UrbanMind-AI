# Model Card: HMDB-51 15-Class Urban Behavior Classifier (R(2+1)D-18)

## What this model does

Classifies short video clips into one of 15 human action classes,
grouped into 5 urban-relevant categories.

**This model is the pretrained-feature version**, fine-tuned from
Kinetics-400. For the scratch-trained version, see
`docs/model_cards/hmdb51_3dcnn_v1.md`.

## What this model does NOT do

- **Not urban surveillance.** HMDB-51 is a human action dataset from
  movies and web videos.
- **Not a threat detection system.** Semantic groupings are for
  evaluation, not deployment claims.
- **Not production-ready.** 1,500 training clips.
- **Does not handle real urban video** (different camera angles,
  occlusions, weather).

## Architecture

- **Model:** R(2+1)D-18 (~31.3M params)
- **Pretraining:** Kinetics-400 (400 action classes)
- **Structure:** 3D ResNet variant — 2D convolutions for spatial
  features + 1D convolutions for temporal
- **Input:** 16 frames × 112×112 RGB, Kinetics normalization
  (mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
- **Output:** 15-class logits
- **Fine-tuning:** differential lr (backbone 30%, head 3e-4)

## Training

- **Dataset:** HMDB-51 official split 1, 15-class subset
- **Train:** 892 clips | **Val:** 158 | **Test:** 450
- **Loss:** Class-weighted cross-entropy
- **Optimizer:** AdamW (backbone lr=9e-5, head lr=3e-4, weight_decay=1e-4)
- **Scheduler:** Cosine annealing over 20 epochs
- **Batch size:** 2
- **Mixed precision:** Yes (torch.amp)
- **Decoder:** decord
- **Augmentation:** Random crop, random horizontal flip (train only)
- **Checkpoint selection:** Best validation macro-F1 (epoch 20)

## Results (test set, 450 clips)

| Metric | Value |
|---|---:|
| Test accuracy | **0.7311** |
| Test macro-F1 | **0.7293** |
| Tiny3DCNN baseline (scratch) | 0.2533 / 0.2364 |
| **Improvement** | **+47.78 / +49.29 pts** |

### Per-category test accuracy

| Category | R(2+1)D-18 | Tiny3DCNN | Δ |
|---|---:|---:|---:|
| mobility | 81.7% | 36.7% | +45.0 |
| normal | 73.9% | 26.7% | +47.2 |
| threat | 70.8% | 22.5% | +48.3 |
| anomaly | 70.0% | 25.0% | +45.0 |
| restricted | 66.7% | 6.7% | +60.0 |

## What this demonstrates

- **Transfer learning value:** Pretrained Kinetics-400 features nearly
  triple accuracy over scratch training on the same data.
- **Fine-tuning strategy:** Differential learning rates (lower for
  pretrained backbone, higher for new head) work correctly.
- **Normalization matters:** Kinetics normalization (not ImageNet) is
  required for pretrained R(2+1)D-18.
- **Video pipeline engineering:** Frame decoding with decord,
  frame sampling, 3D convolutions, mixed precision training.

## What this is NOT a demonstration of

- Real-world threat detection
- Urban surveillance capability
- State-of-the-art action recognition
- Deployment readiness

## Limitations

1. **Small data** — 1,500 clips across 15 classes
2. **Domain gap** — Kinetics (general human actions) → HMDB-51
3. **Only 100 clips per class** — limits further improvement
4. **Smoke class still weakest** (66.7%) — genuinely hard to
   distinguish from stand/walk
5. **Test set is half of official val**, not official test set
6. **Not evaluated on real urban footage**

## Reproduce

```powershell
uv run python scripts\train_hmdb51_r2plus1d.py --epochs 20 --batch-size 2 --num-workers 6