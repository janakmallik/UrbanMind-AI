# Model Card: HMDB-51 15-Class Urban Behavior Classifier (Tiny3DCNN v1)

## What this model does

Classifies short video clips into one of 15 human action classes,
grouped into 5 urban-relevant categories:

| Category | Classes |
|---|---|
| Normal | stand, walk, sit, eat, drink, hug |
| Mobility | run, ride_bike |
| Anomaly | ride_horse, fall_floor |
| Threat | push, hit, punch, shoot_gun |
| Restricted | smoke |

## What this model does NOT do

- **Not trained on urban surveillance footage.** HMDB-51 is a human
  action dataset compiled from movies and web videos.
- **Not a real threat-detection system.** The "threat" and "restricted"
  categories are semantic groupings, not claims that the model detects
  real-world threats or illegal activity.
- **Not production-ready.** 1,500 training clips is too small for
  deployment.
- **Does not handle occlusions, unusual camera angles, or varying
  lighting the way a real urban video pipeline would.**

## Architecture

- **Model:** Tiny3DCNN (~1.17M params)
- **Structure:** 4 blocks of Conv3d + BatchNorm + ReLU + MaxPool3d,
  followed by Global Average Pooling and a Linear head
- **Input:** 16 frames × 112×112 RGB
- **Output:** 15-class logits

## Training

- **Dataset:** HMDB-51 official split 1, 15-class subset
- **Train:** 892 clips | **Val:** 158 | **Test:** 450
- **Loss:** Class-weighted cross-entropy (weights computed from
  training class frequencies)
- **Optimizer:** AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler:** Cosine annealing over 15 epochs
- **Batch size:** 4
- **Mixed precision:** Yes (torch.amp)
- **Decoder:** decord (OpenCV frame-seek unreliable on HMDB-51)
- **Augmentation:** Random crop, random horizontal flip (train only)
- **Checkpoint selection:** Best validation macro-F1 (epoch 14)

## Results (test set, 450 clips)

| Metric | Value |
|---|---:|
| Test accuracy | 0.2533 |
| Test macro-F1 | 0.2364 |
| Majority-class baseline | 0.0667 |
| Random baseline | 0.0667 |

### Per-category test accuracy

| Category | Accuracy |
|---|---:|
| mobility | 36.7% |
| normal | 26.7% |
| anomaly | 25.0% |
| threat | 22.5% |
| restricted | 6.7% |

## Limitations

1. **Small dataset** — 100 clips per class in official split 1
2. **Domain mismatch** — HMDB-51 is movies/web, not urban surveillance
3. **Small model** — 1.17M params, trained from scratch
4. **Restricted (smoke) class underperforms** — 6.7% suggests the
   model confuses smoke with visually similar actions (stand, walk);
   further analysis needed
5. **No temporal consistency across clips** — model treats each clip
   independently; no long-range tracking
6. **Not evaluated on real urban footage**

## What this model demonstrates

- Video pipeline engineering (frame decoding, 3D convolutions,
  temporal modeling, batch data loading)
- Handling small datasets with class weighting
- Per-category evaluation methodology
- Honest reporting of limitations

## What this model is NOT a demonstration of

- Urban surveillance capability
- Production threat detection
- Action recognition at scale
- Real-time inference

## Reproduce

```powershell
uv run python scripts\train_hmdb51_3dcnn.py --epochs 15 --batch-size 4 --num-workers 6
```

## Citation
Kuehne, H., Jhuang, H., Garrote, E., Poggio, T., & Serre, T. (2011).
HMDB: A Large Video Database for Human Motion Recognition.
IEEE International Conference on Computer Vision (ICCV), 2556-2563.
https://doi.org/10.1109/ICCV.2011.6126543
