# Model Card: UCF-Crime 4-Class Anomaly Classifier (R(2+1)D-18 v1)

## What this model does

Classifies a 16-frame urban video clip into 4 categories:

| Category | Includes |
|---|---|
| **Normal** | Everyday activity (no anomaly) |
| **Violence** | Abuse, Assault, Fighting, Shooting, Explosion |
| **Criminal** | Arrest, Burglary, Robbery, Shoplifting, Stealing |
| **Public Hazard** | Arson, RoadAccidents, Vandalism |

Intended for a safety-critical system that flags anomalous urban
footage for human review.

## What this model does NOT do

- **Not trained on real urban surveillance.** UCF-Crime is compiled
  from YouTube movies and web videos.
- **Not a temporal localization system.** Predicts a category per
  clip; does not identify when within a clip the anomaly starts.
- **Not production-ready.** 5,500 training clips is small for a
  31M-parameter model.
- **Does not handle live streaming directly.** Uses pre-extracted
  clips at 16 frames × 112×112.
- **Not a threat detection system.** Semantic groupings are for
  evaluation; not claims of real-world threat detection capability.

## Architecture

- **Model:** R(2+1)D-18 (31.3M params)
- **Pretraining:** Kinetics-400 (400 action classes)
- **Structure:** 3D ResNet variant — 2D convs for spatial features
  + 1D convs for temporal
- **Input:** 16 frames × 112×112 RGB, Kinetics normalization
- **Output:** 4-class logits
- **Head:** linear layer (512 → 4)

## Training

- **Dataset:** UCF-Crime, 1,610 source videos, 5,518 clips
- **Train:** 5,518 clips / **Val:** 1,180 / **Test:** 1,183
  / **Official test:** 1,406
- **Split:** stratified by (class, video_id) — no frame-level leakage
- **Loss:** class-weighted cross-entropy
- **Optimizer:** AdamW with differential LR
  - Backbone: 1e-5 (after unfreeze)
  - Head: 1e-4
- **Schedule:** staged fine-tuning
  - Epochs 1–2: head-only (backbone frozen)
  - Epoch 3+: unfreeze backbone
- **Batch size:** 12
- **Mixed precision:** Yes
- **Early stopping:** patience=3, min_delta=0.005
- **Best epoch:** 5/8 (early stopped at 8)

## Results

| Metric | Value |
|---|---:|
| Test accuracy | 0.6492 |
| Test macro-F1 | 0.5190 |
| **Official test accuracy** | **0.6373** |
| **Official test macro-F1** | **0.4972** |
| ECE | 0.1255 |

### Per-category F1 on official test

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| normal | 0.78 | 0.89 | 0.83 |
| violence | 0.65 | 0.17 | 0.27 |
| criminal | 0.43 | 0.56 | 0.48 |
| public_hazard | 0.38 | 0.43 | 0.40 |

### Confidence fallback behavior

The inference module uses confidence-based routing:

| Confidence | Action | Observed accuracy |
|---|---|---:|
| > 0.70 | Auto-accept, send alert if anomaly | 73.31% |
| 0.40–0.70 | Flag for human review | 44.11% |
| < 0.40 | Return "uncertain", require manual review | 44.83% |

67% of test predictions route to auto-accept; 33% route to human review.

## Limitations

1. **Small dataset** — 5,518 training clips for 31M-param model
2. **Video-level labels** — all clips from a video get the video's
   label, even during non-anomalous segments
3. **Violence class hardest** — F1 0.27, low recall (0.17)
4. **Confusion between criminal and hazard** — precision drops below
   0.50 for those categories
5. **Domain mismatch** — UCF-Crime is movies/web, not surveillance
6. **No temporal localization** — cannot say *when* anomaly occurs
7. **Not evaluated on live streams** — only pre-extracted clips

## What this model demonstrates

- Video pipeline engineering (frame sampling, 3D CNN fine-tuning)
- Staged fine-tuning for pretrained video models on small data
- Confidence-based fallback for safety-critical deployment
- Class-weighted training for imbalanced data
- Calibration measurement (ECE)
- MLflow experiment tracking with per-category metrics

## Reproduce

```powershell
uv run python scripts\build_ucf_clips.py
uv run python scripts\train_ucf_crime_4class.py --epochs 15 --batch-size 12 --num-workers 6
```

## Citations

- Sultani, W., Chen, C., & Shah, M. (2018). *Real-world Anomaly Detection in Surveillance Videos.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
- Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018). *A Closer Look at Spatiotemporal Convolutions for Action Recognition.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
- Kay, W., et al. (2017). *The Kinetics Human Action Video Dataset.* arXiv preprint arXiv:1705.06950.