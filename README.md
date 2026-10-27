# UrbanMind-AI

A multimodal ML/DL system for urban scene understanding: video, image, text, time series, and graph learning fused into one platform. Covers classical ML through transformers, GNNs, generative models, and MLOps.

## What this is

A **multi-modality** ML/DL platform for urban environments. Over 7 weeks,
it has built end-to-end systems across:

- **Tabular** (Week 1) — NYC 311 complaint classification
- **Time series** (Week 2) — METR-LA traffic forecasting
- **Graph** (Week 3) — road-network spatial dependencies
- **Text** (Week 4) — DistilBERT on NYC 311 descriptors
- **Image** (Weeks 5–6) — BDD100K classification, segmentation, detection
- **Video** (Week 7) — HMDB-51 action recognition, scratch vs. pretrained
- Video anomaly (Week 7.5) — UCF-Crime, staged fine-tuning for imbalanced classes
- Audio (Week 7.6) — UrbanSound8K 10-class event classification, 10-fold CV
- Portfolio polish (Week 7.7) — architecture doc, quickstart, CI, download scripts
- Report generation (Week 8) — Qwen2.5-1.5B generating decision-support field reports from modality evidence
- Portfolio polish (Week 7.7) — architecture doc, quickstart, CI, download scripts
- Report generation (Week 8) — Qwen2.5-1.5B generating decision-support field reports from modality evidence

Each week ships with:
- Reproducible code + tests
- MLflow-tracked experiments
- Dataset and model cards
- Honest results including failure modes

**Week 9 will integrate multiple modalities into a fused model.**
Week 9 is multi-view fusion on BDD100K. Cross-modal fusion (image + text, video + audio) requires paired data not available in this project — see docs/week8_1_policy.md for the honest framing.

---

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/janakmallik/UrbanMind-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/janakmallik/UrbanMind-AI/actions/workflows/ci.yml)


## How to read this repo

This is a 9-week portfolio project. If you're new, here's the fastest path:

1. **Start with the README's roadmap table** (below) — 13 tags across 9 weeks of shipped work.
2. **Pick one week that interests you.** Read its README section + model card + dataset card. Each week is a self-contained unit with:
   - Problem definition
   - Dataset audit (what's real, what's limited)
   - Model comparison table (baselines vs. our approach)
   - Honest failures and limitations (not hidden)
3. **Look at `docs/architecture.md`** for the system view — how the 7 modalities relate.
4. **Read `docs/week8_1_policy.md`** — the safety policy for the report generator. It's the closest thing to a "production considerations" doc in the repo.
5. **Check `docs/experiments.md`** — the running journal. Every experiment, every negative result, documented.
6. **Look at the model cards** in `docs/model_cards/` — 8 cards, one per trained model, each with what it does and doesn't do.

**The honest summary:** 7 modalities trained independently, 1 multi-view fusion experiment (Week 9a), 1 report generator (Week 8), and honest documentation of what worked and what didn't. The README is the entry point; the docs directory is where the real detail lives.


## Project summary

| Modality | Task | Best Result | Week |
|---|---|---|---|
| Tabular | NYC 311 complaint classification | LightGBM 68.7% | 1 |
| Time series | METR-LA traffic forecasting | LightGBM MAE 3.11 mph | 2 |
| Graph | METR-LA spatiotemporal | GWNet MAE 3.44 mph | 3 |
| Text | NYC 311 descriptor classification | DistilBERT 100% (trivial task) | 4 |
| Image | BDD100K scene classification | ResNet-50 F1 0.6984 | 5 |

**Skills demonstrated:** Classical ML, deep learning, time series, graph neural networks, text/NLP transformers, computer vision, PyTorch, MLflow, reproducibility, honest evaluation.

## What this is

A research and engineering project that builds end-to-end machine learning systems for urban environments. It covers:

- **Classical ML** — gradient boosting, ensembles, clustering, anomaly detection
- **Computer vision** — CNNs, vision transformers, segmentation, object detection
- **Video understanding** — 3D CNNs, SlowFast, VideoMAE
- Text & LLMs — BERT fine-tuning, prompt-based report generation (Qwen2.5-1.5B)
- **Time series** — LSTM, TCN, transformers, probabilistic forecasting
- **Graph learning** — GCN, GAT, spatiotemporal GNNs
- **Multimodal fusion** — late, early, cross-attention
- Generative models — VAEs, GANs, diffusion (planned, Week 11)
- Uncertainty — conformal prediction, calibration (planned, Week 10)
- **MLOps** — MLflow, FastAPI, Docker, CI/CD, monitoring

## Project status — Weeks 1–9a complete, Week 10 next

✅ **Completed** — Weeks 1, 2, 3, 4, 5, 6, 7, 7.5, 7.6, 7.7, 8, 9a

🚧 **Next** — Week 10 (uncertainty + calibration)

Some milestones have sub-steps (`7.5`, `7.6`) inserted between numbered
weeks when a follow-on experiment deserved its own tag. Milestones are
not calendar weeks — they are units of shippable work.

| Week | Modality | Best Result | Tag |
|------|----------|-------------|-----|
| 1 | Tabular (NYC 311) | LightGBM 68.7% accuracy | `v0.2.0-week1` |
| 2 | Time series (METR-LA) | LightGBM MAE 3.11 mph | `v0.3.0-week2-ml` |
| 3 | Graph (METR-LA) | Graph WaveNet MAE 3.44 mph | `v0.5.0-week3-gnn` |
| 4 | Text (NYC 311 descriptor) | DistilBERT 100% (trivial lookup) | `v0.6.0-week4-text` |
| 5 | Image classification (BDD100K) | ResNet-50 F1 0.6984 | `v0.7.0-week5-image` |
| 6 | Segmentation + Detection (BDD100K) | SegFormer-B0 mIoU 0.7172 | `v0.8.0-week6-cv` |
| **7** | **Video (HMDB-51)** | **R(2+1)D-18 test acc 73.11%** | **`v0.9.1-week7-r2plus1d`** |
| **7.5** | **Video anomaly (UCF-Crime)** | **R(2+1)D-18 F1 0.4972** | **`v0.9.2-week7-5-ucf-crime`** |
| **7.6** | **Audio (UrbanSound8K)** | **1D-CNN test acc 0.6779, macro-F1 0.6878** | **`v0.9.3-week7-6-audio`** |
| **7.6b** | **Speech keywords (Whisper-tiny)** | **Deferred — see `docs/future_work/`** | — |
| **7.7** | **Portfolio polish** | **Architecture doc, quickstart, CI, download scripts** | **`v0.9.5-week7-7-polish`, `v0.9.6-week7-7-infra`** |
| **8** | **Incident report generator (Qwen)** | **Qwen 0.890 ~ 0.902 vs template 0.626** | **`v0.9.8-week8-qwen`** |
| **9a** | **Multi-view fusion (BDD100K)** | **Fusion 0.6910 vs ResNet 0.6986 (CI includes 0); residential recall 0.41→0.67** | **`v0.9.9-week9-multiview-fusion`** |
| 9b | Cross-modal fusion (attempted) | 311 tabular+text rejected — descriptor is categorical lookup | documented |
| 9c | Cross-modal fusion (Urbansas audio+video) | Audio 0.3686 F1 / video 0.4643 F1 / max fusion 0.4713 F1; logistic +0.08 with leakage caveat; TAU transfer limited to video/max | `v0.9.10-week9c-urbansas` |
| 10 | Uncertainty + calibration (conformal, ECE) | Not started | — |
| 11 | Generative models (VAE / diffusion) | Not started | — |
| 12 | MLOps + deployment (FastAPI, Docker, CI/CD) | Not started | — |

**Total experiments logged:** 40+ trained models across 7 modalities
**Total tests:** 81 passing unit tests
**MLflow experiments:** 11 (`urbanintel-week1` through `urbanintel-week8-1`)
---

## Week 1 — NYC 311 Complaint Classification (Tabular ML)

**Task:** Predict the **type** of an NYC 311 service request using only information available **before** the complaint is classified.

**Dataset:** 100,000 rows from NYC Open Data (top-5 classes, 53,135 rows after filtering)
**Features:** borough, ZIP (top-50), calendar (cyclical), latitude/longitude
**Excluded:** descriptor, agency, location_type — all leaky (assigned during classification)

| Model | Accuracy | Macro-F1 |
|-------|---------:|---------:|
| Majority class | 0.3387 | 0.1012 |
| Logistic Regression | 0.5018 | 0.3074 |
| Random Forest | 0.5701 | 0.4839 |
| Hist Gradient Boosting | 0.6378 | 0.5487 |
| MLP | 0.5175 | 0.3376 |
| XGBoost | 0.6145 | 0.4957 |
| LightGBM (untuned) | 0.6551 | 0.5749 |
| **LightGBM (Optuna-tuned)** | **0.6871** | **0.6188** |

**Deliverables:**
- Model card: [`docs/model_cards/nyc311_lightgbm_v1.md`](docs/model_cards/nyc311_lightgbm_v1.md)
- Dataset card: [`docs/dataset_cards/nyc311.md`](docs/dataset_cards/nyc311.md)
- Experiments journal: [`docs/experiments.md`](docs/experiments.md)
- MLflow experiment: `urbanintel-week1`

---

## Week 2 — METR-LA Traffic Forecasting (Time Series DL)

**Task:** Forecast traffic speed at 207 loop detectors in Los Angeles at horizons 15, 30, and 60 minutes ahead.

**Dataset:** [METR-LA](docs/dataset_cards/metr_la.md) — 34,272 timesteps × 207 sensors, 5-minute sampling, March–June 2012.

**Split:** chronological 70/10/20 (no random shuffling)
**Mask:** zero-valued targets treated as suspected missing (8.1% of cells)

### Model comparison (test MAE, mph)

| Model | h=15 min | h=30 min | h=60 min |
|-------|---------:|---------:|---------:|
| Persistence | 4.02 | 5.10 | 6.80 |
| Sensor-hour mean | 7.50 | 7.50 | 7.49 |
| **LightGBM** (lag features) | **3.11** | **3.70** | **4.34** |
| TCN (2 dilated causal blocks) | 3.97 | 4.17 | 4.59 |

**Key finding:** LightGBM with explicit lag features beat the deep learning model at every horizon. The TCN's relative gap narrows as the horizon grows (27.7% worse at 15 min → 5.9% worse at 60 min), suggesting deep models benefit from capturing longer-range dependencies.

**Deliverables:**
- Dataset card: [`docs/dataset_cards/metr_la.md`](docs/dataset_cards/metr_la.md)
- Experiments journal entry with full analysis
- MLflow experiment: `urbanintel-week2`

---

## Week 3 — METR-LA Graph WaveNet-lite (Graph Neural Networks)

**Task:** Same forecasting task, but using the road-network adjacency matrix to model spatial dependencies between sensors.

**Architecture:** Per-timestep graph convolution + causal temporal convolutions, with adaptive adjacency learned alongside the fixed distance graph. 41,629 parameters, 50 epochs, best-epoch checkpoint on validation MAE.

### Model comparison (test MAE, mph)

| Model | h=15 min | h=30 min | h=60 min |
|-------|---------:|---------:|---------:|
| Persistence | 4.02 | 5.10 | 6.80 |
| TCN | 3.97 | 4.17 | 4.59 |
| **Graph WaveNet-lite** | **3.44** | **4.13** | 5.20 |
| **LightGBM** | **3.11** | **3.70** | **4.34** |

**Key findings:**

1. **GWNet beats TCN at h=15 (+13%) and h=30 (+1%)** but loses at h=60 (-13%).
2. **LightGBM wins at every horizon** — a documented pattern in the literature for small-to-medium structured time-series tasks.
3. **The result is stable across two independent runs** (val MAE 3.836 in both).
4. **The simplified GWNet underperforms the full published architecture** (Wu et al. 2019: MAE 3.53 at h=60 vs our 5.20), likely due to reduced parameter count and simplified graph convolution.

**Reference:** Graph WaveNet (Wu et al. 2019) achieves MAE 2.69 / 3.07 / 3.53 on METR-LA. Our lite version is a learning exercise, not a claim to state-of-the-art.

**Deliverables:**
- Graph loader with optional self-loops: `src/urbanintel/data/graph.py`
- Model: `src/urbanintel/models/graph_wavenet.py`
- Portable tests using synthetic fixtures (run in CI without local dataset)
- MLflow experiment: `urbanintel-week3`
- Tag: `v0.5.0-week3-gnn`

---

## Week 4 — NYC 311 Text Classification (DistilBERT)

**Task:** Classify complaint type from the `descriptor` field using a
fine-tuned DistilBERT model.

**Important note:** The `descriptor` field is assigned by the intake
operator at the same time as `complaint_type`. It is a near-deterministic
predictor of the label. This experiment is a HuggingFace workflow
demonstration, **not** a claim about semantic NLP capability.

### Model comparison (test set)

| Model | Accuracy | Macro-F1 |
|-------|---------:|---------:|
| Majority class | 0.3387 | 0.1012 |
| Descriptor lookup | 0.9997 | 0.9994 |
| TF-IDF + LogReg | 1.0000 | 1.0000 |
| **DistilBERT** | **1.0000** | **1.0000** |

**Key finding:** The task is a deterministic lookup. DistilBERT matches
the ceiling but does not exceed it. Every test descriptor appeared in
training (99.97% row overlap).

**Deliverables:**
- Text data prep module: `src/urbanintel/data/nyc311_text.py`
- DistilBERT training script: `scripts/run_distilbert_nyc311.py`
- Model card: `docs/model_cards/nyc311_distilbert_v1.md`
- MLflow experiment: `urbanintel-week4`
- Tag: `v0.6.0-week4-text`

---

---

## Week 5 — BDD100K Scene Classification (Image)

**Task:** Classify urban driving scene keyframes into one of 3 classes:
city street / highway / residential.

**Dataset:** BDD100K (100,000 images, 3 viable scene classes out of 7).
Subset decision based on label audit — other 4 classes had <400 samples each.

### Model comparison (test set, 4,932 images)

| Model | Accuracy | F1 Macro |
|-------|---------:|---------:|
| Majority class | 0.6196 | 0.2550 |
| Frozen ResNet-50 + LogReg | 0.7618 | 0.6780 |
| **Fine-tuned ResNet-50** | **0.7800** | **0.6984** |

**Key finding:** Fine-tuning improved over frozen ImageNet features by
+1.82 accuracy and +2.04 macro-F1. Training loss decreased through
epoch 5 while validation F1 peaked at epoch 2 — consistent with
overfitting after epoch 2.

**Deliverables:**
- Dataset card: `docs/dataset_cards/bdd100k.md`
- Model card: `docs/model_cards/bdd100k_resnet50_v1.md`
- Loader: `src/urbanintel/data/bdd100k.py`
- Baselines: `scripts/run_bdd100k_baselines.py`
- Fine-tune: `scripts/train_bdd100k_resnet50.py`
- MLflow experiment: `urbanintel-week5`
- Tag: `v0.7.0-week5-image`

---

## Week 6 — BDD100K Segmentation + Detection (Computer Vision Deep Dive)

**Task:** Two CV tasks on BDD100K urban driving data:
1. **Semantic segmentation:** pixel-level classification into 5 classes
   (road, sidewalk, vehicle, person, other)
2. **Object detection:** qualitative demo + quantitative IoU evaluation
   on 4 aligned classes (person, car, truck, bus)

**Dataset:** BDD100K segmentation subset
- Segmentation: 7,000 train / 1,000 val images (official split)
  with pixel-level masks
- Train/val/test split: 6,000 / 1,000 / 1,000 (test held out from
  official train, since official BDD100K test has no public labels)
- Detection: 200 randomly sampled val images for evaluation

### Segmentation results — SegFormer-B0 (5 classes, 20 epochs)

| Split | mIoU |
|-------|-----:|
| Validation (best epoch 19) | 0.7480 |
| **Test (held out)** | **0.7172** |

**Per-class test IoU:**

| Class | IoU | Pixel share |
|-------|----:|------------:|
| road | 0.9081 | 22.2% |
| sidewalk | 0.5377 | 1.6% |
| vehicle | 0.8538 | 9.6% |
| person | 0.3319 | 0.11% |
| other | 0.9547 | 46.5% |

**Training:** 20 epochs, 185s/epoch, ~62 min total on RTX 4050.
Loss: 0.5 × CE(class-weighted) + 0.5 × Dice with `ignore_index=255`.
Class weights: `[1.0, 2.0, 1.5, 20.0, 0.5]` for
`[road, sidewalk, vehicle, person, other]`.

**Key findings:**

1. **Val-test gap of only 0.03** — excellent generalization, no overfitting.
2. **Person reached 0.33 IoU** despite being only 0.11% of pixels —
   the 20× class weight was necessary and effective. The class is
   learnable, not "unlearnable" as pixel frequency alone might suggest.
3. **Sidewalk reached 0.54 IoU** at 1.6% pixel share — the class
   is small but consistent enough for a small model to learn well.
4. **The dominant class ("other" at 46.5%) reached 0.95 IoU** —
   easy win, but it's a catch-all for many sub-classes.

**Reference:** Published SegFormer-B0 on full 19-class BDD100K
segmentation achieves ~0.60–0.65 mIoU. Our simplified 5-class task
achieves 0.72 — the task is easier, but the number is legitimate for
the task definition.

### Detection results — YOLOv8n (COCO pretrained, 200 images)

**Setup:** IoU threshold 0.5, confidence threshold 0.25.
Greedy per-class matching. Each GT box matches at most one prediction.

| Class | TP | FP | FN | Precision | Recall | F1 |
|-------|---:|---:|---:|----------:|-------:|---:|
| person | 96 | 42 | 260 | 0.696 | 0.270 | 0.389 |
| car | 758 | 219 | 1243 | 0.776 | 0.379 | 0.509 |
| truck | 23 | 39 | 70 | 0.371 | 0.247 | 0.297 |
| bus | 10 | 11 | 30 | 0.476 | 0.250 | 0.328 |
| **macro** | | | | **0.580** | **0.286** | **0.381** |

**Key finding:** COCO-pretrained YOLOv8n on BDD100K urban footage
shows the classic **domain-shift signature**:
- **High precision** (~0.58 macro): when YOLO detects something,
  it's usually correct
- **Low recall** (~0.29 macro): YOLO misses most real objects,
  especially small/distant pedestrians and BDD100K-specific
  class definitions
- **Person recall is 0.27** — 73% of pedestrians are missed. This
  is expected for a COCO model on urban footage where pedestrians
  are often small and distant.

Lowering the confidence threshold would raise recall at the cost
of precision — the classic tradeoff that a confidence sweep would
explore. That is out of scope (Level 2 evaluation: single IoU
threshold, single confidence threshold).

### Model deliverables

- Segmentation dataset loader: `src/urbanintel/data/bdd100k_segmentation.py`
- Segmentation training script: `scripts/train_bdd100k_segformer.py`
- Segmentation inspection: `scripts/inspect_bdd100k_segmentation.py`
- Detection inference: `scripts/run_yolov8_detection.py`
- Detection IoU evaluation: `scripts/evaluate_yolov8_iou.py`
- Model card: `docs/model_cards/bdd100k_segformer_b0_v1.md`
- Visualizations: `docs/detection_examples/` (12 side-by-side YOLO vs GT images)
- MLflow experiment: `urbanintel-week6`
- Tag: `v0.8.0-week6-cv`

---

## Week 7 — HMDB-51 15-Class Urban Behavior Recognition (Video)

**Task:** 15-class action recognition on HMDB-51 urban-relevant subset.

**Classes:** stand, walk, sit, eat, drink, hug, run, ride_bike,
ride_horse, fall_floor, push, hit, punch, shoot_gun, smoke

**Categories for evaluation:** normal, mobility, anomaly, threat, restricted.

**Important caveat:** HMDB-51 is a human action dataset compiled from
movies and web videos. It is not urban surveillance footage. Actions
like walk/run/fall_floor are urban-relevant, but the dataset was not
designed for urban scene understanding. This is a technical exercise
in video pipeline engineering, not a claim about urban video capability.

### Model comparison (test set, 450 clips)

| Model | Params | Pretrained | Test Acc | Test F1 |
|-------|-------:|------------|---------:|--------:|
| Majority class | — | — | 0.0667 | 0.0083 |
| Random guess | — | — | 0.0667 | 0.0652 |
| Tiny3DCNN (from scratch) | 1.17M | ❌ | 0.2533 | 0.2364 |
| **R(2+1)D-18 (Kinetics-400)** | **31.3M** | **✅** | **0.7311** | **0.7293** |

**Per-category test accuracy:**

| Category | Classes | Tiny3DCNN | R(2+1)D-18 | Δ |
|----------|---------|----------:|-----------:|---:|
| mobility | run, ride_bike | 36.7% | **81.7%** | +45.0 |
| normal | stand, walk, sit, eat, drink, hug | 26.7% | **73.9%** | +47.2 |
| threat | push, hit, punch, shoot_gun | 22.5% | **70.8%** | +48.3 |
| anomaly | ride_horse, fall_floor | 25.0% | **70.0%** | +45.0 |
| restricted | smoke | 6.7% | **66.7%** | +60.0 |

**Key findings:**

1. **Pretrained features nearly triple accuracy.** R(2+1)D-18 fine-tuned
   from Kinetics-400 achieves 73.11% test accuracy vs. 25.33% for a
   scratch-trained 3D CNN on the same 1,500 clips.
2. **Every category improved by 45+ points.** The biggest gains are in
   `restricted`/smoke (+60.0) and `threat` (+48.3) — categories that
   were essentially unlearnable from scratch.
3. **Transfer learning dominates architecture on small data.** The
   pretrained model has 27× more parameters than Tiny3DCNN, but the
   accuracy gain comes from pretraining, not parameter count.
4. **The pipeline works end-to-end.** decord frame decoding, 16-frame
   sampling, 3D convolutions, Kinetics normalization, differential
   learning rates, mixed precision, MLflow tracking.

**Deliverables:**
- Video dataset loader: `src/urbanintel/data/hmdb51.py`, `hmdb51_video.py`
- Baseline training script: `scripts/train_hmdb51_3dcnn.py`
- Pretrained fine-tune script: `scripts/train_hmdb51_r2plus1d.py`
- Model cards: `docs/model_cards/hmdb51_3dcnn_v1.md`,
  `docs/model_cards/hmdb51_r2plus1d_v1.md`
- MLflow experiment: `urbanintel-week7`
- Tags: `v0.9.0-week7-video-pipeline`, `v0.9.1-week7-r2plus1d`

---

---

## Week 7.5 — UCF-Crime Urban Anomaly Classification (Video, 4-class)

**Task:** Classify urban video clips into 4 anomaly categories:
normal, violence, criminal, public_hazard.

**Dataset:** UCF-Crime, 1,610 source videos → 5,518 training clips
(16-frame windows, 5 clips/video max, stratified train/val/test by video).

**Model:** R(2+1)D-18 (Kinetics-400 pretrained, 31.3M params), staged fine-tune.

**Important caveat:** UCF-Crime is compiled from YouTube/movies, not
urban surveillance footage. Clip-level classification with video-level
labels (not temporal localization). The model flags clips for potential
anomaly — it does not claim real-world threat detection capability.

### Results (official UCF-Crime Test, 1,406 clips)

| Metric | Value |
|--------|------:|
| Official test accuracy | 0.6373 |
| Official test macro-F1 | 0.4972 |
| ECE | 0.1255 |

**Per-category F1:**

| Category | F1 |
|----------|----:|
| normal | 0.83 |
| violence | 0.27 |
| criminal | 0.48 |
| public_hazard | 0.40 |

**Key findings:**

1. **Staged fine-tuning (freeze → unfreeze at 10× lower LR) improved
   macro-F1 from 0.4310 → 0.4972** over naive full fine-tuning — a
   15% relative improvement, driven by fixing catastrophic overfitting.

2. **Violence is the hardest class** (F1 0.27) due to visual ambiguity
   and 12% class share.

3. **Confidence fallback validated:** High-confidence predictions
   (>0.70) achieve 73% accuracy; 67% of test predictions route to
   auto-accept, 33% route to human review.

**Deliverables:**
- Loader: `src/urbanintel/data/ucf_crime.py`
- Clip dataset: `src/urbanintel/data/ucf_crime_video.py`
- Clip builder: `scripts/build_ucf_clips.py`
- Training: `scripts/train_ucf_crime_4class.py`
- Inference with fallback: `src/urbanintel/serving/ucf_anomaly_inference.py`
- Model card: `docs/model_cards/ucf_crime_r2plus1d_v1.md`
- MLflow experiment: `urbanintel-week7-5`
- Tag: `v0.9.2-week7-5-ucf-crime`

---

## Week 7.6 — UrbanSound8K Audio Event Classification (Audio)

**Task:** Classify 4-second urban audio clips into one of 10 sound classes:
air_conditioner, car_horn, children_playing, dog_bark, drilling,
engine_idling, gun_shot, jackhammer, siren, street_music.

Two of these classes (`gun_shot`, `siren`) are emergency-relevant — they
are the audio signals a downstream alert system would act on. This week
adds audio as the 7th modality in the project.

**Dataset:** UrbanSound8K, 8,732 clips, 10 predefined folds.
Source audio is 44.1 kHz stereo, resampled to 22.05 kHz mono and cached
as log-mel spectrograms (128 bins × 173 frames, 2.87 GiB memmap).

**Protocol:** 10-fold cross-validation using the **predefined** folds
(dataset-recommended; NOT a random split). Folds are session-stratified,
not class-stratified — minority-class counts per fold vary by up to 3.5×
(see dataset card for the full per-fold class table).

### Results — 1D-CNN 10-fold CV

| Metric | Mean ± Std |
|--------|-----------:|
| **Test accuracy** | **0.6779 ± 0.0497** |
| **Test macro-F1** | **0.6878 ± 0.0454** |

**Per-fold results:**

| Fold | Test accuracy | Test macro-F1 | Best val F1 |
|---:|---:|---:|---:|
| 1 | 0.6735 | 0.6918 | 0.9115 |
| 2 | 0.6295 | 0.6401 | 0.9163 |
| 3 | 0.5708 | 0.5995 | 0.9089 |
| 4 | 0.6939 | 0.6795 | 0.9082 |
| 5 | 0.6987 | 0.6883 | 0.8366 |
| 6 | 0.6561 | 0.6773 | 0.9025 |
| 7 | 0.6897 | 0.6914 | 0.9090 |
| 8 | 0.6886 | 0.7033 | 0.9206 |
| 9 | 0.7071 | 0.7260 | 0.9066 |
| 10 | 0.7706 | 0.7806 | 0.9087 |

**Baselines:** random guess = 0.10, majority class = ~0.114. The model
achieves ~5.9× the random baseline and ~5.9× the majority baseline.

### Key findings

1. **Best-epoch val F1 systematically overestimates test F1.** Val F1
   averages ~0.90 across folds, but test F1 averages ~0.69. This gap
   (~0.21) is the effect of best-epoch selection overfitting to a
   validation set that shares 9-fold provenance with training. Only
   test F1 is a valid generalization estimate. **Reported as a
   limitation, not hidden.**

2. **Fold-to-fold std is large (±0.05) and driven by the data.**
   Folds are session-stratified, not class-stratified: minority-class
   counts per fold vary by up to 3.5× (`car_horn`: 28 in fold 6 vs
   98 in fold 5). A fold where the held-out set has fewer
   minority-class clips (fold 10: 0.77 acc) scores higher than a
   fold where the test set is harder (fold 3: 0.57 acc). The mean ± std
   is the honest number.

3. **From-scratch 1D-CNN is competitive with expected range.**
   0.678 test accuracy is mid-range for from-scratch 1D-CNNs on
   UrbanSound8K (typical: 0.65–0.72). Pretrained audio models
   (AST, PaSST) reach 0.78–0.85, but that is out of scope for
   Week 7.6.

4. **`gun_shot` (smallest class, 374 clips) is the class that matters
   most and the one most likely to lag.** Per-class F1 analysis
   (confusion matrix) is a natural Week 8 follow-up; class-weighted
   loss compensates during training but per-class F1 will be the
   honest measure.

**Deliverables:**
- Loader: `src/urbanintel/data/urbansound8k.py`
- Precompute cache: `scripts/precompute_audio.py`
- Inspection: `scripts/inspect_urbansound8k.py`, `inspect_urbansound8k_deep.py`
- Training: `scripts/train_audio_events.py`
- Dataset card: `docs/dataset_cards/urbansound8k.md`
- Model card: `docs/model_cards/urbansound8k_audio_cnn_v1.md`
- MLflow experiment: `urbanintel-week7-6`
- Tag: `v0.9.3-week7-6-audio`

---

**Why this project**

Cities are the only domain where video, images, text, time series,
and graphs exist naturally and refer to the same locations and events.

After 7 weeks of building:
- Each modality has a working pipeline with honest results
- The pretrained-vs-scratch comparison (Week 7) shows how
  transfer learning dominates small-data training
- The project is a genuine multi-modality platform, with a fused
  system planned for Week 9

## Architecture

```text
Data Layer
(video · image · text · tabular · time series · graphs)
       │
       ▼
Feature Engineering
(per modality + cross-modal)
       │
       ▼
Model Zoo
(classical · vision · video · text · time series · graph)
       │
       ▼
Multimodal Fusion
(late → early → cross-attention)
       │
       ▼
Uncertainty & Calibration
       │
       ▼
Serving & Monitoring
(FastAPI · Streamlit · MLflow)
```

## Getting started

### Requirements

- Python 3.11 or 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- (Optional) NVIDIA GPU with CUDA 12.4 for deep learning workloads

### Installation

```bash
git clone https://github.com/janakmallik/UrbanMind-AI.git
cd UrbanMind-AI

# Install all dependencies (CUDA 12.4 PyTorch on Windows/Linux with NVIDIA GPU)
uv sync --extra dev
```

### Verify installation

```bash
# Run the test suite (43 tests)
uv run pytest

# Check PyTorch and GPU availability
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Download datasets

```powershell
# METR-LA (traffic forecasting)
.\scripts\download_metr_la.ps1
uv run python scripts\verify_metr_la.py
```

### Launch experiment tracking

```bash
uv run mlflow server --backend-store-uri "sqlite:///C:/Users/Janak Mallik/UrbanMind-AI/mlflow.db" --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000 --workers 1
```

Then open http://127.0.0.1:5000.

### Reproduce experiments

```bash
# Week 1 — NYC 311 classification
uv run python scripts/compare_models.py
uv run python scripts/tune_best.py

# Week 2 — METR-LA forecasting
uv run python scripts/run_forecast_baselines.py
uv run python scripts/run_lightgbm_forecast.py
uv run python scripts/run_tcn_forecast.py

# Week 3 — METR-LA graph forecasting
uv run python scripts/run_graph_wavenet.py
```

## Project structure

```text
UrbanMind-AI/
├── configs/              # Hydra configuration files
├── data/                 # Datasets (gitignored)
│   ├── raw/
│   ├── processed/
│   └── features/
├── docs/
│   ├── dataset_cards/    # One per dataset
│   ├── model_cards/      # One per trained model
│   └── experiments.md    # Running journal of every experiment
├── notebooks/            # Exploration only (not production code)
├── scripts/              # Runnable experiment and download scripts
├── src/urbanintel/       # Main package
│   ├── data/             # Dataset loaders, validation, splits
│   ├── features/         # Feature engineering per modality
│   ├── models/           # Model zoo
│   ├── training/         # Training loops, losses, callbacks
│   ├── evaluation/       # Metrics, uncertainty, ablation
│   └── serving/          # FastAPI, Streamlit
├── tests/                # Unit tests (43 passing)
├── pyproject.toml
└── README.md
```

## Datasets

| Modality | Dataset | Use | Week |
|---|---|---|---|
| Tabular + text | NYC 311 | Complaint classification, DistilBERT | 1, 4 |
| Time series | METR-LA | Traffic forecasting | 2, 3 |
| Graph | METR-LA adjacency | Spatiotemporal GNN | 3 |
| Images | BDD100K | Scene classification, segmentation, detection | 5, 6 |
| Video | HMDB-51 | 15-class action recognition | 7 |
| Video anomaly | UCF-Crime | 4-class anomaly classification | 7.5 |
| Audio | UrbanSound8K | 10-class urban sound events | 7.6 |

## Skills covered

By completing Weeks 1–8.1:

### Classical ML
- XGBoost, LightGBM, Random Forest, logistic regression
- Hyperparameter tuning (Optuna), stacking, ensembling
- Class imbalance handling, calibration, cost-sensitive learning

### Deep learning — general
- PyTorch, custom training loops, mixed precision (`torch.amp`)
- Transfer learning, differential learning rates, fine-tuning
- Best-epoch checkpoint selection, early stopping
- MLflow experiment tracking, model registry patterns

### Time series
- Chronological splits, sliding windows, multi-horizon forecasting
- Masked metrics, forecast skill scores, persistence baselines
- TCN (2 dilated causal blocks)

### Graph neural networks
- Adjacency normalization, message passing, adaptive adjacency
- Graph WaveNet-lite architecture
- Spatiotemporal forecasting (GCN + temporal convolution)

### NLP / Text
- HuggingFace `transformers`, tokenizers, fine-tuning
- DistilBERT classification, TF-IDF baselines
- Text-based lookup vs. learned classification (leakage awareness)
- Prompt-based structured generation (Qwen2.5-1.5B, 4-bit)
- Constraint enforcement (forbidden phrases, required sections)
- Policy-compliant LLM output

### Computer vision — classification
- ResNet-50 fine-tuning, transfer learning from ImageNet
- Frozen feature extraction + linear classifier
- Data augmentation (RandomResizedCrop, horizontal flip, color jitter)

### Computer vision — segmentation + detection
- SegFormer-B0 fine-tuning, 5-class semantic segmentation
- mIoU evaluation, per-class IoU, confusion matrix
- Class-weighted CE + Dice loss with `ignore_index` handling
- YOLOv8 inference on BDD100K (COCO-pretrained, domain shift analysis)
- IoU-based detection matching (precision, recall, F1 at threshold 0.5)

### Video understanding
- Frame decoding with decord (reliable vs. OpenCV seek issues)
- Uniform frame sampling, 3D convolutions
- R(2+1)D-18 fine-tuning from Kinetics-400, Kinetics normalization
- Differential learning rates for pretrained backbones
- Transfer learning vs. scratch training comparison (+47.8 point accuracy gain)

### Multimodal fusion
- Multi-view fusion (in progress, Week 9): 3 vision models → 1 classifier head
- Output-level evidence aggregation (Week 8: 7 modality outputs → LLM report)
- Feature extraction on frozen encoders (frozen ResNet, SegFormer, YOLO)
- Split verification (reproduce published metrics before reuse)
- Contamination filtering (SegFormer train ∩ ResNet test = 187 images excluded)

### Experiment management
- MLflow tracking, run comparison, artifact logging
- Reproducibility via dataset hashes, git SHAs, seeds
- Separate experiments for baseline vs. improved approaches
- Honest reporting of limitations and negative results
- Negative result documentation (Week 9.5: 311 tabular+text rejected)

### Testing
- Unit tests with synthetic fixtures (no real dataset needed)
- Causality checks, leakage prevention
- Augmentation determinism tests
- Data contract validation (Pydantic, Pandera)

### Documentation
- Dataset cards (NYC 311, METR-LA, BDD100K, HMDB-51, UrbanSound8K, UCF-Crime, synthetic scenarios)
- Model cards (LightGBM, DistilBERT, ResNet-50, SegFormer, Tiny3DCNN, R(2+1)D-18, audio CNN, Qwen report generator)
- Experiments journal (`docs/experiments.md`)
- Policy documents (`docs/week8_1_policy.md`)
- Architecture doc (`docs/architecture.md`)
- Citation tracking (`CITATIONS.md`)

### Engineering practices
- Reproducible pipelines (uv, locked deps, PyTorch version pinned)
- Git discipline: feature commits, tags per milestone, clean history
- Cross-platform scripts (Windows + `uv`, no OS-specific assumptions)
- No data/artifacts in git (`.gitignore` for `.pt`, `mlflow.db`, `mlruns/`)
- CI via GitHub Actions (pytest + ruff on every push)
- Download scripts for all datasets
- Quickstart script (run one model in <30s)

### Planned (not yet built)
- Cross-modal fusion on paired data (Week 10+, requires new dataset)
- Uncertainty + conformal prediction (Week 10)
- Generative models (VAE, GAN, diffusion — Week 11)
- FastAPI serving + Docker (Week 12)

## Contributing

This is a personal learning and portfolio project. Issues and discussion are welcome, but pull requests are not being accepted at this time.

## References

All datasets, models, and libraries used in this project are documented
in [CITATIONS.md](CITATIONS.md).

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Dataset licenses vary by source; see CITATIONS.md for per-dataset terms.

## Acknowledgments

Built as a deep dive into modern ML/DL engineering, with a focus on multimodal systems and reproducible experimentation. Uses publicly available datasets (NYC Open Data, METR-LA from the DCRNN authors).
