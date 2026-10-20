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

Each week ships with:
- Reproducible code + tests
- MLflow-tracked experiments
- Dataset and model cards
- Honest results including failure modes

**Week 9 will integrate multiple modalities into a fused model.**
Until then, "multi-modality" describes the breadth of tasks, not a
single fused pipeline.

---

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
- **Text & LLMs** — BERT fine-tuning, seq2seq generation, LoRA, RAG
- **Time series** — LSTM, TCN, transformers, probabilistic forecasting
- **Graph learning** — GCN, GAT, spatiotemporal GNNs
- **Multimodal fusion** — late, early, cross-attention
- **Generative models** — VAEs, GANs, diffusion
- **Uncertainty** — conformal prediction, calibration
- **MLOps** — MLflow, FastAPI, Docker, CI/CD, monitoring

## Project status — 7 of 12 weeks complete

🚧 **In development** — Week 8

| Week | Modality | Best Result | Tag |
|------|----------|-------------|-----|
| 1 | Tabular (NYC 311) | LightGBM 68.7% accuracy | `v0.2.0-week1` |
| 2 | Time series (METR-LA) | LightGBM MAE 3.11 mph | `v0.3.0-week2-ml` |
| 3 | Graph (METR-LA) | Graph WaveNet MAE 3.44 mph | `v0.5.0-week3-gnn` |
| 4 | Text (NYC 311 descriptor) | DistilBERT 100% (trivial lookup) | `v0.6.0-week4-text` |
| 5 | Image classification (BDD100K) | ResNet-50 F1 0.6984 | `v0.7.0-week5-image` |
| 6 | Segmentation + Detection (BDD100K) | SegFormer-B0 mIoU 0.7172 | `v0.8.0-week6-cv` |
| **7** | **Video (HMDB-51)** | **R(2+1)D-18 test acc 73.11%** | **`v0.9.1-week7-r2plus1d`** |
| 8 | Text generation / LLMs | Not started | — |
| 9 | Multimodal fusion | Not started | — |
| 10 | Uncertainty + anomaly detection | Not started | — |
| 11 | Generative models | Not started | — |
| 12 | MLOps + deployment | Not started | — |

**Total experiments logged:** 30+ trained models across 7 modalities
**Total tests:** 50+ passing unit tests
**MLflow experiments:** 7 (`urbanintel-week1` through `urbanintel-week7`)

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

## Datasets

| Modality | Dataset | Use | Week |
|----------|---------|-----|------|
| Tabular + text | NYC 311 | Complaint classification, DistilBERT | 1, 4 |
| Time series | METR-LA | Traffic forecasting | 2, 3 |
| Graph | METR-LA adjacency | Spatiotemporal GNN | 3 |
| Images | BDD100K | Scene classification, segmentation, detection | 5, 6 |
| Video | HMDB-51 | 15-class action recognition | 7 |

## Skills covered

By completing Weeks 1–7:

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
- LSTM, TCN, temporal Transformers

### Graph neural networks
- Adjacency normalization, message passing, adaptive adjacency
- Graph WaveNet-lite architecture
- Spatiotemporal forecasting

### NLP / Text
- HuggingFace `transformers`, tokenizers, fine-tuning
- DistilBERT classification, TF-IDF baselines
- Text-based lookup vs. learned classification (leakage awareness)

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

### Experiment management
- MLflow tracking, run comparison, artifact logging
- Reproducibility via dataset hashes, git SHAs, seeds
- Separate experiments for baseline vs. improved approaches
- Honest reporting of limitations and negative results

### Testing
- Unit tests with synthetic fixtures (no real dataset needed)
- Causality checks, leakage prevention
- Augmentation determinism tests
- Data contract validation (Pydantic, Pandera)

### Documentation
- Dataset cards (NYC 311, METR-LA, BDD100K, HMDB-51)
- Model cards (LightGBM, DistilBERT, ResNet-50, SegFormer, Tiny3DCNN, R(2+1)D-18)
- Experiments journal (`docs/experiments.md`)
- Consistent per-week results tables in README
- Citation tracking (`CITATIONS.md`)

### Engineering practices
- Reproducible pipelines (uv, locked deps, PyTorch version pinned)
- Git discipline: feature commits, tags per milestone, clean history
- Cross-platform scripts (Windows + `uv`, no OS-specific assumptions)
- No data/artifacts in git (`.gitignore` for `.pt`, `mlflow.db`, `mlruns/`)

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
