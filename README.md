# UrbanMind-AI

A multimodal ML/DL system for urban scene understanding: video, image, text, time series, and graph learning fused into one platform. Covers classical ML through transformers, GNNs, generative models, and MLOps.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

## Status

🚧 **In development** — Week 3 complete

| Week | Focus | Status |
|------|-------|--------|
| 1 | Foundation + classical ML | ✅ Complete |
| 2 | Time series forecasting | ✅ Complete |
| 3 | Graph neural networks | ✅ Complete |
| 4 | Text classification (BERT) | ⬜ Next |
| 5 | Image classification + transfer learning | ⬜ Not started |
| 6 | Segmentation + object detection | ⬜ Not started |
| 7 | Video understanding | ⬜ Not started |
| 8 | Text generation + LLMs | ⬜ Not started |
| 9 | Multimodal fusion | ⬜ Not started |
| 10 | Uncertainty + anomaly detection | ⬜ Not started |
| 11 | Generative models | ⬜ Not started |
| 12 | MLOps + deployment | ⬜ Not started |

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

## Why this project

Cities are the only domain where video, images, text, time series, and graphs exist naturally and refer to the same locations and events. That makes urban scene understanding a genuine multimodal problem — not synthetic pairing, not decoration.

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

| Modality | Dataset | Use | Status |
|----------|---------|-----|--------|
| Tabular / text | NYC 311 | Urban complaint classification | ✅ Used (Week 1) |
| Time series | METR-LA | Traffic forecasting | ✅ Used (Weeks 2–3) |
| Graph | METR-LA adjacency | Sensor spatial structure | ✅ Used (Week 3) |
| Images | Cityscapes | Semantic segmentation | ⬜ Planned |
| Video | UCF-Crime, BDD100K | Anomaly / incident detection | ⬜ Planned |
| Video + text | MSR-VTT | Cross-modal retrieval | ⬜ Planned |

## Skills covered

By completing Weeks 1–3:

- **Classical ML:** XGBoost, LightGBM, Random Forest, logistic regression, hyperparameter tuning with Optuna
- **Deep learning:** PyTorch, custom training loops, causal convolutions, adaptive graph convolution
- **Time series:** Chronological splits, sliding windows, multi-horizon forecasting, masked metrics, skill scores
- **Graph learning:** Adjacency normalization, message passing, adaptive adjacency
- **Experiment management:** MLflow tracking, reproducibility via dataset hashes and git SHAs
- **Testing:** Unit tests with synthetic fixtures, causality checks, leakage prevention
- **Documentation:** Dataset cards, model cards, experiment journal

## Contributing

This is a personal learning and portfolio project. Issues and discussion are welcome, but pull requests are not being accepted at this time.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built as a deep dive into modern ML/DL engineering, with a focus on multimodal systems and reproducible experimentation. Uses publicly available datasets (NYC Open Data, METR-LA from the DCRNN authors).
