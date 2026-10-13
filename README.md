# UrbanMind-AI
A multimodal ML/DL system for urban scene understanding: video, image, text, time series, and graph learning fused into one platform. Covers classical ML through transformers, GNNs, generative models, and MLOps.

# Urban Scene Intelligence

A multimodal ML/DL system for urban scene understanding — video, image, text, time series, and graph learning fused into one platform.

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

🚧 **In development** — Week 1 complete

| Week | Focus | Status |
|------|-------|--------|
| 1 | Foundation + classical ML | ✅ Complete |
| 2 | Time series + forecasting | ⬜ Not started |
| 3 | Graph neural networks | ⬜ Not started |
| 4 | Image classification + transfer learning | ⬜ Not started |
| 5 | Segmentation + object detection | ⬜ Not started |
| 6 | Video understanding | ⬜ Not started |
| 7 | Text classification + embeddings | ⬜ Not started |
| 8 | Text generation + LLMs | ⬜ Not started |
| 9 | Multimodal fusion | ⬜ Not started |
| 10 | Uncertainty + anomaly detection | ⬜ Not started |
| 11 | Generative models | ⬜ Not started |
| 12 | MLOps + deployment | ⬜ Not started |

## Week 1 Results — NYC 311 Complaint Classification

Task: predict the **type** of an NYC 311 service request using only
information available **before** the complaint is classified.

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

- Model card: [`docs/model_cards/nyc311_lightgbm_v1.md`](docs/model_cards/nyc311_lightgbm_v1.md)
- Dataset card: [`docs/dataset_cards/nyc311.md`](docs/dataset_cards/nyc311.md)
- Experiments tracked in MLflow (`mlflow.db`)

**Reproduce:**
```bash
uv sync --extra dev
uv run python scripts/compare_models.py
uv run python scripts/tune_best.py
```

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
# Run the test suite
uv run pytest

# Check PyTorch and GPU availability
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
### Launch experiment tracking
```bash
uv run mlflow ui
```

Then open http://localhost:5000 in your browser.

## Project structure

```text
UrbanMind-AI/
├── configs/              # Hydra configuration files
├── data/                 # Datasets (gitignored)
│   ├── raw/
│   ├── processed/
│   └── features/
├── docs/                 # Documentation, model cards, experiment journal
├── notebooks/            # Exploration only (not production code)
├── src/urbanintel/       # Main package
│   ├── data/             # Dataset loaders, validation, splits
│   ├── features/         # Feature engineering per modality
│   ├── models/           # Model zoo
│   ├── training/         # Training loops, losses, callbacks
│   ├── evaluation/       # Metrics, uncertainty, ablation
│   └── serving/          # FastAPI, Streamlit
├── tests/                # Unit, integration, data, model tests
├── pyproject.toml
└── README.md
```

## Datasets

Planned public datasets for experiments across the project:

| **Modality** | **Dataset** | **Use** |
| :--- | :--- | :--- |
| Tabular / text | NYC 311 | Urban complaint classification |
| Time series | METR-LA | Traffic forecasting |
| Graph | OpenStreetMap | Road network construction and graph learning |
| Images | Cityscapes | Semantic segmentation |
| Video | UCF-Crime, BDD100K | Anomaly or incident detection |
| Video + text | MSR-VTT | Cross-modal video-text retrieval |

## Contributing

This is a personal learning and portfolio project. Issues and discussion are welcome, but pull requests are not being accepted at this time.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built as a deep dive into modern ML/DL engineering, with a focus on multimodal systems and reproducible experimentation.