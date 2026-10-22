# UrbanMind-AI — System Architecture

One page. What this project builds, how the pieces relate, and where
it is heading.

---

## The shape of the system

UrbanMind-AI is **one repo, seven independent modality pipelines, one
shared contract**. Each modality is a self-contained week of work with
the same deliverables — a loader, a trainer, a model card, a dataset
card, a tag. The pipelines do not depend on each other at runtime.
They share *conventions*, not *code*.

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER (7 modalities)                                  │
│  ┌──────────┬──────────┬──────────┬──────────┐              │
│  │ Tabular  │ Time     │ Graph    │ Text     │              │
│  │ NYC 311  │ METR-LA  │ METR-LA  │ NYC 311  │              │
│  └──────────┴──────────┴──────────┴──────────┘              │
│  ┌──────────┬──────────┬──────────┐                         │
│  │ Image    │ Video    │ Audio    │                         │
│  │ BDD100K  │ HMDB/UCF │UrbanSound│                         │
│  └──────────┴──────────┴──────────┘                         │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  PER-MODALITY PIPELINE (identical contract, 7 times)        │
│                                                             │
│    loader → trainer → evaluator → MLflow run                │
│       │        │          │           │                     │
│       ▼        ▼          ▼           ▼                     │
│    dataset  best-epoch  P/R/F1     urbanintel-weekN         │
│    card     checkpoint  mIoU/MAE   experiment               │
│                             │                               │
│                             ▼                               │
│                       model card + tag                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  FUTURE: MULTIMODAL FUSION (Week 9, planned)                │
│  Cross-attention over aligned video + audio + text + tabular│
│  Requires a paired dataset, not yet collected.              │
└─────────────────────────────────────────────────────────────┘
```

---

## The shipped modality pipelines

Each row below is *one week* of work with the same deliverables.

| Modality | Dataset | Model | Best Result | Tag |
|---|---|---|---|---|
| Tabular | NYC 311 | LightGBM (Optuna) | 68.7% acc | `v0.2.0-week1` |
| Time series | METR-LA | LightGBM + lags | MAE 3.11 mph @ h=15 | `v0.3.0-week2-ml` |
| Graph | METR-LA | Graph WaveNet-lite | MAE 3.44 mph @ h=15 | `v0.5.0-week3-gnn` |
| Text | NYC 311 | DistilBERT | 100% (trivial lookup) | `v0.6.0-week4-text` |
| Image | BDD100K | ResNet-50 fine-tune | F1 0.6984 | `v0.7.0-week5-image` |
| Image | BDD100K | SegFormer-B0 + YOLOv8n | mIoU 0.7172 | `v0.8.0-week6-cv` |
| Video | HMDB-51 | R(2+1)D-18 (Kinetics) | 73.11% acc | `v0.9.1-week7-r2plus1d` |
| Video anomaly | UCF-Crime | R(2+1)D-18 (staged) | F1 0.4972 | `v0.9.2-week7-5-ucf-crime` |
| Audio | UrbanSound8K | 1D-CNN (10-fold) | 0.6779 ± 0.0497 acc | `v0.9.3-week7-6-audio` |

---

## The contract (what every modality ships)

Every week of work ships **the same eight artifacts**, verified by
`git log --tag`:

1. **Loader** — `src/urbanintel/data/{modality}.py`
2. **Preprocessing** (if needed) — `scripts/precompute_*.py`
3. **Trainer** — `scripts/train_*.py`
4. **MLflow run** — `urbanintel-weekN`, with metrics, params, artifact
5. **Model card** — `docs/model_cards/{model}_v1.md`
6. **Dataset card** — `docs/dataset_cards/{dataset}.md`
7. **Experiments entry** — `docs/experiments.md`
8. **Git tag** — `vX.Y.Z-weekN-{slug}`

This is the *coordination* the project demonstrates. Any new modality
added tomorrow would follow the same eight-step pattern. The consistency
is the point.

---

## Cross-modality conventions

| Convention | Where it lives |
|---|---|
| Chronological splits for time data | `src/urbanintel/data/metr_la.py` |
| Stratified splits for classification | `src/urbanintel/data/{bdd100k,hmdb51,ucf_crime,urbansound8k}.py` |
| Best-epoch checkpoint selection | all `scripts/train_*.py` |
| Early stopping (patience 3, min_delta 0.005) | all `scripts/train_*.py` |
| Class-weighted loss for imbalance | all classification trainers |
| Class-weighted CE + Dice for segmentation | `scripts/train_bdd100k_segformer.py` |
| Mixed precision | video + image trainers |
| Per-class breakdowns | all model cards |
| Calibration (ECE) | UCF-Crime card |
| Honest "Known Limitations" section | every model card |

These conventions are **not enforced by a shared library** — they are
enforced by *discipline*. A new modality added in Week 8 would follow
them by reading any one of the seven existing trainers as a template.

---

## What the system is NOT

To be explicit, because the "multi-modality" label can overclaim:

- **Not a single fused model.** Each modality predicts a different target.
  "Multi-modality" describes *breadth of tasks*, not *one fused pipeline*.
- **Not a real-time system.** Every model is trained offline; inference is
  per-clip, batch-style.
- **Not deployed as a service.** There is no FastAPI, no Docker, no
  streaming ingester. Week 12 covers deployment patterns.
- **Not trained on urban deployment data.** Every dataset is a public
  benchmark. Real urban audio, video, and 311 data would differ in ways
  the model cards document as limitations.

The *fusion* stage (Week 9) is the point at which this becomes a single
pipeline rather than seven parallel ones. Until then, "multi-modality"
means "seven independent capabilities, shared conventions."

---

## Where future work goes

| Planned | Week | Status |
|---|---|---|
| LLMs (LoRA + RAG) | 8 | Next |
| Multimodal fusion (cross-attention) | 9 | Planned |
| Uncertainty + calibration (conformal) | 10 | Planned |
| Generative models (VAE / diffusion) | 11 | Planned |
| MLOps + deployment | 12 | Planned |

Deferred to future scope:

- **Speech keyword detection (originally Week 7.6b).** Needs a labeled
  audio corpus; design doc preserved at
  `docs/future_work/speech_keywords_design.md`.
- **Multilingual emergency phrases.** English-only taxonomy for now.
- **Live streaming ingester.** Belongs in Week 12 if at all.

---

## How to read this repo

If you're new to the project:

1. **Start here** — this file.
2. **Skim the README's roadmap table** — one tag per shipped week of work.
3. **Pick one week** — e.g., Week 7.6 (audio). Read its README section
   and model card. Everything you need to understand what was built,
   why, and what it can't do is in those two files.
4. **Run one model** — `uv run python scripts/quickstart.py` (Week 7.7
   deliverable).
5. **Read one trainer end-to-end** — e.g.,
   `scripts/train_audio_events.py`. Every other trainer follows the
   same shape.

---