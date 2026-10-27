# UrbanMind-AI — System Architecture

One page. What this project builds, how the pieces relate, and where
it is heading.

---

## The shape of the system

UrbanMind-AI is **one repo, seven independent modality pipelines, one
report generation layer, one multi-view fusion experiment**. Each
modality is a self-contained week of work with the same deliverables —
a loader, a trainer, a model card, a dataset card, a tag. The pipelines
do not depend on each other at runtime; they share **conventions**, not
code.

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
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌──────────────────┐ ┌─────────────┐ ┌─────────────────┐
│ WEEK 8           │ │ WEEK 9a     │ │ FUTURE (Week 11)│
│ Report generation│ │ Multi-view  │ │ Cross-modal     │
│ Qwen consumes    │ │ fusion on   │ │ fusion on paired│
│ 7 modality       │ │ BDD100K     │ │ data (MAVAD)    │
│ outputs → field  │ │ (3 vision   │ │ audio + video   │
│ report           │ │ models → 1  │ │                 │
│                  │ │ classifier) │ │                 │
│ ✅ Shipped       │ │ ✅ Shipped  │ │ ⏳ Planned      │
└──────────────────┘ └─────────────┘ └─────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  FUTURE (Weeks 10–12)                                       │
│  Uncertainty + calibration → Generative models → MLOps      │
└─────────────────────────────────────────────────────────────┘
```

**Two distinct fusion approaches, both shipped:**

- **Output-level fusion** (Week 8): 7 modality outputs → Qwen report generator → field report
- **Feature-level multi-view fusion** (Week 9a): 3 vision models → 1 fused scene classifier

**Not yet shipped:** cross-modal fusion (requires paired audio+video data — MAVAD is the candidate).

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

## The shipped integration layers

| Layer | Approach | Result | Tag |
|---|---|---|---|
| Report generation | Qwen2.5-1.5B consumes 7 modality outputs → field report | Qwen 0.890–0.902 vs template 0.626 | `v0.9.8-week8-qwen` |
| Multi-view fusion | ResNet + SegFormer + YOLO features → LightGBM scene classifier | 0.6910 F1 vs ResNet 0.6986 (CI includes 0); residential recall 0.41→0.67 | `v0.9.9-week9-multiview-fusion` |

**The report generator** is output-level fusion (evidence aggregation + LLM narrative). **The multi-view fusion** is feature-level fusion of same-modality features. Neither is cross-modal.

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
| Bootstrap confidence intervals | Week 9a fusion evaluation |
| Contamination filtering before reuse | Week 9a fusion (SegFormer train ∩ ResNet test) |
| Split reconstruction verification | Week 9a (reproduced 0.7800/0.6984 before proceeding) |

These conventions are **not enforced by a shared library** — they are
enforced by *discipline*. A new modality added tomorrow would follow
them by reading any one of the seven existing trainers as a template.

---

## What the system is NOT

To be explicit, because the "multi-modality" label can overclaim:

- **Not a single end-to-end fused model.** Each modality predicts a
  different target. Week 8 fuses at the output layer (report
  generation); Week 9a fuses at the feature layer (multi-view). Neither
  is cross-modal fusion of audio + video + text simultaneously.
- **Not comprehensive cross-modal fusion.** Week 9a (multi-view on BDD100K) is same-modality. Week 9b's attempts on 311 tabular+text and MAVAD were rejected (descriptor is categorical lookup; MAVAD's public release is video-only). Week 9c is the first genuine cross-modal experiment: late fusion of audio and video on the paired Urbansas dataset. The result is a documented marginal in-domain gain (best pooled F1 0.4713 with max fusion, vs 0.4643 video-only) with unresolved leakage caveats for the learned fusion heads, and no cross-city transfer beyond what video-only already achieves. Higher-capability fusion methods (feature-level, cross-attention) remain future work.
- **Not a real-time system.** Every model is trained offline; inference
  is per-clip, batch-style.
- **Not deployed as a service.** There is no FastAPI, no Docker, no
  streaming ingester. Week 12 covers deployment patterns.
- **Not trained on urban deployment data.** Every dataset is a public
  benchmark. Real urban audio, video, and 311 data would differ in ways
  the model cards document as limitations.
- **Not a claim that fusion improves accuracy.** Week 9a multi-view
  fusion does not beat the original ResNet on overall macro-F1 (0.6910
  vs 0.6986, bootstrap CI includes zero). The residential-recall
  improvement (0.41 → 0.67) is a class trade-off, not a general win.
- **Not a claim that cross-modal fusion is achieved.** The 311 tabular +
  text fusion was attempted (Week 9b) and rejected — the "text"
  descriptor is a categorical lookup, not linguistic signal.

---

## Where future work goes

| Planned | Week | Status |
|---|---|---|
| **Uncertainty + calibration** (conformal, ECE) | 10 | **Next** |
| Feature-level / cross-attention fusion on Urbansas | 9c-ext | Planned — builds on Week 9c late fusion |
| **Generative models** (VAE / diffusion) | 12 | Planned |
| **MLOps + deployment** (FastAPI, Docker, CI/CD) | 13 | Planned |

Deferred to future scope:

- **Speech keyword detection (originally Week 7.6b).** Needs a labeled
  audio corpus; design doc preserved at
  `docs/future_work/speech_keywords_design.md`.
- **NYC 311 tabular + text fusion (Week 9b).** Attempted; rejected.
  Descriptor is a categorical lookup — TF-IDF and one-hot encoding both
  score 0.802–0.804 F1. Not linguistic signal. See `docs/experiments.md`
  for the audit.
- **Multilingual emergency phrases.** English-only taxonomy for now.
- **Live streaming ingester.** Belongs in Week 13 if at all.

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
6. **For the fusion work** — read `docs/model_cards/bdd100k_multiview_fusion_v1.md`
   for the Week 9a negative result, and `docs/week8_1_policy.md` for
   the report generator's safety policy.
