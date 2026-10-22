# Changelog

All notable changes to UrbanMind-AI, one entry per tagged milestone.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [v0.9.5-week7-7-polish] — 2026-10-22

Week 7.7 — Portfolio polish (Session 1).

### Added
- `docs/architecture.md` — one-page system view across all 7 modality pipelines
- `scripts/quickstart.py` — run one trained model end-to-end in <30s
- `assets/quickstart_sample.wav` — 3s siren clip for quickstart demo
- `docs/integration_example.md` — how a city system would consume the models
- `docs/future_work/speech_keywords_design.md` — deferred 7.6b design preserved
- `checkpoints/.gitkeep` — placeholder for gitignored checkpoint directory

### Fixed
- `scripts/train_audio_events.py` — save best-epoch checkpoint after training (Week 7.6 bug: checkpoints were discarded)
- `scripts/quickstart.py` — sample regenerated from a known-class (siren) clip

### Changed
- `.gitignore` — explicitly ignore `checkpoints/*.pt`, keep `.gitkeep`

---

## [v0.9.3-week7-6-audio] — 2026-10-21

Week 7.6 — UrbanSound8K audio event classification.

### Added
- 1D-CNN on 128-mel log-spectrograms, 10-fold CV
- Loader: `src/urbanintel/data/urbansound8k.py`
- Precompute cache: `scripts/precompute_audio.py`
- Training: `scripts/train_audio_events.py`
- Dataset card, model card, experiments entry
- MLflow experiment `urbanintel-week7-6`

### Results
- Test accuracy: 0.6779 ± 0.0497 (10-fold CV)
- Test macro-F1: 0.6878 ± 0.0454
- Val-test gap of ~0.21 documented as known limitation

### Deferred
- Week 7.6b (speech keyword detection) — see `docs/future_work/`

---

## [v0.9.2-week7-5-ucf-crime] — 2026-10-20

Week 7.5 — UCF-Crime 4-class anomaly classification.

### Added
- R(2+1)D-18 (Kinetics-400 pretrained), staged fine-tune
- Loader, clip builder, trainer, inference with confidence fallback
- Model card, experiments entry, MLflow run

### Results
- Official test accuracy: 0.6373
- Official test macro-F1: 0.4972
- Violence class F1: 0.27 (documented as hardest class)

---

## [v0.9.1-week7-r2plus1d] — 2026-10-20

Week 7 — HMDB-51 pretrained comparison.

### Added
- R(2+1)D-18 fine-tune on 15-class urban behavior subset
- Split inspection script

### Results
- Test accuracy: 0.7311 (vs 0.2533 for scratch-trained Tiny3DCNN)
- +47.8 point gain from transfer learning

---

## [v0.9.0-week7-video-pipeline] — 2026-10-20

Week 7 — HMDB-51 baseline (Tiny3DCNN from scratch).

### Added
- 3D CNN on 15-class urban behaviors
- Video loader (`hmdb51.py`, `hmdb51_video.py`)
- Baseline trainer, model card, dataset card, experiments entry

### Results
- Test accuracy: 0.2533
- 3.8× random baseline

---

## [v0.8.0-week6-cv] — 2026-10-19

Week 6 — BDD100K segmentation + detection.

### Added
- SegFormer-B0 semantic segmentation (5 classes)
- YOLOv8n inference + IoU evaluation
- 12 side-by-side visualizations

### Results
- Segmentation mIoU: 0.7172 (test)
- Detection macro F1: 0.381 (domain-shift signature documented)

---

## [v0.7.0-week5-image] — 2026-10-18

Week 5 — BDD100K scene classification.

### Added
- ResNet-50 fine-tune (3 classes: city/highway/residential)
- Frozen baseline + fine-tune comparison

### Results
- Fine-tuned ResNet-50: F1 0.6984
- +2.04 macro-F1 over frozen baseline

---

## [v0.6.0-week4-text] — 2026-10-17

Week 4 — NYC 311 descriptor classification.

### Added
- DistilBERT fine-tune
- TF-IDF + LogReg baseline

### Results
- 100% accuracy — documented as trivial lookup (descriptor is
  near-deterministic predictor of complaint_type)

---

## [v0.5.0-week3-gnn] — 2026-10-16

Week 3 — METR-LA graph WaveNet-lite.

### Added
- Per-timestep graph convolution + causal temporal convolution
- Adaptive adjacency learned alongside fixed distance graph

### Results
- Test MAE 3.44 mph @ h=15 (beats TCN at h=15/h=30, loses at h=60)

---

## [v0.4.0-week2-tcn] — 2026-10-15

Week 2 — METR-LA TCN baseline.

### Added
- 2-block dilated causal TCN

### Results
- Test MAE 3.97/4.17/4.59 @ h=15/30/60

---

## [v0.3.0-week2-ml] — 2026-10-15

Week 2 — METR-LA LightGBM forecasting.

### Added
- LightGBM with lag features
- Naive baseline comparison

### Results
- Test MAE 3.11/3.70/4.34 @ h=15/30/60 (best of all METR-LA models)

---

## [v0.2.0-week1] — 2026-10-13

Week 1 — NYC 311 tabular classification.

### Added
- Classical ML pipeline: LogReg, RF, XGBoost, LightGBM, MLP
- Optuna tuning

### Results
- LightGBM tuned: 68.7% accuracy, 0.6188 macro-F1

---

## [v0.1.0] — 2026-10-12

Initial repo skeleton.

### Added
- `pyproject.toml` with dependencies
- `src/urbanintel/` package layout
- `tests/unit/` structure
- `.gitignore`, `.env.example`, `LICENSE`