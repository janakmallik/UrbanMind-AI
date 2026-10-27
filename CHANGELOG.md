# Changelog

All notable changes to UrbanMind-AI, one entry per tagged milestone.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [v0.9.10-week9c-urbansas] — 2026-10-27

Week 9c — Urbansas cross-modal audio-video fusion.

### Added

- **Audio baseline** (`08f5e24`): 1D-CNN on 128-mel log-spectrograms,
  5-fold leave-one-location-out on MAVD. Pooled OOF ROC-AUC 0.6237, PR-AUC
  0.3296, F1 0.3686 at threshold 0.5. Per-location ROC-AUC is bimodal
  (0.48–0.70) for unidentified reasons. TAU external collapses to F1 0.0000.
- **Video baseline** (`94833c6`): R(2+1)D-18 Kinetics-400 pretrained,
  same folds. Pooled OOF F1 0.4643, ROC-AUC 0.6130, TAU external F1 0.2362.
  Disagreement analysis reveals 55% of high-confidence false positives are
  on `only_visible` clips — the model learns "visible vehicle → predict
  offscreen" as a scene-context shortcut, not direct detection.
- **Late fusion** (`94833c6`): six models tested — audio, video, mean, max,
  logistic, logistic+interaction. Best pooled MAVD F1 is max fusion 0.4713
  (+0.007 over video-only, marginal). Weighted logistic fusion improves
  MAVD F1 over audio by +0.08 (clip-level CI excludes zero, location-level
  CI includes zero). TAU transfer is preserved only by max fusion (0.2308)
  and video-only (0.2362); logistic collapses to 0.0230.
- **Documentation**:
  - `docs/model_cards/urbansas_audio_cnn_v1.md`
  - `docs/model_cards/urbansas_video_cnn_v1.md`
  - `docs/model_cards/urbansas_fusion_v1.md`
  - `docs/dataset_cards/urbansas.md`
  - Three `docs/experiments.md` entries (9c-audio, 9c-video, 9c-fusion)
- **Scripts**:
  - `scripts/train_urbansas_audio.py`
  - `scripts/train_urbansas_video.py`
  - `scripts/train_urbansas_fusion.py`
  - `scripts/precompute_urbansas_mels.py`
  - `scripts/precompute_urbansas_video_frames.py`
  - `scripts/diagnose_urbansas_audio.py`
  - `scripts/audit_urbansas.py`
  - `scripts/check_urbansas_labels.py`
  - `scripts/inspect_urbansas.py`
- **Data pipeline**: `src/urbanintel/data/urbansas.py` with loaders for
  audio-only, video-only, cached variants, and augmentation.

### Documented

- **Negative result**: late fusion does not transfer to unseen cities. Best
  TAU F1 across all fusion models is max fusion 0.2308, roughly matching
  video-only 0.2362. No fusion model exceeds video-only on external test.
- **Complementary-signal hypothesis**: audio and video are at chance on
  *different* MAVD locations and *different* subclasses. Consistent with
  complementary information; not causal.
- **Leakage concern**: learned fusion heads' training probabilities come
  from base models that saw the held-out location in other folds. A fully
  independent evaluation requires nested cross-validation (~20 hours
  compute); not done. Zero-parameter mean and max fusion are unaffected.
- **Video cannot detect offscreen vehicles.** It cannot see them. The
  ROC-AUC above chance reflects scene-context correlation, not detection.

### Not committed

- Model checkpoints (`*.pt` — gitignored)
- Prediction CSVs (`data/urbansas_results/` — gitignored)
- Frame cache (`data/urbansas_video_frames/` — gitignored)
- Mel cache (`data/urbansas_mels/` — gitignored)

### Deferred to Week 9c-extended

- Pretrained audio baseline (AST/PaSST/BEATs)
- Feature-level fusion (frozen embeddings + MLP head)
- Cross-attention fusion
- 4-class subclass task (requires label-semantics audit first)
- Nested cross-validation to resolve the logistic fusion leakage

---

## [v0.9.9-week9-multiview-fusion] — 2026-10-25

Week 9a — Multi-view fusion on BDD100K (negative result).

### Added
- `scripts/reconstruct_resnet_split.py` — reproduce ResNet's BDD100K split (verified 0.7800 acc / 0.6984 F1)
- `scripts/verify_resnet_split.py` — verify split reconstruction against published metrics
- `scripts/prepare_fusion_data.py` — contamination filtering (SegFormer train ∩ ResNet val/test)
- `scripts/extract_fusion_features.py` — 14-feature extraction (ResNet 3 + SegFormer 5 + YOLO 6)
- `scripts/train_bdd100k_fusion.py` — fusion head training + 8-model comparison + bootstrap CI
- `scripts/audit_nyc311.py` — 311 tabular+text fusion gate (rejected — descriptor is categorical lookup)
- `src/urbanintel/reporting/header.py` — shared evidence header for template + Qwen
- `data/splits/` — split manifests for reproducibility (clean/excluded ID lists)
- `data/features/bdd100k_fusion/` — 14-feature parquet files (train/test)
- `data/fusion_results/session4_results.json` — raw metrics from 8-model comparison
- `docs/model_cards/bdd100k_multiview_fusion_v1.md` — model card with all results + limitations

### Results
- **Full fusion (14 features):** 0.7372 test acc / 0.6910 test macro-F1
- **Original ResNet checkpoint:** 0.7819 test acc / 0.6986 test macro-F1
- **Bootstrap CI on macro-F1 difference:** [−0.0229, +0.0091], P(fusion > ResNet) = 0.191
- **Residential recall:** 0.41 (ResNet) → 0.67 (fusion), +0.26 trade-off
- Ablation: ResNet features dominate (−0.158 without), SegFormer marginal (−0.011), YOLO redundant (−0.001)

### Documented
- **Negative result:** Multi-view fusion does not improve overall accuracy over the original ResNet
- **Class trade-off:** Fusion improves residential recall at the cost of overall accuracy
- **Same-modality caveat:** All three models process images. Not cross-modal.
- **311 fusion rejected:** Descriptor is a categorical lookup, not linguistic signal (TF-IDF ≈ one-hot)

---

## [v0.9.8-week8-qwen] — 2026-10-22

Week 8 — Incident report generation (Qwen2.5-1.5B).

### Added
- `src/urbanintel/reporting/schema.py` — Pydantic evidence schema for 6 modality sources
- `src/urbanintel/reporting/template_baseline.py` — deterministic template report generator
- `src/urbanintel/reporting/qwen_generator.py` — Qwen2.5-1.5B 4-bit NF4 report generator
- `src/urbanintel/reporting/scenarios.py` — 31 synthetic incident scenarios
- `scripts/report_baseline.py`, `scripts/generate_qwen_report.py`, `scripts/evaluate_reports.py`
- `scripts/feasibility_qwen.py` — VRAM check (1.15 GB model, 2.19 GB peak, 4.25 GB headroom)
- `tests/unit/test_reporting.py` — 11 unit tests
- `tests/unit/test_week8_1_regression.py` — 10 regression tests
- `docs/week8_plan.md`, `docs/week8_1_policy.md`
- `docs/dataset_cards/week8_synthetic_scenarios.md`
- `docs/model_cards/week8_report_qwen_v1.md`

### Results
- **Template baseline:** 0.626 overall on corrected metrics
- **Qwen zero-shot:** 0.890–0.902 overall (two runs, temperature=0.3)
- **Gap:** +0.264 to +0.276

### Fixed
- **Week 8 evaluator was tautological:** `no_hallucination` was guaranteed 1.0 by construction; `severity_consistency` used the same rule to generate and score. Corrected in Week 8.1.

### Documented
- **Policy change:** Escalate = "urgent human review", NOT "dispatch officers". Decision-support, not enforcement.
- **Over-escalation limitation:** Qwen-1.5B recommends dispatch on 6 of 23 review-tier scenarios. Documented, not fixed.

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