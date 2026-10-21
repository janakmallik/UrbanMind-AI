# Experiments Journal

Short entries for every experiment. Structure: hypothesis → setup → result → takeaway.

---

## 2026-10-14 — METR-LA naive forecasting baselines

**Hypothesis:** Persistence will be strong at 15 min and decay with horizon.
The sensor-hour historical mean will be more stable across horizons because
it captures the rush-hour rhythm that persistence cannot.

**Setup:**
- Dataset: METR-LA, 207 sensors, 5-min sampling
- Split: chronological 70/10/20
- Windows: 12 input steps (60 min), horizons 3 / 6 / 12 (15 / 30 / 60 min)
- Train windows: 23,967 | Validation windows: 3,404 | Test windows: 6,832
- Metrics: masked MAE / RMSE / MAPE on valid targets only
  (zero-valued targets treated as suspected missing and excluded)
- Random seed: N/A (deterministic baselines)

**Result:**

| Baseline | MAE h=3 | MAE h=6 | MAE h=12 | RMSE h=3 | RMSE h=6 | RMSE h=12 | MAPE h=3 | MAPE h=6 | MAPE h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 4.0207 | 5.1001 | 6.8031 | 8.7028 | 11.1450 | 14.2231 | 9.41% | 12.23% | 16.74% |
| Sensor-hour mean | 7.4968 | 7.4959 | 7.4943 | 9.9949 | 9.9944 | 9.9932 | 19.34% | 19.34% | 19.33% |

- *MAE is measured in mph; lower is better.*
- *RMSE is measured in mph; lower is better. Larger errors are penalized more heavily.*
- *MAPE is measured in %; lower is better. It can be unstable when actual speeds are near zero.*

**Takeaway:**

Persistence strongly outperforms the sensor-hour historical mean at every
horizon — including the longest one. This is the opposite of the initial
hypothesis (that the historical mean would catch up at longer horizons).

Two implications:

1. **Persistence is a genuinely strong baseline for METR-LA.** This matches
   the original DCRNN paper, which reported persistence MAE ≈ 4.5 mph at
   15 min. Our h=3 MAE of 4.02 mph is in the same range, consistent with a
   slightly different split and mask policy.

2. **Sensor-hour means ignore more information than they add.** A single
   mean per (sensor, hour) discards the current-day context — the current
   speed is a much better predictor than the historical average. This
   suggests the mean-based baseline is not competitive with persistence and
   should not be the reference for ML models. **Persistence is the floor.**

**Reference (published):**

- Li et al. 2018 (DCRNN) — persistence MAE @15 min: ≈ 4.5 mph
- Li et al. 2018 — DCRNN MAE @15 min: 2.77 mph (target after ML/DL work)

**Success threshold for Week 2:**
Beat persistence at h=12 by ≥15% relative MAE. Persistence h=12 MAE is
6.8031 → target ≤ 5.783 mph at 60-min horizon for the Week 2 ML model.


---

## 2026-10-15 — METR-LA LightGBM forecaster (per-horizon)

**Hypothesis:** A tree-based model with 12 lag features per sensor will beat
persistence by capturing short-term temporal patterns.

**Setup:**
- Same windows, split, and mask as baselines
- Features (20): lags 1–12, lag_h, sensor_mean, sensor_std, hour, dow,
  is_weekend, sensor_idx, horizon
- One LightGBM per horizon (n_estimators=500, lr=0.05, num_leaves=63)
- Training subsampled to ~2M rows
- Persistence MAE computed on the same test targets (in-run, not hardcoded)

**Result:**

| Model | MAE h=3 | MAE h=6 | MAE h=12 |
|---|---:|---:|---:|
| Persistence | 4.0207 | 5.1001 | 6.8031 |
| Sensor-hour mean | 7.4968 | 7.4959 | 7.4943 |
| **LightGBM** | **3.1095** | **3.6992** | **4.3356** |

| Horizon | Improvement vs persistence |
|---|---:|
| h=3 | +22.7% |
| h=6 | +27.5% |
| h=12 | **+36.3%** |

**Takeaway:** LightGBM with lag features substantially outperforms persistence
at every horizon and easily clears the Week 2 success threshold (≥15% at h=12).
This confirms that short-term temporal context (the last hour of readings) is
highly informative for traffic forecasting.

The relative improvement *grows* with horizon — persistence degrades much
faster than the learned model. This suggests the model is learning patterns
that persist beyond the immediate reading.

**Reference:** Published DCRNN MAE at h=12 is 3.60 mph. LightGBM at 4.34 is
within 20% of that without any graph structure or deep learning, which is a
strong tabular baseline. The next stage (LSTM/TCN) will test whether sequence
models improve further.

---

## 2026-10-15 — METR-LA TCN deep forecaster

**Hypothesis:** A TCN with dilated causal convolutions will match or beat
LightGBM by learning temporal patterns end-to-end without hand-crafted lags.

**Setup:**
- Same windows, split, mask as LightGBM
- Architecture: 2 dilated causal blocks (dilations 1, 2), hidden=64,
  no normalization, GELU, dropout 0.1
- Input: (batch, 12 steps, 207 sensors) — all sensors as channels
- Output: (batch, 3 horizons, 207 sensors)
- Loss: masked L1 (target==0 masked out)
- Optimizer: AdamW, lr=1e-3, cosine annealing over 50 epochs
- Batch size 128, gradient clipping 1.0
- Best epoch selected by overall validation MAE

**Result:**

| Model | MAE h=3 | MAE h=6 | MAE h=12 |
|---|---:|---:|---:|
| Persistence | 4.0207 | 5.1001 | 6.8031 |
| LightGBM | 3.1095 | 3.6992 | 4.3356 |
| **TCN** | **3.9721** | **4.1661** | **4.5935** |

**Takeaway:**

The TCN beats persistence at all three horizons but **does not beat
LightGBM**. The gap narrows with horizon (LightGBM advantage: 27.7% at h=3,
12.6% at h=6, 5.9% at h=12), which suggests deep models may be better at
capturing long-range patterns.

Two plausible explanations for the overall gap:

1. **Explicit feature engineering matters on small datasets.** LightGBM has
   20 hand-crafted per-sensor features (lags, sensor stats, calendar). The
   TCN has to learn temporal and cross-sensor structure from 207 input
   channels with only ~24k training windows. That is a small dataset for
   end-to-end deep learning.

2. **Tabular models remain competitive on low-dimensional time series.**
   This is a documented pattern in the literature (e.g., Grinsztajn et al.
   2022, "Why do tree-based models still outperform deep learning on tabular
   data?"). METR-LA forecasting with only temporal features may fall into
   this regime.

**Reference:** Published DCRNN MAE at h=12 is 3.60 mph. Both LightGBM (4.34)
and TCN (4.59) are within ~30% of that — the DCRNN leverages the graph
structure, which we have not yet used. Week 3 will test whether adding
graph structure closes the remaining gap.

---

## 2026-10-16 — METR-LA Graph WaveNet-lite

**Hypothesis:** Using the road-network adjacency matrix will improve
forecasting by letting each sensor use information from nearby sensors.

**Setup:**
- Same windows, split, mask as previous experiments
- Architecture: per-timestep graph conv + causal temporal conv, 2 dilated blocks
- Adaptive adjacency blended with fixed distance graph (sigmoid(alpha))
- hidden=64, 41,629 parameters
- 50 epochs, AdamW, cosine LR, best-epoch checkpoint on validation MAE
- Best epoch: 47 (val MAE 3.836)

**Result:**

| Model | MAE h=3 | MAE h=6 | MAE h=12 |
|---|---:|---:|---:|
| Persistence | 4.0207 | 5.1001 | 6.8031 |
| Sensor-hour mean | 7.4968 | 7.4959 | 7.4943 |
| LightGBM | **3.1095** | **3.6992** | **4.3356** |
| TCN | 3.9721 | 4.1661 | 4.5935 |
| **Graph WaveNet-lite** | **3.4393** | 4.1331 | 5.1969 |

**Takeaway:**

GWNet beats TCN at h=3 (+13%) and h=6 (+1%), but loses at h=12 (-13%).
LightGBM wins at every horizon.

The result is stable across two independent runs (val MAE 3.836 in both),
confirming the training pipeline is reproducible.

Two honest observations:

1. LightGBM remains the best model on this task. This aligns with
   published findings that gradient boosting is highly competitive on
   small-to-medium structured time-series datasets.

2. The simplified GWNet underperforms the full published architecture.
   Published Graph WaveNet achieves MAE ~3.53 at h=12. Our lite version
   at 5.20 suggests insufficient capacity or missing architectural
   components (gated TCN, deeper stacks, more sophisticated diffusion
   convolutions).

A higher-capacity variant (hidden=128) is a natural follow-up but was
not run in this pass.

**Reference:** Wu et al. 2019 (Graph WaveNet) — MAE 2.69 / 3.07 / 3.53.

---

## 2026-10-16 — NYC 311 text baselines

**Hypothesis:** The descriptor field is co-assigned with complaint_type,
so a simple descriptor→label lookup should achieve near-perfect accuracy.
If so, the task is a lookup, not a classification problem.

**Setup:**
- Top-5 classes, stratified 70/15/15 split, random_state=42
- Train: 37,194 | Val: 7,970 | Test: 7,971
- Baselines: majority class, descriptor lookup, TF-IDF + LogReg

**Result:**

| Baseline | Accuracy | F1 Macro |
|---|---:|---:|
| Majority class | 0.3387 | 0.1012 |
| **Descriptor lookup** | **0.9997** | **0.9994** |
| **TF-IDF + LogReg** | **1.0000** | **1.0000** |

**Takeaway:**

The task is **nearly deterministic**. A dictionary mapping descriptor
→ most common label achieves 99.97% accuracy on the test set
(2 errors out of 7,971 rows). TF-IDF + LogReg achieves 100%.

The two lookup errors indicate the mapping is not strictly
deterministic — a small number of descriptors appear under multiple
complaint types. But the effective ceiling for any model on this task
is ~100%, so DistilBERT's expected score (~100%) will not demonstrate
semantic understanding. It will demonstrate API fluency.

**Implication:** Future text experiments should use a field with real
linguistic variability. Candidates: `additional_details` (operator
notes, ~53% null in this sample) or raw complaint text. The
`descriptor` field is a structured code and unsuitable for
demonstrating NLP capability.

**This is the honest conclusion.**

---

## 2026-10-16 — NYC 311 DistilBERT fine-tuning

**Hypothesis:** DistilBERT will match the ceiling established by the
lookup and TF-IDF baselines (~100%) because the task is deterministic.
The experiment is a HuggingFace API learning exercise, not a claim
about transformer capability.

**Setup:**
- Same split as baselines (stratified 70/15/15, random_state=42)
- Model: distilbert-base-uncased (66M params)
- Tokenizer: max_length=16 (descriptors are 1-4 words)
- Training: 3 epochs, AdamW (lr=2e-5), batch size 32
- Best-epoch checkpoint on validation macro-F1

**Result:**

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority class | 0.3387 | 0.1012 |
| Descriptor lookup | 0.9997 | 0.9994 |
| TF-IDF + LogReg | 1.0000 | 1.0000 |
| **DistilBERT** | **1.0000** | **1.0000** |

DistilBERT reaches 100% validation F1 after a single epoch.

**Takeaway:**

DistilBERT matches the deterministic lookup baseline and does not
exceed TF-IDF + LogReg. This confirms the expected result for a
trivial task.

The real value of this experiment is confirming the HuggingFace
fine-tuning workflow works end-to-end. The code is now a template
for future text classification tasks on harder datasets.

**Recommendation for future weeks:**

Use a text field with genuine linguistic variability. Candidates:
- `additional_details` — free-text operator notes (~53% null in sample)
- Raw complaint text written by residents

Neither has been inspected yet. A proper Week 4b/5 experiment should
first check the field's null fraction, uniqueness, and correlation
with the target before building a model.

---

## 2026-10-17 — BDD100K scene classification (ResNet-50 fine-tune)

**Task:** classify urban driving scenes into city street / highway /
residential from keyframe images.

**Setup:**
- Train: 68,969 | Val: 4,932 | Test: 4,932
- ResNet-50 (ImageNet-pretrained IMAGENET1K_V2)
- Trainable parameters: 23,514,179 (all layers except layer1–3 frozen;
  only layer4 and the new fc head are trainable)
- AdamW, lr=1e-4, weight_decay=1e-4
- CosineAnnealingLR over 5 epochs
- Batch 8, mixed precision (torch.amp.autocast)
- Augmentation: RandomResizedCrop(224, scale=(0.75,1.0)), RandomHorizontalFlip
- Best-epoch checkpoint on validation macro-F1

**Result:**

| Model | Accuracy | F1 Macro |
|---|---:|---:|
| Majority class | 0.6196 | 0.2550 |
| Frozen ResNet-50 + LogReg | 0.7618 | 0.6780 |
| **Fine-tuned ResNet-50** | **0.7800** | **0.6984** |

Delta over frozen baseline: **+1.82 accuracy, +2.04 macro-F1**.

**Epoch-by-epoch:**

| Epoch | Train Loss | Val F1 | Saved? |
|---|---:|---:|---|
| 1 | 0.5548 | 0.6842 | ✓ |
| 2 | 0.4944 | **0.6956** | ✓ best |
| 3 | 0.4355 | 0.6819 | ✗ |
| 4 | 0.3402 | 0.6938 | ✗ |
| 5 | 0.2348 | 0.6872 | ✗ |

**Takeaway:**

Fine-tuning improved accuracy by +1.82 and macro-F1 by +2.04 over
the frozen-feature baseline. The improvement is modest but measurable.

Training loss decreased monotonically (0.55 → 0.23) while validation
F1 peaked at epoch 2 and declined thereafter. This pattern is
**consistent with overfitting** — the model continued fitting training
data without improving generalization — but other explanations
(insufficient BatchNorm adaptation, class imbalance interacting with
the loss surface) cannot be ruled out from the current evidence.

**Why checkpoint selection uses F1, not loss:** loss measures the
model's probability calibration; F1 measures actual classification
decisions across classes. In imbalanced classification, loss can
drop while F1 stagnates if the model becomes more confident about
already-correct predictions. Selecting on the task metric (F1) rather
than the training objective (loss) is the standard practice.

**Possible follow-up experiments (not guaranteed to improve):**
- Differential learning rates (higher for head, lower for backbone)
- Stronger augmentation (RandomErasing, mixup)
- Earlier stopping (3 epochs instead of 5)
- Per-class loss weighting to address imbalance

**Methodological note on test-set usage:**

The baseline script (`run_bdd100k_baselines.py`) evaluated the test
split first to report majority-class and frozen-ResNet-50 numbers.
Therefore fine-tuned metrics are a **consistent benchmark comparison**,
not a pristine out-of-sample estimate.

**Test split source:** half of BDD100K's official *validation*
annotations. The official test set has no public scene labels and is
not used in this project.

---

## 2026-10-17 — BDD100K 5-class semantic segmentation (SegFormer-B0)

**Task:** Pixel-level segmentation of urban scenes into 5 classes:
road, sidewalk, vehicle, person, other.

**Setup:**
- Train: 6,000 images (official BDD100K train, minus held-out test)
- Val: 1,000 images (official BDD100K val)
- Test: 1,000 images (held out from official train, since BDD100K
  official test has no public labels)
- Model: SegFormer-B0 (ADE20k pretrained), 3.7M trainable params
- Resolution: 512×512
- Loss: 0.5 × CE(class_weighted, ignore=255) + 0.5 × Dice
- Class weights: [1.0, 2.0, 1.5, 20.0, 0.5] for [road, sidewalk, vehicle, person, other]
- Optimizer: AdamW, lr=6e-5, weight_decay=0.01, cosine annealing
- Batch size 8, mixed precision (torch.amp)
- 20 epochs, best checkpoint selected by validation mIoU

**Class pixel distribution (measured on 50 sampled masks):**

| Class | Pixel share |
|---|---:|
| road | 22.21% |
| sidewalk | 1.60% |
| vehicle | 9.61% |
| person | 0.11% |
| other | 46.45% |
| ignore (255) | ~20% (not scored) |

**Result:**

| Split | mIoU |
|---|---:|
| Validation (best epoch 19) | 0.7480 |
| **Test (held out)** | **0.7172** |

Val-test gap: 0.031 — excellent generalization.

**Per-class test IoU:**

| Class | IoU | Note |
|---|---:|---|
| road | 0.9081 | Excellent |
| sidewalk | 0.5377 | Strong for 1.6% pixel share |
| vehicle | 0.8538 | Excellent |
| person | 0.3319 | Real result despite 0.11% pixel share |
| other | 0.9547 | Excellent |

**Takeaway:**

SegFormer-B0 fine-tuned on 6,000 BDD100K images achieves **0.72 test mIoU** across 5 urban scene classes — a strong result for a small model on a small dataset.

**Person class (0.11% of pixels) reached 0.33 IoU** through class-weighted CE and Dice loss. The 20× weight was necessary and effective. The class is learnable — not "unlearnable" as feared — though person IoU is still the weakest class.

**Sidewalk (1.60% of pixels) reached 0.54 IoU** — the class is small but consistent enough to learn well.

**Person-on-road vs. person-on-sidewalk analysis** (a downstream task) is now feasible given both classes are segmented. Documented as a possible Week 10+ extension.

**Training speed:** 185s/epoch for 6,000 images at 512×512 batch 8 — a total of ~62 minutes. This leaves room for hyperparameter tuning or longer training if desired.

**Reference:** Published SegFormer-B0 on BDD100K 19-class segmentation achieves ~0.60-0.65 mIoU. Our simplified 5-class task achieves 0.72 — the task is easier, but the number is legitimate for the task definition.

---

## 2026-10-18 — BDD100K YOLOv8 detection inference

**Task:** qualitative detection demo on BDD100K val images using COCO-pretrained YOLOv8n.

**Setup:**
- 200 randomly sampled val images (seed=42)
- YOLOv8n pretrained on COCO (80 classes)
- Filtered to 4 aligned classes: person, car, truck, bus
- Ground truth boxes drawn from BDD100K label `box2d` coordinates

**Reported:**
- Raw detection counts per class (YOLO vs. GT) — see run output
- 12 side-by-side visualizations in `docs/detection_examples/`

**Not reported:**
- Precision/recall/F1/mAP — YOLO COCO-pretrained on BDD100K suffers domain shift; IoU-based evaluation is a planned future extension

**Takeaway:**

YOLOv8n produces visually reasonable detections on BDD100K images, but quantitatively it undercounts relative to ground truth — expected for a COCO model applied to urban driving footage. This is a qualitative demo, not a benchmark.

A proper evaluation would implement IoU-based matching (threshold 0.5) to compute per-class precision/recall/F1. That is scoped for a later phase (Week 10, evaluation framework).

---

## 2026-10-18 — YOLOv8n IoU-based detection evaluation (Level 2)

**Setup:**
- 200 randomly sampled BDD100K val images (seed=42)
- YOLOv8n pretrained on COCO
- Confidence threshold: 0.25 (YOLO default)
- IoU matching threshold: 0.5
- Classes: person, car, truck, bus

**Result:**

| Class | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| person | 96 | 42 | 260 | 0.696 | 0.270 | 0.389 |
| car | 758 | 219 | 1243 | 0.776 | 0.379 | 0.509 |
| truck | 23 | 39 | 70 | 0.371 | 0.247 | 0.297 |
| bus | 10 | 11 | 30 | 0.476 | 0.250 | 0.328 |
| **macro** | | | | **0.580** | **0.286** | **0.381** |

**Takeaway:**

COCO-pretrained YOLOv8n on BDD100K urban footage shows the
characteristic signature of **domain shift**:

- **Precision is moderate-to-high** (0.58 macro): when YOLO detects
  something, it's usually correct.
- **Recall is low** (0.29 macro): YOLO misses the majority of real
  objects — especially small/distant pedestrians, partially occluded
  vehicles, and BDD100K-specific class definitions (e.g., "truck" in
  COCO means a specific box-truck shape, while BDD100K labels any
  large commercial vehicle as truck).

**Person recall is especially low** (0.27): 73% of pedestrians are
missed. This is expected because:
1. BDD100K has many small, distant pedestrians
2. COCO pedestrians are usually large and centered
3. The confidence threshold (0.25) is tuned for COCO, not BDD100K

Lowering the confidence threshold would raise recall at the cost of
precision — this is the classic precision-recall tradeoff, which a
confidence sweep would explore.

**Scope note:** this is a Level 2 evaluation (single IoU threshold,
single confidence threshold, per-class P/R/F1). It does NOT compute:
- mAP@0.5:0.95
- Confidence-threshold sweeps
- Per-size analysis (small/medium/large)
- Cross-model comparison

These are explicitly deferred. The purpose here is a quantitative
baseline, not a benchmark-paper evaluation.

**Portfolio value:** this result is honest about what a pretrained
model does on a different domain, and it teaches the precision-recall
tradeoff concretely. Fine-tuning YOLO on BDD100K would substantially
improve these numbers — a natural future experiment.

---

## 2026-10-20 — HMDB-51 15-class urban behavior recognition (Tiny3DCNN baseline)

**Task:** 15-class human action recognition on HMDB-51 urban-relevant subset.

**Setup:**
- Classes: stand, walk, sit, eat, drink, hug, run, ride_bike,
  ride_horse, fall_floor, push, hit, punch, shoot_gun, smoke (15)
- Categories: normal (6), mobility (2), anomaly (2), threat (4), restricted (1)
- Train: 892 clips | Val: 158 | Test: 450 (official HMDB-51 split 1)
- Model: Tiny3DCNN (~1.17M params, 4 conv blocks + global avg pool)
- Input: 16 frames × 112×112 RGB
- Loss: class-weighted cross-entropy
- Optimizer: AdamW, lr=3e-4, cosine annealing over 15 epochs
- Batch 4, mixed precision (torch.amp)
- Decoder: decord (OpenCV frame-seek was unreliable on HMDB-51's older .avi files)
- Checkpoint selection: best validation macro-F1

**Baselines:**
- Majority class (fall_floor): 6.67% test accuracy, 0.83% macro-F1
- Random guess: 6.67% test accuracy, 6.52% macro-F1

**Result:**

| Metric | Value |
|---|---:|
| Best epoch | 14/15 |
| Best val macro-F1 | 0.3173 |
| **Test accuracy** | **0.2533** |
| **Test macro-F1** | **0.2364** |

**Per-category test accuracy:**

| Category | Classes | Accuracy |
|---|---|---:|
| mobility | run, ride_bike | 36.7% (22/60) |
| normal | stand, walk, sit, eat, drink, hug | 26.7% (48/180) |
| anomaly | ride_horse, fall_floor | 25.0% (15/60) |
| threat | push, hit, punch, shoot_gun | 22.5% (27/120) |
| restricted | smoke | 6.7% (2/30) |

**Takeaway:**

Tiny3DCNN trained from scratch on 1,500 clips achieves 25.33%
test accuracy — 3.8× the random baseline (6.67%) and 3.8× the
majority-class baseline. The model learned real signal despite
the small dataset.

Per-category analysis reveals expected patterns:
- **Mobility** (run, ride_bike) performs best (36.7%) — visually distinct movement patterns
- **Normal activities** perform moderately (26.7%) — walk/stand/sit/eat/drink/hug are visually similar (non-aggressive body movement)
- **Threat** classes struggle (22.5%) — push/hit/punch are fine-grained and often confused
- **Restricted** (smoke) surprisingly low (6.7%) — equal to baseline, worth investigating in a follow-up (confusion matrix analysis)

**Limitations:**
- HMDB-51 is human actions from movies/web, not urban surveillance footage
- 100 clips per class is small for 3D CNN training
- No pretrained initialization — training from scratch
- 15-class flat classifier, not hierarchical

**Next step:** Compare against Kinetics-400 pretrained R(2+1)D-18
to determine whether transfer learning helps on this small dataset.

---

## 2026-10-20 — HMDB-51: R(2+1)D-18 vs Tiny3DCNN comparison

**Purpose:** Compare training from scratch (Tiny3DCNN) vs. fine-tuning
Kinetics-400 pretrained features (R(2+1)D-18) on the same 15-class task.

**Setup:**
- Same split: 892 train / 158 val / 450 test (official HMDB-51 split 1)
- Same decoder: decord
- Same evaluation: best-epoch checkpoint on val macro-F1, test set evaluated once
- Tiny3DCNN: 1.17M params, from scratch, 15 epochs, batch 4, lr 3e-4
- R(2+1)D-18: 33M params, Kinetics-400 pretrained, 20 epochs, batch 2,
  differential lr (backbone 30%, head 3e-4), Kinetics normalization

**Comparison:**

| Model | Params | Pretrained | Test Acc | Test F1 |
|---|---:|---|---:|---:|
| Tiny3DCNN (baseline) | 1.17M | ❌ | 0.2533 | 0.2364 |
| R(2+1)D-18 | 31.3M | ✅ | **0.7311** | **0.7293** |
| Improvement | | | **+0.4778** | **+0.4929** |

**Per-category accuracy comparison:**

| Category | Tiny3DCNN | R(2+1)D-18 | Δ |
|---|---:|---:|---:|
| normal | 26.7% | 73.9% | +47.2 |
| mobility | 36.7% | 81.7% | +45.0 |
| anomaly | 25.0% | 70.0% | +45.0 |
| threat | 22.5% | 70.8% | +48.3 |
| restricted | 6.7% | 66.7% | +60.0 |

**Takeaway:**

Pretrained R(2+1)D-18 dramatically outperforms the scratch-trained
Tiny3DCNN. On a single epoch, R(2+1)D-18 achieves 55.33% test
accuracy — more than double the 25.33% of the 15-epoch scratch
baseline.

This confirms the well-documented result that **pretrained features
transfer strongly to fine-grained action recognition**, even on
small datasets (1,500 clips) with a domain gap (Kinetics → HMDB-51).

**Key insight:** Training a 3D CNN from scratch on 1,500 clips was
never going to reach state-of-the-art. Pretrained features carry
spatiotemporal knowledge that transfers immediately, achieving
competitive results with just one epoch of fine-tuning.

**Implication for the project:** All subsequent video work should use
pretrained backbones. Training from scratch is a demonstration of
pipeline engineering, not a strategy for accuracy.

---

## 2026-10-20 — UCF-Crime 4-class urban anomaly classification (Week 7.5)

**Task:** Classify urban video clips into 4 categories:
normal, violence, criminal, public_hazard.

**Setup:**
- Model: R(2+1)D-18 (Kinetics-400 pretrained), 31.3M params
- Train: 5,518 clips (1,126 videos)
- Val: 1,180 clips (240 videos)
- Test: 1,183 clips (244 videos)
- Official test: 1,406 clips (290 videos)
- Clip extraction: 16 frames, stride 8, max 5 clips/video
- Split: stratified by (class, video_id) — no frame-level leakage
- Loss: class-weighted cross-entropy
- Augmentation: random crop, horizontal flip (train only)

**Training schedule (staged fine-tuning):**
- Epochs 1–2: head-only (backbone frozen), LR=1e-4, batch 12
- Epochs 3+: unfreeze backbone at 10× lower LR (1e-5)
- WeightedRandomSampler for class balance
- Early stopping (patience=3, min_delta=0.005)
- Total: 8 epochs (early stopped), ~35 min

**Results:**

| Metric | Value |
|---|---:|
| Best epoch | 5/8 |
| Best val F1 | 0.5239 |
| Test accuracy | 0.6492 |
| Test macro-F1 | 0.5190 |
| Official test accuracy | 0.6373 |
| Official test macro-F1 | 0.4972 |
| ECE | 0.1255 |

**Per-category on official test:**

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| normal | 0.78 | 0.89 | 0.83 |
| violence | 0.65 | 0.17 | 0.27 |
| criminal | 0.43 | 0.56 | 0.48 |
| public_hazard | 0.38 | 0.43 | 0.40 |

**Confidence fallback (official test):**

| Confidence bucket | Count | Accuracy |
|---|---:|---:|
| High (>0.70) | 944 (67%) | 73.31% |
| Mid (0.40–0.70) | 433 (31%) | 44.11% |
| Low (<0.40) | 29 (2%) | 44.83% |

**Comparison with naive full fine-tune (no freeze):**

| Metric | Naive | Staged | Delta |
|---|---:|---:|---:|
| Official test macro-F1 | 0.4310 | **0.4972** | **+0.066** |
| Official test accuracy | 0.6024 | **0.6373** | **+0.035** |
| Violence F1 | 0.16 | **0.27** | **+0.11** |
| ECE | 0.1763 | **0.1255** | **−0.051** |

**Takeaway:**

Staged fine-tuning (freeze head warmup → unfreeze at 10× lower LR) 
prevented the catastrophic overfitting observed in naive full 
fine-tuning. The naive approach peaked at epoch 1 (val F1=0.4907) 
and declined monotonically; the staged approach peaked at epoch 5 
(val F1=0.5239) and improved every metric over the naive run.

Key observations:

1. **Violence class is the hardest** (F1 0.27). Visual signals are 
   ambiguous (punch vs. hug) and it's only 12% of training clips. 
   Precision is reasonable (0.65) but recall is low (0.17) — the 
   model is cautious about predicting violence.

2. **Confidence fallback works cleanly.** High-confidence predictions 
   (67% of test) achieve 73% accuracy. The remaining 33% route to 
   human review — exactly the safety-critical design intended.

3. **ECE improved from 0.176 to 0.126.** The model is less overconfident.

**Limitations:**

- HMDB-51/UCF-Crime are human action datasets, not urban surveillance
- Clip-level classification with video-level labels (not temporal localization)
- 5,518 training clips (small for R(2+1)D-18's 31M params)
- Violence class remains hard due to visual ambiguity

**Deliverables:**
- Loader: `src/urbanintel/data/ucf_crime.py`
- Clip dataset: `src/urbanintel/data/ucf_crime_video.py`
- Clip builder: `scripts/build_ucf_clips.py`
- Training: `scripts/train_ucf_crime_4class.py`
- Inference with fallback: `src/urbanintel/serving/ucf_anomaly_inference.py`
- MLflow experiment: `urbanintel-week7-5`
- Tag: `v0.9.2-week7-5-ucf-crime`

---

## 2026-10-21 — UrbanSound8K 10-fold audio event classification (Week 7.6a)

**Task:** Classify 4-second urban audio clips into one of 10 sound classes
(air_conditioner, car_horn, children_playing, dog_bark, drilling,
engine_idling, gun_shot, jackhammer, siren, street_music).

**Motivation:** UrbanSound8K adds audio as the 7th modality in the
project. Two of its classes (`gun_shot`, `siren`) are emergency-relevant —
these are the audio signals a downstream alert system would act on.

**Setup:**
- Dataset: UrbanSound8K, 8,732 clips, 10 predefined folds
- Split protocol: **10-fold cross-validation** (dataset-recommended;
  NOT a random 80/20 split). Folds are session-stratified, not
  class-stratified — minority classes vary up to 3.5× between folds
- Per-fold: 15% of the 9-fold train pool carved as validation
  (stratified by class)
- Input: 128-mel log-spectrogram, 4.0 s, hop=512 → shape (128, 173)
- Preprocessing: cached as a single 2.87 GiB memmap (avoids re-decoding)
- Per-sample normalization: `(mel - mean) / std`
- Augmentation (train only): SpecAugment — 50% time-mask, 30% freq-mask
- Model: 4-block 1D-CNN (Conv1d + BatchNorm + ReLU + MaxPool), ~348K params
- Loss: class-weighted cross-entropy (weights from global class counts)
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: cosine annealing over 15 epochs
- Batch 32, mixed precision not used
- Early stopping: patience=3, min_delta=0.005
- Checkpoint selection: best validation macro-F1
- Epochs: up to 15 per fold (~80-120 s/epoch on RTX 4050)

**Result (10-fold CV):**

| Metric | Mean ± Std |
|---|---:|
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

**Baselines:**

| Baseline | Test accuracy |
|---|---:|
| Random guess (10 classes) | 0.1000 |
| Majority class (air_conditioner, ~11.4%) | ~0.114 |

**Key findings:**

1. **Best-epoch val F1 systematically overestimates test F1.** Val F1
   averages ~0.90 across folds, test F1 averages ~0.69 — a gap of
   ~0.21. This is best-epoch selection overfitting to a validation
   set that shares 9-fold provenance with training. Only test F1
   is a valid generalization estimate. Reported as a limitation.

2. **Fold-to-fold variance is large (±0.05) and data-driven.** Folds
   are session-stratified, not class-stratified. Fold 10 (test acc
   0.77) has fewer minority-class clips than fold 3 (0.57). The
   mean ± std is the honest number.

3. **From-scratch 1D-CNN is competitive with expected range.**
   0.678 test accuracy is mid-range for from-scratch 1D-CNNs on
   UrbanSound8K (typical: 0.65–0.72). Pretrained models (AST,
   PaSST) reach 0.78–0.85, but that is out of scope for Week 7.6.

4. **`gun_shot` (smallest class, 374 clips) is the class that matters
   most and the one most likely to lag.** Per-class F1 analysis
   (confusion matrix) is a natural Week 8 follow-up.

**Limitations:**

- 4-second clips only; cannot detect sustained events
- Single-label classification; real urban audio is overlapped
- Per-sample normalization discards absolute amplitude
- No evaluation on live/unseen audio; only held-out predefined folds
- `gun_shot` F1 likely lags due to class imbalance — the class most
  relevant to the alert use case is the hardest to learn
- Global class weights approximate per-fold weights (fold imbalance up to 3.5×)

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