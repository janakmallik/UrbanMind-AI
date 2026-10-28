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

## 2026-10-22 — Week 8 report baseline (Session 1a)

**Task:** Establish a template baseline for incident report generation
from multimodal evidence.

**Setup:**
- 31 synthetic evidence bundles (see dataset card)
- Template baseline: deterministic enumeration of evidence (no LLM)
- Evaluation: 6 metrics (see week8_plan.md)
- MLflow experiment: `urbanintel-week8`

**Result (template baseline):**

Results are measured by running `scripts/evaluate_reports.py --approach template`.
Paste actual numbers here after first run.

**Takeaway:**

The template is expected to score high on evidence_fidelity,
signal_coverage, no_hallucination, and severity consistency (it enumerates
what's in the evidence). It is expected to score low on
uncertainty_ack, recommendation_quality, and epistemic_clarity (it does
not reason or recommend). This is a hypothesis, not a measured result.

Session 1b will run Qwen2.5-1.5B zero-shot on the same 31 scenarios and
compare. The measured difference — if any — is the deliverable.

**Deliverables:**
- Schema: `src/urbanintel/reporting/schema.py`
- Template: `src/urbanintel/reporting/template_baseline.py`
- Scenarios: `src/urbanintel/reporting/scenarios.py`
- CLI: `scripts/report_baseline.py`
- Evaluation: `scripts/evaluate_reports.py`
- Dataset card: `docs/dataset_cards/week8_synthetic_scenarios.md`
- Tag: `v0.9.7-week8-schema`

---

## 2026-10-22 — Week 8: Incident report generation (Qwen vs template)

**Task:** Generate a natural-language incident report from structured
evidence (video anomaly, audio events, image scene, segmentation,
traffic forecast, complaints) — decision-support for a human responder,
not automated enforcement.

**Setup:**
- 31 synthetic scenarios (see `docs/dataset_cards/week8_synthetic_scenarios.md`)
- Template baseline: deterministic enumeration of evidence
- Qwen baseline: `Qwen2.5-1.5B-Instruct` (4-bit NF4), few-shot prompting
- Evaluation: 5 constraint checks (see `docs/week8_1_policy.md`)
- MLflow experiment: `urbanintel-week8-1`

**Result:**

| Metric | Template | Qwen | Δ |
|---|---:|---:|---:|
| Source availability | 0.763 | 0.839 | +0.076 |
| Forbidden phrase | 1.000 | 0.871 | −0.129 |
| Scenario specific | 0.952 | 0.984 | +0.032 |
| Policy compliance | 0.032 | 0.903 | +0.871 |
| Section separation | 0.000 | 1.000 | +1.000 |
| **Overall** | **0.626** | **0.902** | **+0.276** |

**Key findings:**

1. **Qwen outperforms template significantly** — from 0.626 to 0.902
   across the honest metrics. The template is a log dump; Qwen produces
   a structured field report with OBSERVED / INTERPRETATION / UNKNOWN /
   RECOMMENDED FOLLOW-UP sections.

2. **The first (unfixed) evaluator was tautological.** Original metrics
   (evidence_fidelity, signal_coverage, no_hallucination,
   severity_consistency) were broken — `no_hallucination` was guaranteed
   to score 1.0 by construction. The initial 0.956 score was inflated.

3. **Qwen's remaining weakness is forbidden phrase violations** — 5 of
   31 scenarios contain at least one prohibited phrase (e.g., "criminal
   activity" when the model should say "the classifier placed this event
   in its criminal category"). Documented as a small-model limitation.

4. **The policy was revised** — Week 8.1 changed "escalate = dispatch"
   to "escalate = urgent human review." The system is decision-support,
   not dispatch. This change is enforced in the prompt and the evaluator.

**Limitations:**
- 31 synthetic scenarios, not real incidents
- Cross-modal co-occurrence is synthetic (video and audio datasets not paired)
- Qwen-1.5B is small; a larger model would reduce forbidden phrase violations
- Human rubric not performed

**Deliverables:**
- Schema: `src/urbanintel/reporting/schema.py`
- Template: `src/urbanintel/reporting/template_baseline.py`
- Qwen generator: `src/urbanintel/reporting/qwen_generator.py`
- Header: `src/urbanintel/reporting/header.py`
- Scenarios: `src/urbanintel/reporting/scenarios.py`
- Policy: `docs/week8_1_policy.md`
- Model card: `docs/model_cards/week8_report_qwen_v1.md`
- Dataset card: `docs/dataset_cards/week8_synthetic_scenarios.md`
- MLflow: `urbanintel-week8-1`
- Tags: `v0.9.7-week8-schema`, `v0.9.8-week8-qwen`

---

## 2026-10-24 — Week 9.5 gate: NYC 311 tabular+text fusion viability audit

**Hypothesis:** Fuse 311 tabular features + descriptor text to predict
resolution time (`resolved within 7 days`). If text predicts resolution
better than categorical encoding, this is true cross-modal fusion.

**Data:** NYC 311, 100,000 complaints, 98,926 with valid created+closed
dates, date range 2020-07-13 to 2026-10-01.

**Result:**
- Target: 86.0% close in ≤7 days, 14.0% take longer
- Majority-class macro-F1: 0.123 (imbalanced target)
- TF-IDF + LogReg 5-fold CV: 0.802 ± 0.015
- Descriptor one-hot + LogReg 5-fold CV: 0.804 ± 0.018
- **TF-IDF − one-hot: −0.002**

**Interpretation:** The descriptor behaves as a categorical lookup, not
linguistic text. TF-IDF and one-hot encoding produce identical F1 (within
noise). DistilBERT would similarly fail to beat categorical encoding.

**Decision:** 311 text+tabular fusion is **NOT viable** as a true
cross-modal experiment. The "text" is a categorical lookup, not linguistic
signal. Same trap as Week 4's trivial DistilBERT task.

**Documented as negative result.** Week 9.5 does not proceed. Cross-modal
fusion requires genuinely different input types (e.g., audio + video on
paired data), not categorical encodings in text format.

**Deliverables:**
- Audit script: `scripts/audit_nyc311.py`
- Raw result: 5-fold CV on 98,926 complaints
- Tag: included in `v0.9.9-week9-multiview-fusion` documentation

---

## 2026-10-25 — Week 9a: Multi-view fusion on BDD100K (negative result)

**Task:** Fuse ResNet-50 + SegFormer-B0 + YOLOv8n features to predict
scene classification (city / highway / residential) on BDD100K.

**Setup:**
- 14 features: 3 ResNet probs + 5 SegFormer pixel shares + 6 YOLO counts
- Model: LightGBM classifier with `class_weight="balanced"`
- Train: 4,727 clean images (ResNet's val split, contamination filtered)
- Test: 4,745 clean images (ResNet's test split, contamination filtered)
- Protocol: 5-fold stratified CV on train; test evaluated per
  configuration × 3 seeds (42, 123, 7). Mean ± std reported.
- Contamination filter: 187 images excluded from test, 205 from train
  (SegFormer train ∩ ResNet split)
- Feature provenance: 4,727 train images are from ResNet's **validation**
  split (not training). Overlap with ResNet train IDs = 0.

**Result:**

| Model | Test acc | Test macro-F1 |
|---|---:|---:|
| ResNet-only stacker | 0.7043 | 0.6662 |
| SegFormer-only stacker | 0.5355 | 0.4790 |
| YOLO-only stacker | 0.4872 | 0.4562 |
| **Full fusion (14)** | **0.7372** | **0.6910** |
| Ablation: no ResNet | 0.5903 | 0.5325 |
| Ablation: no SegFormer | 0.7216 | 0.6804 |
| Ablation: no YOLO | 0.7357 | 0.6899 |
| **Original ResNet** | **0.7819** | **0.6986** |

**Bootstrap CI on macro-F1 difference (fusion − ResNet):**
- Point estimate: −0.0076
- 95% CI: [−0.0229, +0.0091]
- P(fusion > ResNet): 0.191
- Method: 1,000 resamples, paired, percentile

**Per-class (clean_test, n=4,745):**

| Model | Class | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Fusion | city | 0.85 | 0.73 | 0.79 |
| Fusion | highway | 0.68 | 0.78 | 0.72 |
| Fusion | residential | 0.48 | **0.67** | 0.56 |
| ResNet | city | 0.80 | 0.88 | 0.84 |
| ResNet | highway | 0.74 | 0.72 | 0.73 |
| ResNet | residential | 0.74 | **0.41** | 0.53 |

**Key findings:**

1. **Fusion does NOT improve overall macro-F1.** −0.0076 point estimate,
   95% CI includes zero. Cannot claim better or worse statistically. The
   point estimate favors ResNet.

2. **Ablation shows ResNet dominates.** Removing ResNet drops F1 from
   0.6910 to 0.5325 (−0.158). SegFormer contributes +0.011 (marginal).
   YOLO contributes +0.001 (nothing — redundant with ResNet's visual
   features).

3. **Fusion improves residential recall dramatically.** +0.26 (0.41 →
   0.67), at the cost of overall accuracy (−0.045) and city-street recall
   (−0.15). This is a precision-recall trade-off, not a general win.

4. **The trade-off is a legitimate application.** When accuracy-equity
   trade-offs matter (e.g., detecting residential scenes is prioritized
   over overall accuracy), fusion is a valid tool.

**Limitations:**
- Multi-view fusion on one modality, not cross-modal
- Test set reused 22 times (exploratory, not untouched benchmark)
- ResNet val-split features are post-checkpoint-selection, not pristine
- Small training data (4,727 images)

**Deliverables:**
- Model card: `docs/model_cards/bdd100k_multiview_fusion_v1.md`
- Training script: `scripts/train_bdd100k_fusion.py`
- Feature extraction: `scripts/extract_fusion_features.py`
- Raw metrics: `data/fusion_results/session4_results.json`
- Tag: `v0.9.9-week9-multiview-fusion`

---

## 2026-10-27 — Week 9c-audio: Urbansas offscreen-vehicle audio classifier

**Hypothesis:** A from-scratch 1D-CNN on 128-mel log-spectrograms can
detect `offscreen` vehicle events in 10-second urban audio clips,
generalizing across Madrid locations (MAVD) and failing on European
cities (TAU). This is the audio-only half of the Week 9c cross-modal
fusion experiment.

**Setup:**

- **Dataset:** Urbansas, 1,080 clips (540 MAVD + 540 TAU).
  MAVD training, TAU held out.
- **Target:** `has_offscreen = "offscreen" in audio_annotations[filename]`.
  MAVD positive rate 25.6%; TAU positive rate 15.0%.
- **Protocol:** 5-fold leave-one-location-out on MAVD. Final model
  retrained on all MAVD, evaluated once on TAU at threshold 0.5
  (fixed before TAU evaluation). No tuning on TAU.
- **Model:** 4-block 1D-CNN, ~348K params, same architecture as
  Week 7.6 `AudioCNN`.
- **Preprocessing:** 48 kHz mono → 128-mel log-spectrogram
  (n_fft=2048, hop=2048, n_mels=128, fmin=20, fmax=24000) →
  per-sample normalize. Cached at `data/urbansas_mels/mels.npy`.
- **Training:** AdamW lr=1e-3, wd=1e-4, cosine over 15 epochs,
  batch 16, class-weighted CE (`pos_weight = n_neg/n_pos` per fold).
  No augmentation.
- **Seeds:** 42 + fold_idx (per fold), 142 (final).

**Result — MAVD CV:**

| Fold | Location | F1 | Precision | Recall | Positives |
|---:|---|---:|---:|---:|---:|
| 1 | `acevedo0103` | 0.4138 | 0.3273 | 0.5625 | 32 |
| 2 | `magallanes0308` | 0.2154 | 0.4118 | 0.1458 | 48 |
| 3 | `rivera0922` | 0.5000 | 0.5000 | 0.5000 | 2 |
| 4 | `rivera0923` | 0.4483 | 0.4333 | 0.4643 | 28 |
| 5 | `templete0103` | 0.3797 | 0.2941 | 0.5357 | 28 |

**Mean F1 (unweighted):** 0.3914 ± 0.0966
**Pooled OOF:** F1 0.3686, P 0.3484, R 0.3913, ROC-AUC 0.6237, PR-AUC 0.3296

**Per-location ROC-AUC is bimodal:**

| Location | ROC-AUC |
|---|---:|
| `acevedo0103` | 0.481 |
| `magallanes0308` | 0.703 |
| `rivera0922` | 0.500 (n=2 pos, noise) |
| `rivera0923` | 0.687 |
| `templete0103` | 0.507 |

**Two locations transfer; two are at chance.** The cause is not
identified. Two hypotheses were tested and refuted:

1. *Class composition.* The "both" subclass fraction is 12-20%
   across the four meaningful folds with no correlation to ROC-AUC.
   `rivera0923` has the highest "both" fraction (15%) and scores
   0.687. **Refuted.**
2. *Night vs. day.* Only `rivera0922` has night clips, and
   `rivera0922` is not one of the meaningful folds. All four
   meaningful folds are 100% daytime. **Refuted.**

Working hypothesis is acoustic domain shift between Madrid locations
(microphone geometry, street reflections, background noise). Untested
in this experiment.

**Result — per-subclass ROC-AUC (vs `only_visible` + `empty` negatives):**

| Positive subclass | n | ROC-AUC |
|---|---:|---:|
| `only_offscreen` | 66 | 0.660 |
| `both` (offscreen + visible) | 72 | 0.590 |

**Gap of 0.070** in the predicted direction, but 0.660 for the
"easiest" subgroup is still weak — the model does not confidently
detect even pure offscreen events.

**Result — TAU external:**

| Metric | Value |
|---|---:|
| ROC-AUC | 0.5587 |
| PR-AUC | 0.2073 |
| **F1 at threshold 0.5** | **0.0000** |
| Accuracy | 0.8444 |

**The model predicted no positive TAU clips at threshold 0.5.**
0.8444 accuracy = majority-class rate (459 negatives / 540).

Probability collapse on TAU:

| Statistic | Value |
|---|---:|
| Mean prob (negatives) | 0.0098 |
| Mean prob (positives) | 0.0202 |
| Median prob (all) | 0.000252 |

**Absolute probabilities are near-zero for both classes**, so this
is not a threshold problem — the ranking itself is near chance
(ROC-AUC 0.5587).

**Key findings:**

1. **Audio-only baseline is honest but weak.** Pooled MAVD ROC-AUC
   of 0.6237 is 1.29× the random PR-AUC baseline. The model learned
   *some* ranking signal but no useful classifier.

2. **Per-location transfer is bimodal and unexplained.** `magallanes0308`
   (0.703) and `rivera0923` (0.687) transfer; `acevedo0103` (0.481)
   and `templete0103` (0.507) do not. Two plausible explanations
   (class composition, day/night) were tested and refuted. The
   remaining hypothesis (per-location acoustic domain shift) is
   untested.

3. **The "only_offscreen" vs "both" gap is real but small.** 0.660
   vs 0.590 — 0.070 gap. Suggests the model is partly learning
   pure-offscreen signatures, but 0.660 is far from confident.

4. **TAU fails.** 0.000 F1 at threshold 0.5, ROC-AUC 0.5587.
   Genuine cross-dataset domain shift. Causes not verified:
   could be domain shift, class-prior shift (MAVD 25.6% → TAU 15.0%),
   or underfit + shift. Documented, not resolved.

5. **This is the audio-only half of a fusion experiment.** Its
   purpose was not to be a strong classifier but to be a fair
   baseline that fusion must beat. It is a fair baseline.

**Limitations:**

- From-scratch, 540 training clips. Pretrained audio backbones
  (AST, PaSST, BEATs) are future work.
- No augmentation (SpecAugment used in Week 7.6 was not applied).
- Bimodal location transfer. `rivera0922` (2 positives) contaminates
  any pooled F1 estimate.
- TAU domain shift. Not fixed.
- Per-location analysis is exploratory; the fold-level metrics were
  chosen first. No correction for multiple comparisons.

**Deliverables:**

- Model card: `docs/model_cards/urbansas_audio_cnn_v1.md`
- Dataset card: `docs/dataset_cards/urbansas.md`
- Training script: `scripts/train_urbansas_audio.py`
- Diagnostic script: `scripts/diagnose_urbansas_audio.py`
- Precompute cache: `scripts/precompute_urbansas_mels.py`
- Label semantics check: `scripts/check_urbansas_labels.py`
- Raw predictions: `data/urbansas_results/audio_only_oof.csv`,
  `audio_only_tau_external.csv` (gitignored)
- Checkpoint: `data/urbansas_results/audio_only_final.pt` (gitignored)

**Next:** video-only baseline using the same 5 LOLA folds, same
target definition, comparable metrics, TAU external test. Then
audio+video fusion comparison. TAU remains untouched for selection.

---

## 2026-10-27 — Week 9c-video: Urbansas offscreen-vehicle video classifier

**Hypothesis:** A Kinetics-400 pretrained R(2+1)D-18 can detect `offscreen`
vehicle events in 10-second urban video clips. The hypothesis is
**theoretically doubtful** — video cannot see offscreen vehicles — but a
pretrained visual backbone may pick up scene-context correlates that
correlate with the offscreen annotation policy. Measured, not assumed.

**Setup:**

- **Dataset**: Urbansas, 540 MAVD clips (training), 540 TAU clips (external).
- **Target**: `has_offscreen = "offscreen" in audio_annotations[filename]`.
  MAVD positive rate 25.6%, TAU 15.0%.
- **Protocol**: 5-fold leave-one-location-out on MAVD. Final model retrained
  on all MAVD, evaluated once on TAU at threshold 0.5.
- **Model**: R(2+1)D-18 (Kinetics-400), 31.3M params, head replaced with
  Linear(512, 2).
- **Input**: 16 uniformly-sampled frames at 112×112, Kinetics normalization.
  Frame cache precomputed to memmap (2.6 GB) for 6× training speedup.
- **Training**: AdamW, differential LR (backbone 0.3× head), cosine over 15
  epochs, batch 4, class-weighted CE with per-fold `n_neg/n_pos`.
- **Augmentation**: random crop pad=8, hflip 50%.

**Result:**

| Fold | Location | F1 | n_pos |
|---|---:|---:|---:|
| 1 | acevedo0103 | 0.5067 | 32 |
| 2 | magallanes0308 | 0.4466 | 48 |
| 3 | rivera0922 | 0.0000 | 2 (noise) |
| 4 | rivera0923 | 0.3333 | 28 |
| 5 | templete0103 | 0.5846 | 28 |

Mean F1: 0.3742 ± 0.2043
Pooled OOF: F1 0.4643, P 0.358, R 0.659, ROC-AUC 0.6130, PR-AUC 0.3063.

**TAU external**: F1 0.2362, P 0.326, R 0.185, ROC-AUC 0.6343.

**Key findings:**

1. **Video alone achieves F1 0.4643 on MAVD — higher than audio-only's
   0.3686.** But this is not because video "detects offscreen vehicles."
   It is because the video model learns scene-level correlations with the
   annotation policy. On `only_visible` clips (visible vehicle, no offscreen
   annotation), video's mean p is 0.80 — confident false positive — while
   audio correctly predicts near zero (mean p 0.08). On `empty` clips (no
   vehicles), video's mean p is 0.73 — also confident false positive. The
   model has learned "empty road → predict offscreen likely" as a shortcut.

2. **Video transfers to TAU much better than audio.** ROC-AUC 0.6343 vs
   audio-only 0.5587. Kinetics-400 pretraining provides a general visual
   representation that survives the Madrid → European cities shift.

3. **Per-location ROC-AUC is bimodal but at different locations than audio.**
   Video: 0.641 on `acevedo0103`, 0.501 on `magallanes0308`, 0.650 on
   `rivera0923`, 0.721 on `templete0103`. Audio's pattern was 0.481 /
   0.703 / 0.687 / 0.507. **The modalities are at chance on different
   locations** — this is the strongest evidence for complementarity.

4. **Per-subclass AUC is also complementary.** Video 0.746 on
   `only_offscreen`, 0.491 on `both`. Audio 0.660 / 0.590. Opposite pattern.

**Limitations:**

- The video model cannot detect offscreen vehicles. Its ROC-AUC reflects
  scene-context correlation, not detection.
- 55% of high-confidence false positives are on `only_visible` clips. The
  model has learned "visible vehicle → predict offscreen."
- `rivera0922` (2 positives) is noise.
- From-scratch on 540 clips with 31M pretrained parameters — overfitting
  to MAVD-specific visual patterns is likely.
- Per-location bimodality is unexplained.

**Deliverables:**

- Model card: `docs/model_cards/urbansas_video_cnn_v1.md`
- Training script: `scripts/train_urbansas_video.py`
- Precompute cache: `scripts/precompute_urbansas_video_frames.py`
- Raw OOF predictions: `data/urbansas_results/urbansas_video_oof.csv`
- TAU predictions: `data/urbansas_results/urbansas_video_tau_external.csv`
- MLflow: `urbanintel-week9b-urbansas`

**Next:** audio-video late fusion, using these two baselines.

---

## 2026-10-27 — Week 9c-fusion: Urbansas audio-video late fusion

**Hypothesis:** Combining audio and video at the probability level improves
over either single-modality baseline, and may transfer better to TAU. Given
that both baselines have ROC-AUC ~0.61 but fail on different locations and
different subclasses (see 9c-audio and 9c-video entries), the modalities may
provide complementary signals.

**Setup:**

- **Inputs**: OOF probabilities from the two baselines
  (`audio_only_oof.csv`, `urbansas_video_oof.csv`, aligned on `filename`,
  both 540 rows, `y_true` matches).
- **Protocol**: 5-fold leave-one-location-out on MAVD. For each held-out
  location: fit fusion head on the 4 non-held-out locations' OOF
  probabilities, evaluate on the held-out location. TAU external: fit
  fusion head on all 540 MAVD OOF probabilities, apply to TAU OOF probs.
- **Threshold**: fixed at 0.5. No tuning on MAVD test or TAU.
- **Six models**: `audio` (baseline), `video` (baseline), `mean`,
  `max`, `logistic`, `logistic_interaction`. Logistic heads use
  `LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced")`.
  Without class weighting, both logistic variants collapsed to F1 ≈ 0.09;
  the unweighted results are preserved as `fusion_results_unweighted.json`
  for the pre-fix record.
- **Bootstrap CIs**: 1000 paired resamples, at clip level (540 units) and
  location level (5 units).

**Result — MAVD pooled OOF:**

| Model | F1@0.5 | P | R | ROC-AUC |
|---|---:|---:|---:|---:|
| audio | 0.3686 | 0.348 | 0.391 | 0.6237 |
| video | 0.4643 | 0.358 | 0.659 | 0.6130 |
| mean | 0.4167 | 0.400 | 0.435 | **0.6704** |
| max | **0.4713** | 0.333 | **0.804** | 0.6176 |
| logistic (weighted) | 0.4480 | 0.326 | 0.717 | 0.6315 |
| logistic+I (weighted) | 0.4510 | 0.329 | 0.717 | 0.6263 |

**Result — TAU external:**

| Model | F1 | P | R |
|---|---:|---:|---:|
| audio | 0.0000 | 0.000 | 0.000 |
| video | **0.2362** | 0.326 | 0.185 |
| mean | 0.0000 | 0.000 | 0.000 |
| max | **0.2308** | 0.306 | 0.185 |
| logistic (weighted) | 0.0230 | 0.167 | 0.012 |
| logistic+I (weighted) | 0.0230 | 0.167 | 0.012 |

**Result — bootstrap CI (weighted logistic vs audio):**

- Clip-level: mean diff +0.0800, 95% CI [+0.0158, +0.1444], P ≈ 0.994
- Location-level: mean diff +0.0770, 95% CI [−0.0258, +0.1653], P ≈ 0.934

**Result — per-subclass F1:**

| Model | only_offscreen (66) | both (72) |
|---|---:|---:|
| audio | 0.5957 | 0.5306 |
| video | 0.9524 | 0.6019 |
| max | **0.9767** | **0.8000** |
| logistic | 0.9524 | 0.7027 |

(`only_visible` and `empty` subclasses have zero positive labels — F1 is
trivially 0 for all models, no information gained. Reported as 0 in the
model card with a note.)

**Key findings:**

1. **Zero-parameter max fusion is the best pooled MAVD F1 (0.4713) and best
   per-fold stability (mean 0.4493 ± 0.0663).** Its MAVD improvement over
   video-only is marginal (+0.007) — statistically indistinguishable — but
   it is the only fusion model that preserves TAU transfer (0.2308 vs
   video-only 0.2362).

2. **Weighted logistic fusion substantially improves MAVD F1 over audio
   (+0.08) with clip-level CI excluding zero, but location-level CI
   includes zero.** The two views disagree because with 5 locations, the
   location-level CI is the more honest estimate for generalization.

3. **Logistic fusion collapses on TAU (0.0230).** The in-domain MAVD gain
   does not transfer. Any production claim would need cross-domain
   evaluation.

4. **The subclass breakdown supports the complementary-signal hypothesis.**
   `only_offscreen` F1 goes from 0.5957 (audio) to 0.9524 (logistic) to
   0.9767 (max). `both` goes from 0.5306 to 0.7027 to 0.8000. Consistent
   with audio providing signal where video fails and vice versa. Not
   causal — the fusion head's training features come from base models
   that have seen the held-out location in *other* folds (see limitations).

5. **Mean fusion has the highest ROC-AUC (0.6704) but not the highest F1.**
   Threshold tuning on training data is a natural follow-up; not done here.

**Limitations:**

- **Leakage concern for logistic fusion heads.** The fusion head's training
  probabilities for fold `L` come from base models that were trained on
  location `L`'s data in other folds. This is a subtle leakage channel not
  eliminated by the 5-fold LOLA structure. A fully independent evaluation
  requires nested cross-validation (retraining base models 4× per outer
  fold); not done. **The logistic fusion results should be read with this
  caveat.** Zero-parameter `mean` and `max` are unaffected — they fit no
  parameters.
- **Max fusion's MAVD improvement over video-only is marginal** (+0.007
  pooled F1). Bootstrap CI not computed for this pair, but the point
  estimate is close to zero.
- **No threshold tuning on TAU.** Fixed at 0.5 before evaluation.
- **Two weak baselines, one weak external transfer.** Fusion helps MAVD
  in-domain but does not produce a generalizing classifier. The best TAU
  transfer (max 0.2308) roughly matches video-only, not exceeds it.
- **`rivera0922` fold has 2 positives.** Per-fold F1 is noise.

**Deliverables:**

- Model card: `docs/model_cards/urbansas_fusion_v1.md`
- Training script: `scripts/train_urbansas_fusion.py`
- Raw predictions:
  - `data/urbansas_results/fusion_oof.csv` (weighted logistic)
  - `data/urbansas_results/fusion_tau_external.csv`
  - `data/urbansas_results/fusion_results.json`
  - `data/urbansas_results/fusion_results_unweighted.json` (pre-fix)
- MLflow: `urbanintel-week9b-urbansas`

**Week 9c conclusion:** Audio-only and video-only baselines both achieve
ROC-AUC ~0.61 on MAVD but fail on different locations and different
subclasses. Late fusion on the two probabilities:
- Substantially improves in-domain MAVD F1 (weighted logistic +0.08, max
  fusion best pooled F1)
- **Does not transfer to TAU** (best transfer is max fusion at 0.2308,
  roughly matching video-only)
- Provides suggestive evidence for complementary signals on the two positive
  subclasses, but does not prove causal complementarity or cross-domain
  generalization.

**Future work:** feature-level fusion (frozen backbone embeddings +
concatenation + MLP head), nested cross-validation to resolve the leakage
concern, threshold tuning on training data for mean fusion, TAU-specific
calibration.

---

## 2026-10-27 — Week 9c 10a: Label audit and scene-metadata baseline

**Hypothesis:** The Week 9c fusion result depends on labels and evaluation
protocol that we have not independently verified. Two specific concerns:
(1) The `-1` annotation sentinel might not be a clean negative; (2) the video
model's apparent TAU transfer might be largely explained by city-level priors.

**Setup:**

- **Audit 1 — `-1` sentinel semantics:** Inspected all 130 `-1` rows in
  `audio_annotations.csv`. Cross-referenced with `non_identifiable_vehicle_sound`
  column and per-clip label composition.
- **Audit 2 — scene-metadata baseline:** Attempted a logistic regression on
  observable metadata (`night` flag, `city`) to test whether video's signal is
  largely metadata-driven. Built on MAVD and TAU separately.
- **Audit 3 — same-set comparison:** None. The city-only baseline on TAU used
  503 clips (7 cities ≥20 clips); the video model used 540 clips. **Not
  apples-to-apples.**

**Result — Audit 1 (`-1` sentinel):**

| Property | Value |
|---|---:|
| `-1` rows | 130 |
| Unique filenames | 130 (no clip has both `-1` and other labels) |
| `non_identifiable_vehicle_sound = 1` on `-1` rows | **74 (57%)** |
| `non_identifiable_vehicle_sound = 0` on `-1` rows | 56 (43%) |
| `start = end = -1` on all `-1` rows | Yes |

**Interpretation:** `-1` is not a clean negative. It is a placeholder for
clips where the annotator could not classify the audio. 57% of `-1` clips
contain vehicle-like sounds that were not classified. The `empty` subclass
(130 clips) is ambiguous, not confirmed negative.

**Result — Audit 2 (scene-metadata baseline):**

MAVD:
- 1 city (Montevideo), 5 locations. **City is constant on MAVD — no city
  baseline possible.**
- `night` flag: 30 clips at night (all in `rivera0922`), 510 at day. Only 2
  positives are in the night clips. **Near-constant — no night baseline
  possible on MAVD.**

TAU:
- 10 cities. 7 cities with ≥20 clips (503 clips total).
- **City-only logistic regression (5-fold stratified): mean F1 = 0.2982 ± 0.0525**
- Video-only TAU F1 (on all 540 clips): **0.2362**
- **The city-only baseline on TAU outperforms the video model on TAU.**
  Different clip sets, but the direction is clear.

**Interpretation:** A non-trivial fraction of the video model's TAU signal is
likely captured by city-level priors. The video model's contribution *beyond*
city identity is not established. This does not mean the video model is useless
— it means its TAU signal is not clearly beyond what a trivial city-classifier
achieves.

**Key findings:**

1. **The `-1` sentinel is not a clean negative.** 57% of `empty` clips carry
   `non_identifiable_vehicle_sound = 1`. The `empty` subclass is ambiguous.
   All Week 9c per-subclass diagnostics that used `empty` as a negative should
   be re-read with this caveat. The `only_visible` subclass (357 clips) is a
   clean negative and the `only_offscreen` + `both` subclasses (138 positive
   clips) are confirmed positives — the complementarity pattern between these
   three clean groups is not affected.

2. **The city-only baseline on TAU (F1 0.2982) beats the video model
   (F1 0.2362).** Video's TAU transfer may be substantially city-prior-driven.
   The two evaluated on different clip sets (503 vs 540); a strict comparison
   would evaluate the city baseline on the same 540 clips. But the point
   stands: video's TAU signal is not clearly beyond a trivial city classifier.

3. **MAVD cannot support a metadata baseline.** City is constant (Montevideo);
   `night` is near-constant. There is no observable metadata on MAVD that
   varies enough to build a fair baseline. This is a limitation of the dataset,
   not of the audit.

**Limitations of the audit:**

- The `-1` sentinel semantics were inferred from the annotation CSV. **They
  have not been verified against the Urbansas paper or annotation protocol
  documentation.** A reviewer with access to the original Urbansas release
  notes could confirm or refute our interpretation.
- The city baseline used 503 clips (7 cities ≥20 clips). The comparison to
  video's 540-clip F1 is directional, not strict.
- The `non_identifiable_vehicle_sound` column's semantics are also assumed —
  we interpret it as "annotator heard a vehicle-like sound that could not be
  classified," but it may mean something else in the original protocol.

**Deliverables:**

- Dataset card amendment: `docs/dataset_cards/urbansas.md` — new section
  "Annotation semantics — the `-1` sentinel"
- Video model card amendment: `docs/model_cards/urbansas_video_cnn_v1.md` —
  re-cast "55% high-confidence false positives" with the city-baseline finding
- Fusion model card amendment: `docs/model_cards/urbansas_fusion_v1.md` —
  add the negative-pool-pollution caveat

**Next:** Nested cross-validation for the weighted logistic fusion head.
This is the remaining methodological concern from Week 9c: the +0.08 MAVD
improvement is subject to an outer-fold leakage channel that requires
retraining the base models per outer fold. Nested CV resolves it at ~20 hours
of compute.

---

## 2026-10-27 — Week 9c: Second nested-CV pilot (magallanes0308)

**Setup:** Same pilot design as the `acevedo0103` pilot, but with
`magallanes0308` as the outer fold and a different inner split:

- Outer: `magallanes0308` (175 clips, 48 positives)
- Inner train: `templete0103` + `rivera0923` (235 clips, 56 positives)
- Inner test: `rivera0922` + `acevedo0103` (130 clips, 34 positives)

**Configuration:** 15 epochs, `num_workers=4`, batch 16 (audio) / 4 (video).
This is the same configuration as the first pilot and the full baselines.
The 10-epoch / `num_workers=0` optimization discussed earlier was not applied
in the actual run.

**Runtime (from console output):** audio 1,031s (17.2 min), video 350s
(5.8 min), total pilot ~23 min.

**Result:**

| Model on `magallanes0308` | F1 | P | R |
|---|---:|---:|---:|
| audio-only | 0.2154 | 0.412 | 0.146 |
| video-only | 0.4466 | 0.291 | 0.958 |
| logistic (standard OOF) | 0.4402 | 0.286 | 0.958 |
| logistic (nested, leak-free) | 0.4279 | 0.275 | 0.958 |

**Delta (nested − standard) = −0.0123.**

**Combined with pilot 1 (`acevedo0103`, delta +0.0237):**

| Pilot | Outer fold | Delta |
|---|---|---:|
| 1 | `acevedo0103` | +0.0237 |
| 2 | `magallanes0308` | −0.0123 |
| Average | | ≈ +0.0057 |

**Interpretation.** Two diagnostic folds with different inner splits show
deltas within ±0.024 of zero. **This provides limited evidence against a
large, consistently positive effect from the suspected leakage channel. It
does not establish that leakage is negligible or rule it out as a contributor
to the pooled +0.08 result.**

The standard head had ~2–3× the training data in both pilots (440 vs 265,
365 vs 130). If leakage were dominant, we would expect the standard head to
show a clear advantage on at least one fold. It does not.

**Caveats:**
- Two folds with different inner splits. Not full nested CV.
- Training-set size and location distribution are confounded with leakage.
  A near-zero delta is consistent with "small leakage" OR "leakage offset by
  training-size effects" OR "leakage fold-dependent."
- Full nested CV across all 5 outer folds with 4 inner splits each is deferred.
  **Revised cost estimate: 12–18h** on this hardware (15-epoch basis).
- **The +0.08 remains "promising, not proven."** The pilot is diagnostic,
  not validation.

**Methodology note:** the pilot scripts are preserved at
`scripts/diagnose_urbansas_nested_cv_acevedo.py` and
`scripts/diagnose_urbansas_nested_cv_magallanes.py`. They record the exact
configurations used and are not parameterized for arbitrary folds (re-running
on a different fold requires editing the config block). They are diagnostic
scripts, not part of the standard reproducibility pipeline.

**Deliverables:**
- Model card limitation #1 amended: `docs/model_cards/urbansas_fusion_v1.md`
- This entry: `docs/experiments.md`
- Diagnostic scripts renamed: `scripts/diagnose_urbansas_nested_cv_*.py`

**Week 9c methodological state:**
- ✅ Label semantics: audited; `-1` is ambiguous, documented.
- ✅ Video's TAU signal: measured to be partly city-level, documented.
- ⚠️ Logistic fusion leakage: two-fold pilot provides limited evidence
  against a large leakage effect, but does not resolve the concern. Full
  nested CV deferred (12–18h).

Full nested CV remains a documented future-work item.