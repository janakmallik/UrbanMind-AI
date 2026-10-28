# Model Card: Urbansas Offscreen-Vehicle Audio Classifier (1D-CNN v1)

## What this model does

Given a 10-second Urbansas clip's audio (48 kHz mono, log-mel
spectrogram), predicts a probability that the clip's audio annotations
include the `offscreen` vehicle label.

The `offscreen` label marks a vehicle sound whose source is not visible
in the synchronized video. **Important caveat:** `offscreen` coexists
with visible-vehicle labels (`car`, `bus`, `motorbike`, `truck`) in
52% of MAVD positives and 68% of TAU positives. This is not a simple
"visible vs invisible vehicle" task — it is "does the clip's audio
annotation set include an offscreen vehicle event, possibly alongside
visible-vehicle events."

This is the audio half of the Week 9c cross-modal fusion experiment.
Its purpose is to establish an honest audio-only baseline before
testing whether audio+video fusion improves on it.

## What this model does NOT do

- **Not a vehicle-presence detector.** A clip with a visible car and
  no offscreen annotation is a *negative* under this task, not a
  positive. The model does not predict "is there a vehicle."
- **Not a vehicle-localization model.** It does not say *where* the
  offscreen vehicle is or how many there are.
- **Not a cross-location generalizer.** Per-location ROC-AUC varies
  from 0.481 to 0.703 on MAVD. Two of five MAVD locations are at
  chance. The model learned acoustic signatures that do not transfer
  uniformly across Madrid street scenes.
- **Not a cross-dataset generalizer.** TAU (Helsinki, Lisbon, Lyon,
  Milan, Paris, Prague, Stockholm, Vienna, Barcelona, London) has
  ROC-AUC 0.5587 and 0.000 F1 at threshold 0.5. Genuine domain shift.
- **Not evaluated at deployment conditions.** All evaluation is on
  annotated clips from the Urbansas corpus. Real microphone feeds,
  varying bitrates, and overlapping urban sources are untested.
- **Not the same model as Week 7.6.** The architecture is identical
  (4-block 1D-CNN over log-mel), but it was trained from scratch on
  540 Urbansas clips, not on UrbanSound8K. It has no pretrained
  weights.

## Architecture

- **Type:** 4-block 1D convolutional network over log-mel spectrograms
  (same as Week 7.6's `AudioCNN`)
- **Params:** ~348,000
- **Input:** (1, 128, 234) — 1 channel, 128 mel bins, ~234 frames
  (10.0 s @ 48 kHz, n_fft=2048, hop_length=2048)
- **Structure:**
  ```
  Conv1d(128→64,  k=3) + BatchNorm + ReLU + MaxPool1d(2)
  Conv1d(64→128,  k=3) + BatchNorm + ReLU + MaxPool1d(2)
  Conv1d(128→256, k=3) + BatchNorm + ReLU + MaxPool1d(2)
  Conv1d(256→256, k=3) + BatchNorm + ReLU + AdaptiveAvgPool1d(1)
  Dropout(0.5)
  Linear(256→2)
  ```
- **Output:** 2-class logits (class 0 = no offscreen, class 1 = offscreen)

## Training

- **Dataset:** Urbansas audio, 1,080 clips (540 MAVD + 540 TAU).
  Training uses MAVD only; TAU is held out as external test.
- **Target:** `has_offscreen` = `"offscreen" in set(audio_annotations[filename])`
- **MAVD class balance:** 138 positives / 540 clips (25.6%)
- **Protocol:** 5-fold **leave-one-location-out** on MAVD. Each fold
  holds out one entire Madrid location; no scene-level leakage.
  Fold order (per `scripts/train_urbansas_audio.py` line 207,
  alphabetically): `acevedo0103`, `magallanes0308`, `rivera0922`,
  `rivera0923`, `templete0103`.
- **Final model:** retrained on all 540 MAVD clips after CV, then
  evaluated once on TAU. Threshold fixed at 0.5 **before** TAU
  evaluation. No tuning on TAU.
- **Input preprocessing:**
  - Source: 48 kHz mono (downmixed from stereo if needed)
  - 128-mel log-spectrogram, n_fft=2048, hop=2048, fmin=20, fmax=24000
  - Per-sample normalization: `(mel - mel.mean()) / (mel.std() + 1e-6)`
  - Cache: `data/urbansas_mels/mels.npy` (~130 MB)
- **Loss:** class-weighted cross-entropy
  - Per-fold weight: `n_neg / n_pos` for that fold's training labels
  - Final model weight: 402/138 ≈ 2.91
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** cosine annealing over 15 epochs
- **Batch size:** 16
- **Epochs:** 15 (fixed, no early stopping, no validation set —
  the loop trains for 15 epochs and evaluates the final-epoch model.
  This differs from Week 7.6, which used best-epoch selection on a
  validation split with patience=3.)
- **Precision:** fp32
- **Random seed:** 42 + fold_idx (per fold), 142 (final model)
- **No augmentation.** SpecAugment (used in Week 7.6) was not
  applied. This is a documented difference from the UrbanSound8K
  baseline, not an oversight.

## Results — MAVD leave-one-location-out CV

| Fold | Location | F1 | Precision | Recall | Positives |
|---:|---|---:|---:|---:|---:|
| 1 | `acevedo0103` | 0.4138 | 0.3273 | 0.5625 | 32 |
| 2 | `magallanes0308` | 0.2154 | 0.4118 | 0.1458 | 48 |
| 3 | `rivera0922` | 0.5000 | 0.5000 | 0.5000 | 2 |
| 4 | `rivera0923` | 0.4483 | 0.4333 | 0.4643 | 28 |
| 5 | `templete0103` | 0.3797 | 0.2941 | 0.5357 | 28 |

**Mean F1 (unweighted):** 0.3914 ± 0.0966
**Pooled OOF F1:** 0.3686
**Pooled OOF precision:** 0.3484
**Pooled OOF recall:** 0.3913

**`rivera0922` (2 positives) is not a reliable F1 estimate.** Its
0.5000 is one true-positive out of two. Report it as noise, not
signal.

### Pooled OOF ROC-AUC and PR-AUC

| Metric | Value |
|---|---:|
| ROC-AUC | 0.6237 |
| PR-AUC (average precision) | 0.3296 |
| Random baseline PR-AUC (positive rate) | 0.256 |

**Interpretation:** the model ranks MAVD positives above negatives
better than chance (0.6237 vs 0.5) but well below useful. PR-AUC of
0.3296 is 1.29× the random baseline (0.256), not a confident
classifier.

### Per-location ROC-AUC (bimodal)

| Location | n | Pos | ROC-AUC | Note |
|---|---:|---:|---:|---|
| `acevedo0103` | 100 | 32 | 0.481 | chance |
| `magallanes0308` | 175 | 48 | 0.703 | useful |
| `rivera0922` | 30 | 2 | 0.500 | n=2, noise |
| `rivera0923` | 145 | 28 | 0.687 | useful |
| `templete0103` | 90 | 28 | 0.507 | chance |

**Two locations transfer, two do not.** The cause was not identified
in this experiment. Two hypotheses were tested and refuted:
1. **Class composition** — the "both" subclass fraction is 12–20%
   across all four non-noise locations, with no correlation to
   ROC-AUC. `rivera0923` has the highest "both" fraction (15%) and
   scores 0.687. Refuted.
2. **Night vs. day** — only `rivera0922` contains night clips, and
   `rivera0922` is not one of the meaningful folds. All four real
   folds are 100% daytime. Refuted.

The working hypothesis is **acoustic domain shift between Madrid
locations** (microphone geometry, street reflections, background
noise floor) — untested here because it requires per-clip acoustic
analysis beyond the current scope.

### Per-subclass ROC-AUC (vs negative pool of `only_visible` + `empty`)

| Positive subclass | n | ROC-AUC vs negatives |
|---|---:|---:|
| `only_offscreen` | 66 | 0.660 |
| `both` (offscreen + visible) | 72 | 0.590 |

**Gap of 0.070** in the predicted direction: pure-offscreen clips
rank higher than mixed clips. But **0.660 for the "easiest" subgroup
is still weak** — the model is not confidently detecting even pure
offscreen events.

## Results — TAU external test

Trained on all 540 MAVD clips. Evaluated **once** on all 540 TAU
clips. Threshold 0.5 fixed before evaluation.

| Metric | Value |
|---|---:|
| Positives | 81 / 540 (15.0%) |
| ROC-AUC | 0.5587 |
| PR-AUC | 0.2073 |
| Accuracy (threshold 0.5) | 0.8444 |
| **F1 (threshold 0.5)** | **0.0000** |
| **Precision (threshold 0.5)** | **0.0000** |
| **Recall (threshold 0.5)** | **0.0000** |

**At threshold 0.5 the model predicted no positive TAU clips.** The
0.8444 accuracy is the majority-class rate (459 negatives / 540).
The model has effectively collapsed on TAU.

### Where the probabilities went

| Statistic | Value |
|---|---:|
| Mean prob (negatives) | 0.0098 |
| Mean prob (positives) | 0.0202 |
| Median prob (all) | 0.000252 |
| 75th percentile | 0.00131 |
| Max | 0.7844 |

**Absolute probabilities collapsed to ~0.01.** This is not a
threshold-calibration problem — the ranking is also near chance
(ROC-AUC 0.5587). The model is producing near-zero for both classes
on TAU.

### Possible causes (not verified)

- **Domain shift.** TAU clips are from Barcelona, Helsinki, Lisbon,
  Lyon, London, Milan, Paris, Prague, Stockholm, Vienna — different
  microphones, different street acoustics from Madrid's MAVD.
- **Class-prior shift.** TAU offscreen prevalence (15.0%) is 59%
  of MAVD's (25.6%). A model trained at the MAVD prior will
  under-predict on TAU's lower base rate.
- **Label-definition shift.** Event duration medians are compatible
  (3.85s MAVD vs 3.21s TAU), and events-per-clip is virtually
  identical (1.09 vs 1.05). Label semantics appear compatible, but
  this is not proven by duration statistics alone.
- **Underfit + domain shift.** 540 clips is small for a from-scratch
  audio event classifier. The MAVD-only ranking was already weak
  (ROC-AUC 0.6237); TAU would be weak even without shift.

**Do not claim any of these as the cause without further evidence.**
The honest statement is: "the audio-only model failed to transfer
to TAU; the failure is real and reproducible; the mechanism is
unidentified."

## Known limitations

1. **From-scratch, small-data.** 540 training clips is small for
   audio event detection. Pretrained audio backbones (AST, PaSST,
   BEATs) are the standard remedy. Out of scope for this experiment.
2. **No augmentation.** SpecAugment (used in Week 7.6) was omitted.
   Adding it is a plausible +3-5 ROC-AUC experiment, untested.
3. **Bimodal location transfer.** Two of four MAVD folds are at
   chance. The cause is unidentified (see above). Reporting a single
   mean ± std hides this.
4. **TAU domain shift.** Genuine failure on external test. Not
   fixed; documented.
5. **Label overlap.** 52% of MAVD positives and 68% of TAU positives
   also contain visible-vehicle annotations. The model does not
   distinguish "only offscreen" from "both"; per-subclass AUC shows
   a small but real gap.
6. **`rivera0922` is 2 positives.** Its 0.500 F1 is noise.
7. **10-second fixed window.** No temporal cropping; the model sees
   the whole clip and pools globally.
8. **No per-class F1.** Binary task, so precision / recall / F1 at
   thresholds are reported instead.

## What this model demonstrates

- Audio ML pipeline on a *paired* audio+video dataset
- 5-fold leave-one-location-out cross-validation with scene
  leakage prevention
- Per-fold `pos_weight` from training-fold labels (not global)
- Threshold sweep analysis on both MAVD and TAU
- Per-location and per-subclass diagnostic decomposition
- Honest negative results: bimodal location transfer, TAU collapse
- Reproducible diagnostic script
  (`scripts/diagnose_urbansas_audio.py`)

## What this model is NOT a demonstration of

- Cross-modal fusion (this is audio-only)
- **Pretrained audio backbones — see `urbansas_audio_pretrained_v1.md`
  for an AST-based baseline with substantially higher MAVD F1 (0.4673
  vs 0.3686, +27% relative)**
- Audio-visual correspondence
- Generalizable offscreen-vehicle detection
- Deployment-quality performance

## Reproduce

```powershell
# one-time, ~5 min on a modern CPU
uv run python scripts/precompute_urbansas_mels.py

# 5-fold CV + final MAVD + TAU, ~20 min on RTX 4050
uv run python scripts/train_urbansas_audio.py --epochs 15 --batch-size 16 --num-workers 4

# diagnostics (read-only, writes nothing)
uv run python scripts/diagnose_urbansas_audio.py

## Citation

% TODO: Replace with exact citation from CITATIONS.md before final portfolio polish.

```bibtex
@inproceedings{urbansas_2022,
  title  = {The Urbansas Dataset: Urban Scene Understanding from
            Paired Audio and Video},
  author = {...},
  year   = {2022},
  note   = {See https://zenodo.org/records/6658386 for current citation}
}
```