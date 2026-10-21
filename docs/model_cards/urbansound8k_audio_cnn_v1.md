# Model Card: UrbanSound8K 10-Class Audio Event Classifier (1D-CNN v1)

## What this model does

Classifies a 4-second urban audio clip into one of 10 sound classes:

air_conditioner, car_horn, children_playing, dog_bark, drilling,
engine_idling, gun_shot, jackhammer, siren, street_music.

Designed as the audio-input modality for UrbanMind-AI. Two classes
(`gun_shot`, `siren`) are emergency-relevant; the other eight are
urban background / context classes.

## What this model does NOT do

- **Not a complete emergency audio detector.** UrbanSound8K does not
  include fighting, screaming, crowd noise, or spoken distress signals.
  Detecting "call the police" requires a separate speech model (see
  Week 7.6b — Whisper keywords, conditional).
- **Not trained on continuous audio.** Trained on 4-second clips; behavior
  on long-form streams is untested.
- **Not robust to overlapped audio.** UrbanSound8K is single-label;
  real urban audio frequently has siren + traffic + construction
  simultaneously.
- **Not evaluated on live/deployed audio.** Only on the predefined
  UrbanSound8K 10-fold splits.
- **Not a threat-classification system.** `gun_shot` is acoustically
  confused with fireworks, backfires, and slammed doors — the model
  cannot distinguish these.

## Architecture

- **Type:** 4-block 1D convolutional network over log-mel spectrograms
- **Params:** ~348,000
- **Input:** (1, 128, 173) — 1 channel, 128 mel bins, 173 frames
  (4.0 s @ 22050 Hz, hop=512)
- **Structure:**
  ```
  Conv1d(128→64,  k=3) + BatchNorm + ReLU + MaxPool1d(2)
  Conv1d(64→128,  k=3) + BatchNorm + ReLU + MaxPool1d(2)
  Conv1d(128→256, k=3) + BatchNorm + ReLU + MaxPool1d(2)
  Conv1d(256→256, k=3) + BatchNorm + ReLU + AdaptiveAvgPool1d(1)
  Dropout(0.5)
  Linear(256→10)
  ```
- **Output:** 10-class logits

## Training

- **Dataset:** UrbanSound8K, 8,732 clips, 10 classes
- **Protocol:** 10-fold cross-validation using the **predefined folds**
  (dataset-recommended; NOT a random split)
- **Per-fold split:** ~6,700 train / ~1,180 val / ~873 test
  (val carved from the 9-fold train pool at 15%, stratified)
- **Input preprocessing:**
  - Source: 44.1 kHz stereo → 22.05 kHz mono
  - 128-mel log-spectrogram, n_fft=1024, hop=512, fmin=20, fmax=8000
  - Per-sample normalization: `(mel - mel.mean()) / mel.std()`
- **Augmentation (train only):** SpecAugment — 50% time-mask (width 1-17),
  30% frequency-mask (width 1-12), filled with `mel.min()`
- **Loss:** class-weighted cross-entropy
  - Weights computed from global class counts:
    `w_i = total / (10 × count_i)`
  - Effective weights: `gun_shot` ≈ 2.34, `car_horn` ≈ 2.04, others ≈ 0.88-1.0
  - **Approximation:** folds are session-stratified, not class-stratified
    (see dataset card), so per-fold weights would differ from these global
    values by up to ~3.5× for `car_horn`. Global weights were used for
    simplicity; a per-fold reweighting ablation is natural future work.
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** cosine annealing over 15 epochs
- **Batch size:** 32
- **Epochs:** up to 15 per fold
- **Early stopping:** patience=3, min_delta=0.005 on val macro-F1
- **Checkpoint selection:** best validation macro-F1
- **Precision:** fp32 (no mixed precision)
- **Data loading:** memmap-backed `CachedAudioDataset`,
  4 workers, `pin_memory=True`

## Results (10-fold CV)

| Metric | Mean ± Std |
|---|---:|
| **Test accuracy** | **0.6779 ± 0.0497** |
| **Test macro-F1** | **0.6878 ± 0.0454** |

**Per-fold:**

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
| Majority class (air_conditioner, 11.4%) | ~0.114 |

**Per-class F1 (test, mean across folds):** not computed in this run.
A confusion matrix analysis is a natural Week 8 follow-up — `gun_shot`
is the class most likely to lag and the one most relevant to the
emergency use case.

## Known limitations

1. **Val F1 overestimates test F1 by ~0.21.** Best-epoch selection
   overfits to the per-fold validation set, which shares 9-fold
   provenance with training. Val F1 averages ~0.90, test F1 averages
   ~0.69. This is documented, not hidden.
2. **Per-sample normalization discards amplitude information.** The model
   cannot learn "gun_shot is louder than air_conditioner" because that
   absolute information is normalized away. A dataset-wide per-mel-bin
   normalization ablation is a natural next step.
3. **Class imbalance persists in test.** `gun_shot` (374 training clips)
   is the smallest class and the one most relevant to the emergency use
   case. Per-class F1 on `gun_shot` is expected to lag — this is the
   central limitation of the model for its intended purpose.
4. **Fold-to-fold variance is large (±0.05).** Folds are session-
   stratified, not class-stratified; minority-class counts per fold
   vary by up to 3.5×. The mean ± std is the honest summary; a single
   split number would be misleading.
5. **No overlapping-audio handling.** Each clip is single-label; the
   model cannot say "this clip contains both siren and traffic".
6. **Untested on live audio.** All evaluation is on the predefined
   UrbanSound8K folds; performance on real-world streams is unknown.
7. **Small model for a hard task.** ~348K parameters for a 10-class
   problem with a class-imbalanced, acoustically-overlapping input
   distribution. Larger architectures (AST, PaSST) exist but were out
   of scope for Week 7.6.
8. **Global class weights approximate per-fold weights.** Because folds
   are session-stratified, not class-stratified, the minority-class
   composition varies by up to ~3.5× between folds. Global class weights
   (used here) do not track this. A per-fold reweighting ablation is a
   natural Week 8+ extension.

## What this model demonstrates

- Audio ML pipeline: mel-spectrogram preprocessing, 1D convolutions
- Memmap-backed cache for 3× training speedup vs. re-decoding
- 10-fold cross-validation on predefined folds (dataset protocol)
- Class-weighted loss for imbalanced urban sound taxonomy
- SpecAugment for audio augmentation
- Best-epoch checkpoint selection with early stopping
- Honest reporting of the val/test gap and its cause

## What this model is NOT a demonstration of

- Complete urban audio understanding
- Emergency detection at production quality
- Real-time inference on continuous streams
- Robustness to acoustic confusions (fireworks vs. gunshots)

## Reproduce

```powershell
# one-time, ~10-15 min on a modern CPU
uv run python scripts/precompute_audio.py

# 10-fold CV, ~30-40 min per fold on RTX 4050
uv run python scripts/train_audio_events.py --epochs 15 --batch-size 32 --num-workers 4

# smoke test on a single fold
uv run python scripts/train_audio_events.py --single-fold 1 --epochs 15
```

## Citation

```bibtex
@inproceedings{salamon2014urbansound,
  title={A Dataset and Taxonomy for Urban Sound Research},
  author={Salamon, Justin and Jacoby, Christopher and Bello, Juan Pablo},
  booktitle={Proceedings of the 22nd ACM International Conference on Multimedia},
  pages={1041--1044},
  year={2014}
}
```