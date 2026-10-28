# Model Card: Urbansas Offscreen-Vehicle Audio Classifier (AST Pretrained v1)

## What this model does

Given a 10-second Urbansas clip's audio (16 kHz mono), predicts a
probability that the clip's audio annotations include the `offscreen`
vehicle label.

This is the **pretrained audio** variant of the Week 9c audio baseline.
It replaces the from-scratch 1D-CNN with a **frozen AST encoder**
(Audio Spectrogram Transformer, AudioSet-pretrained) + a linear head.
Only 1,538 parameters (the linear head) are trained.

The purpose is to answer a specific question: **is the from-scratch
audio model underpowered, or is audio genuinely less informative for
this task?** The answer, measured: pretraining substantially improves
MAVD in-domain performance but does not improve cross-city TAU ranking.

**The same task caveats apply as the from-scratch baseline.** The
`offscreen` label coexists with visible-vehicle labels in 52% of MAVD
positives. This is not a simple "visible vs invisible vehicle"
detection task. See the dataset card for details.

## What this model does NOT do

- **Not the best audio model for this task.** A further-fine-tuned
  AST or a task-specific architecture might do better. Only the frozen
  encoder + linear head was tried.
- **Does not transfer cross-city in ranking.** AST's TAU ROC-AUC
  (0.5572) is essentially identical to the from-scratch model's
  (0.5587). The apparent TAU F1 improvement (0.0000 → 0.2754) is a
  threshold artifact: AST's probability distribution shape hits the
  fixed 0.5 threshold at a more favorable point, but the *ranking* on
  TAU is the same as from-scratch.
- **Not evaluated at deployment conditions.** All evaluation is on
  annotated Urbansas clips.
- **Not calibrated for TAU.** AST over-predicts positive on TAU
  (precision 0.182 at threshold 0.5). Threshold calibration requires
  labelled target-city data we don't have.

## Architecture

- **Type:** Frozen AST encoder + linear classification head
- **Encoder:** `MIT/ast-finetuned-audioset-10-10-0.4593` (AudioSet,
  527 classes including vehicle categories)
- **Frozen params:** 86,187,264 (encoder)
- **Trainable params:** 1,538 (linear head: 768 × 2 + 2 bias)
- **Input:** Precomputed AST features (1024 time frames × 128 mel bins)
  from `data/urbansas_ast_features/ast_features.npy`
- **Feature extraction:** `ASTFeatureExtractor` applied once to 16 kHz
  audio, cached (566 MB on disk)
- **Forward pass:** encoder wrapped in `torch.no_grad()`, mean-pooled
  over 1214 token positions to 768-dim vector, then linear head
- **Output:** 2-class logits

## Training

- **Data:** 540 MAVD clips (5 locations), 540 TAU clips (external test)
- **Protocol:** 5-fold leave-one-location-out on MAVD. Final model
  retrained on all 540 MAVD, evaluated once on TAU at threshold 0.5
- **Target:** `has_offscreen = "offscreen" in audio_annotations[filename]`
- **Class weight:** per-fold `n_neg / n_pos` for CrossEntropyLoss
  (2.59–3.15 across folds; 2.91 for final model)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4) on the head only
- **Scheduler:** CosineAnnealingLR over 10 epochs
- **Batch size:** 8
- **Epochs:** 10
- **Precision:** fp32
- **Random seed:** 42 + fold_idx (per fold), 142 (final)
- **No augmentation.** Frozen encoder means no augmentation needed;
  the encoder doesn't update.

## Results — MAVD LOLA CV

| Fold | Location | F1 | P | R | n_pos |
|---|---:|---:|---:|---:|---:|
| 1 | `acevedo0103` | 0.4103 | 0.348 | 0.500 | 32 |
| 2 | `magallanes0308` | 0.4486 | 0.407 | 0.500 | 48 |
| 3 | `rivera0922` | 0.0000 | 0.000 | 0.000 | 2 (noise) |
| 4 | `rivera0923` | 0.4815 | 0.500 | 0.464 | 28 |
| 5 | `templete0103` | 0.5500 | 0.423 | 0.786 | 28 |

**Mean per-fold F1:** 0.3781 ± 0.1945
**Pooled OOF:** F1 0.4673, P 0.410, R 0.543, ROC-AUC 0.6801, AP 0.3933

### Comparison to from-scratch audio baseline

| Metric | From-scratch | **AST pretrained** | Δ | Relative |
|---|---:|---:|---:|---:|
| MAVD pooled F1 | 0.3686 | **0.4673** | **+0.0987** | **+27%** |
| MAVD ROC-AUC | 0.6237 | **0.6801** | **+0.0564** | **+9%** |
| MAVD AP | 0.3296 | **0.3933** | **+0.0637** | **+19%** |
| TAU F1 | 0.0000 | 0.2754 | +0.2754 | — |

**AST substantially improves in-domain MAVD performance.**

### Per-location ROC-AUC vs from-scratch

| Location | From-scratch | AST | Change |
|---|---:|---:|---|
| `acevedo0103` | 0.481 | **0.496** | +0.015 |
| `magallanes0308` | **0.703** | 0.658 | **−0.045** |
| `rivera0922` | 0.500 | 0.929 (n=2, noise) | — |
| `rivera0923` | 0.687 | **0.773** | **+0.086** |
| `templete0103` | 0.507 | **0.672** | **+0.165** |

**AST improves 3 of 4 meaningful folds but regresses on
`magallanes0308`** (0.703 → 0.658). The `magallanes0308` regression is
small but real. The largest improvement is on `templete0103`
(+0.165 ROC-AUC), where from-scratch was at chance.

### Per-subclass ROC-AUC vs negative pool

| Subclass | n | From-scratch | **AST** |
|---|---:|---:|---:|
| `only_offscreen` | 66 | 0.660 | **0.755** |
| `both` | 72 | 0.590 | 0.612 |

**AST improves on both subclasses**, most on `only_offscreen`
(+0.095).

## Results — TAU external

| Metric | Value |
|---|---:|
| Positives | 81 / 540 (15.0%) |
| **ROC-AUC** | **0.5572** |
| AP | 0.1798 |
| **F1 @ 0.5** | 0.2754 |
| Precision @ 0.5 | 0.182 |
| Recall @ 0.5 | 0.568 |
| Accuracy @ 0.5 | 0.552 |

### Critical finding: TAU ROC-AUC is not better than from-scratch

| Metric | From-scratch | AST |
|---|---:|---:|
| TAU ROC-AUC | 0.5587 | **0.5572** |
| TAU AP | 0.2073 | 0.1798 |

**AST's TAU ROC-AUC (0.5572) is essentially identical to
from-scratch (0.5587).** The F1 improvement (0.0000 → 0.2754) is a
threshold artifact: AST's probability distribution has mean ~0.47 for
both classes on TAU, so the 0.5 threshold lands at a place where F1 is
non-zero. From-scratch's TAU probabilities collapsed near zero, so
threshold 0.5 predicted all-negative (F1 = 0). **Neither model ranks
TAU clips meaningfully better than the other.**

### Threshold sensitivity

| Threshold | MAVD F1 | TAU F1 |
|---|---:|---:|
| 0.10 | 0.408 | 0.261 |
| 0.20 | 0.438 | 0.271 |
| **0.30** | **0.478** | 0.251 |
| 0.40 | 0.473 | 0.256 |
| 0.50 | 0.467 | **0.275** |

**AST's best MAVD F1 is at threshold 0.30 (0.478), not 0.50.** The
fixed threshold of 0.50 used for the primary protocol understates
AST's MAVD performance by ~0.01. Conversely, TAU's best F1 is at 0.50
(0.275), not at lower thresholds. **The two datasets prefer different
thresholds, which is why a single fixed threshold produces the
"threshold artifact" in the TAU comparison.**

## Known limitations

1. **Pretraining ≠ fair architecture comparison.** AST has 86M frozen
   pretrained parameters; from-scratch has ~350K trained from random
   init. This is not a comparison of architectures — it is a comparison
   of pretrained vs from-scratch. The finding ("pretraining helps")
   is expected and confirmed; the finding is *how much* it helps on
   this specific task.

2. **TAU F1 improvement is a threshold artifact.** TAU ROC-AUC is
   identical to from-scratch. The TAU F1 gain (0.0000 → 0.2754) is
   entirely due to AST's different probability distribution shape
   hitting threshold 0.5 differently. **Do not claim AST improves
   cross-city generalization.** It does not, in a ranking sense.

3. **`magallanes0308` regression.** AST's ROC-AUC on this fold
   (0.658) is worse than from-scratch (0.703). Cause not identified.
   Possibly AudioSet pretraining happens to misalign with this
   location's acoustic signature.

4. **Frozen encoder only.** Full fine-tuning of AST was not attempted.
   It would likely overfit on 540 clips but is a natural next
   experiment if more data becomes available.

5. **Per-fold F1 std is very high (±0.19).** This is driven by
   `rivera0922` (n_pos=2, F1=0) and the general location variability.
   The mean per-fold F1 is not a good summary of this model's
   performance; pooled OOF F1 (0.4673) is the honest aggregate.

6. **Not calibrated for TAU.** AST over-predicts positive on TAU
   (precision 0.182). Calibration would require labelled target-city
   data.

7. **Not the model to use for fusion.** With AST as the audio input,
   late fusion with video does not improve over AST alone on MAVD
   (Δ ≈ 0, CI includes zero). See the fusion model card's sensitivity
   section.

8. **`rivera0922` fold is noise.** 2 positives, 1 missed. F1=0 is
   uninformative.

## What this model demonstrates

- Frozen pretrained audio transfer learning on a small dataset
  (540 clips)
- Caching AST features to disk for tractable training (~10 min
  end-to-end instead of ~35 min on-the-fly)
- Same 5-fold LOLA protocol as the from-scratch baseline
- Honest reporting of a threshold artifact as an artifact (not
  overclaiming TAU generalization)
- Threshold sensitivity analysis showing the model's best F1 is not
  at the fixed 0.5 threshold

## What this model is NOT a demonstration of

- Cross-city generalization in ranking
- Fair architecture comparison (AST vs from-scratch is pretrained
  vs not-pretrained)
- Best-possible audio performance on this task
- Fusion value when audio is strong

## Reproduce

```powershell
# 1. 16 kHz audio cache (once, ~3 min)
uv run python scripts\precompute_urbansas_audio16k.py

# 2. AST feature cache (once, ~3 min)
uv run python scripts\precompute_urbansas_ast_features.py

# 3. Train the model (~10 min)
uv run python scripts\train_urbansas_audio_pretrained.py --epochs 10 --batch-size 8 --num-workers 4
```

## MLflow

- Experiment: `urbanintel-week9b-urbansas`

## Citation

% TODO: Replace with exact citations from CITATIONS.md before final portfolio polish.

```bibtex
@inproceedings{gong2021ast,
  title={AST: Audio Spectrogram Transformer},
  author={Gong, Yuan and Chung, Yu-An and Glass, James},
  booktitle={Interspeech},
  year={2021}
}

@inproceedings{urbansas_2022,
  title  = {The Urbansas Dataset: Urban Scene Understanding from Paired Audio and Video},
  author = {...},
  year   = {2022},
  note   = {See https://zenodo.org/records/6658386}
}
```