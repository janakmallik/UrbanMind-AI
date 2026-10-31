# Model Card: Urbansas Audio-Video Feature-Level Fusion (v1)

## What this model does

Combines frozen pretrained audio and video embeddings to predict
`has_offscreen`: whether a 10-second Urbansas clip's audio annotations
contain an `offscreen` vehicle event.

Three MLPs are compared, all with hidden=256 and dropout=0.3:
- **Audio-only**: 768-dim AST embedding → 256 → 2
- **Video-only**: 512-dim R(2+1)D-18 embedding → 256 → 2
- **Feature-fusion**: 1280-dim concatenated → 256 → 2

The purpose is to test whether feature-level fusion of frozen
pretrained encoders improves over either modality alone, or over the
probability-level late fusion from Week 9c.

## What this model does NOT do

- **Not a claim that feature fusion helps.** MAVD pooled OOF F1 0.4079
  vs 0.3956 audio-only (Δ +0.0123) is not statistically conclusive
  (95% CI [−0.060, +0.088] includes zero). Audio-only achieves
  stronger ROC-AUC, AP, and Brier score.
- **Not a fair test of video's potential contribution.** Uses Kinetics-only
  video embeddings (never fine-tuned on Urbansas). A fine-tuned video
  encoder cannot be used for feature fusion without per-fold retraining.
  See "Known limitations" #1.
- **Not a cross-modal transformer or attention model.** Simple MLP on
  concatenated embeddings. Cross-attention is deferred.
- **Not a proof that multimodal fusion cannot work.** Establishes only
  that this frozen-feature setup did not demonstrate a conclusive
  improvement under this evaluation protocol.

## A methodological error and its correction

**C2 (initial run, invalid):** extracted video embeddings from
`urbansas_video_final.pt`, which was the Phase 2 model from Week 9c
trained on **all 540 MAVD clips**. The 5-fold LOLA evaluation of C2
therefore did not measure generalization to unseen locations — the
encoder had seen every location during fine-tuning. Video-only MLP got
MAVD F1 0.9065, which reflects encoder-level leakage, not task
generalization.

**C2b (corrected, this card):** re-extracts video embeddings from
**fresh Kinetics-400 pretrained R(2+1)D-18 weights**, never fine-tuned
on Urbansas. Verified by comparing encoder weights to Kinetics-pretrained
reference: 222 mismatches in C2 (fine-tuned), 0 expected in C2b
(fresh weights). The audio encoder (AST) was verified to be unchanged
from AudioSet-pretrained (0 mismatches, max diff 0.0), so audio
embeddings are clean in both C2 and C2b.

**Both runs' artifacts are preserved:**
- C2: `feature_fusion_oof.csv`, `feature_fusion_results.json`,
  `video_embeddings.npy`
- C2b: `feature_fusion_oof_kinetics.csv`, `feature_fusion_results_kinetics.json`,
  `video_embeddings_kinetics.npy`

**C2's TAU external result is honest** (TAU was never used in training).
C2's MAVD OOF result is contaminated. Only C2b is used for MAVD claims.

## Architecture

Three MLP variants, each with the same structure:

```
input (768 or 512 or 1280) → Linear(256) → ReLU → Dropout(0.3) → Linear(2)
```

- **Audio-only MLP:** 768 → 256 → 2, 197,378 params
- **Video-only MLP:** 512 → 256 → 2, 131,842 params
- **Feature-fusion MLP:** 1280 → 256 → 2, 328,450 params

**Parameter count caveat:** feature fusion has 66% more parameters than
audio-only. If it had won, the gain could not be attributed solely to
cross-modal interaction — additional capacity is a confound. That
caveat applies regardless of outcome.

## Training protocol

- **Data**: 540 MAVD clips, 540 TAU clips (external test)
- **Split**: 5-fold leave-one-location-out on MAVD
- **Scaling**: `StandardScaler` fit on 4 training locations only, applied
  to both train and test (no leakage)
- **Epochs**: 20 fixed, no early stopping (avoids using the held-out
  test fold for model selection)
- **Loss**: `CrossEntropyLoss` with per-fold `n_neg/n_pos` class weight
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR over 20 epochs
- **Batch size**: 32
- **Seeds**: 42 + fold_idx (per fold), 142 (final model); deterministic
  mode enabled
- **No augmentation** (fixed embeddings, no meaningful augmentation)

## Results — MAVD pooled OOF (C2b, honest)

| Model | F1@0.5 | P | R | ROC-AUC | AP | Brier | Pos rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Audio-only MLP | 0.3956 | 0.4000 | 0.3913 | **0.7042** | **0.4334** | **0.2240** | 0.250 |
| Video-only MLP | 0.3862 | 0.3206 | 0.4855 | 0.5788 | 0.3154 | 0.2996 | 0.387 |
| **Feature fusion** | **0.4079** | 0.3349 | **0.5217** | 0.6605 | 0.3899 | 0.2959 | 0.398 |

**Confusion matrices (threshold 0.5):**

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Audio-only | 321 | **81** | 84 | 54 |
| Video-only | 260 | 142 | 71 | 67 |
| Feature fusion | 259 | 143 | **66** | **72** |

**Feature fusion gets the highest F1 by catching 18 more true positives
than audio-only — but at the cost of 62 more false positives.**

## Results — per-location F1

| Location | n / pos | Audio | Video | Fusion |
|---|---|---|---|---:|
| `acevedo0103` | 100 / 32 | 0.3836 | 0.3396 | 0.3793 |
| `magallanes0308` | 175 / 48 | 0.3951 | 0.4434 | **0.4523** |
| `rivera0922` | 30 / 2 | 0.5714 | 0.0000 | 0.0000 |
| `rivera0923` | 145 / 28 | **0.5106** | 0.4400 | 0.4762 |
| `templete0103` | 90 / 28 | **0.3077** | 0.0000 | 0.2308 |

**Feature fusion wins on 1 of 5 locations (`magallanes0308`), loses on
2, and roughly ties on 2 (with `rivera0922` having too few positives to
be meaningful).** The mean improvement is not from uniform gains.

## Results — TAU external

| Model | F1 | P | R | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| Audio-only MLP | 0.2848 | 0.1946 | 0.5309 | 0.6059 | 0.2321 |
| Video-only MLP | 0.2890 | 0.1875 | 0.6296 | 0.5776 | 0.1947 |
| **Feature fusion** | **0.3154** | **0.2222** | 0.5432 | **0.6158** | **0.2656** |

**All three models are weak on TAU.** Fusion leads on F1, ROC-AUC, and
AP, but the margin over audio-only is +0.031 F1 / +0.010 AUC / +0.034
AP — within noise of a single evaluation. Absolute values remain far
below deployment quality.

**TAU predicted positive rates:** audio 40.9%, video 50.4%, fusion 36.7%
(actual 15.0%). All three over-predict substantially, with video the
worst offender.

## Location-dependent prediction behavior (the central finding)

**The video-only MLP's predicted positive rate varies dramatically
across held-out MAVD locations:**

| Location | Actual pos rate | Video predicted pos rate |
|---|---:|---:|
| `acevedo0103` | 32.0% | 21.0% |
| `magallanes0308` | 27.4% | **93.7%** |
| `rivera0922` | 6.7% | 6.7% |
| `rivera0923` | 19.3% | 15.2% |
| `templete0103` | 31.1% | **0.0%** |

**At `magallanes0308`, the video MLP predicts positive for 94% of
clips. At `templete0103`, it predicts positive for 0%.** Actual positive
rates are 27.4% and 31.1% respectively — similar. **The MLP's predicted
positive rate is not tracking the actual positive rate.**

**Probability distributions confirm:**

- `magallanes0308` (video): 10th percentile 0.62, median 0.88. **Every
  clip gets probability above 0.6.**
- `templete0103` (video): 90th percentile 0.19. **Every clip gets
  probability below 0.2.**

**Because the held-out location was excluded from the training fold
for that evaluation, this is not direct memorization of the held-out
location.** The MLP cannot have seen `magallanes0308`'s labels when
`magallanes0308` was the test fold. What we observe instead is:

> **The video representation and downstream classifier are sensitive to
> location-associated visual characteristics, with prediction behavior
> that does not transfer consistently across unseen locations.**

The most plausible explanation is **scene-context shortcut learning**:
the Kinetics-pretrained video embeddings capture visual scene texture
sufficient to distinguish MAVD locations, and the MLP learns to use
those visual characteristics as a proxy for "is this likely positive."
When the held-out location's visual characteristics happen to correlate
with the training folds' positive examples, the model over-predicts;
when they don't, it under-predicts.

**This is inherited from the same shortcut documented in Week 9c's
video classifier** (55% of high-confidence false positives were on
`only_visible` clips — "visible vehicle → predict offscreen"). **The
embedding does not escape the shortcut; it encodes the features that
enable it.**

**Feature fusion partially inherits the pattern:** at `magallanes0308`,
fusion pos rate is 86.3% (following video); at `templete0103`, 26.7%
(partially following audio but still below its 41.1%). The fusion
MLP doesn't resolve the location dependence — it averages between
audio's more honest behavior and video's shortcut.

## Known limitations

1. **C2 contamination (documented above).** The initial C2 run used a
   fine-tuned video encoder that had trained on all 540 MAVD clips. Its
   MAVD OOF results are contaminated at the embedding level. Only C2b's
   results are used for MAVD claims.

2. **Fusion's F1 gain is not statistically conclusive.** Δ F1 +0.0123
   at clip level, 95% CI [−0.0605, +0.0884]. Location-level bootstrap
   (5 clusters) gives Δ −0.0026 [−0.0836, +0.0482] — essentially zero
   and slightly negative.

3. **Fusion loses on threshold-independent metrics.** Audio-only
   achieves stronger ROC-AUC (0.7042 vs 0.6605), AP (0.4334 vs 0.3899),
   and Brier score (0.2240 vs 0.2959). **The F1 gain at threshold 0.5
   comes from a higher predicted positive rate (0.398 vs 0.250), not
   better ranking.** Higher recall, lower precision.

4. **Parameter count confound.** Feature fusion has 328,450 params vs
   audio-only's 197,378 — 66% more. Any small advantage cannot be
   attributed solely to cross-modal interaction.

5. **Video uses pretrained-only embeddings.** Fine-tuning the video
   encoder on Urbansas would require per-fold checkpointing for fair
   feature-fusion evaluation. Deferred to future work (documented below).

6. **Location-dependent prediction behavior is observed but not fully
   explained.** Scene-context shortcut learning is the most plausible
   hypothesis. A direct test — e.g., training a classifier on scene
   metadata alone and comparing its behavior — is deferred.

7. **TAU F1 gain over audio is small and single-evaluation.** Δ F1
   +0.031 with no bootstrap CI computed. Do not claim reliable external
   generalization.

8. **`rivera0922` is 2 positives** — F1 and AUC from that fold should
   not be interpreted as meaningful signal.

9. **No threshold tuning.** Threshold fixed at 0.5, evaluated once on
   TAU. Not tuned on any held-out data.

## What this model demonstrates

- Frozen pretrained audio + video embeddings, cached and verified
- Three MLP variants with identical architecture, differing only in input
- Per-fold `StandardScaler` (no leakage)
- Same 5-fold LOLA as Week 9c baselines
- Audio-only MLP control (parameter-matched to fusion at hidden=256)
- Video-only MLP control
- Paired bootstrap CIs at clip and location levels
- Per-location probability distribution analysis
- Honest reporting of a **methodological error (C2 contamination)** and
  its **correction (C2b)**
- Honest reporting of a **negative/inconclusive result** when simpler
  metrics (F1 at 0.5) would have supported a positive claim

## What this model is NOT a demonstration of

- That feature fusion improves over audio-only
- That cross-modal interactions are learned
- That video embeddings are task-relevant
- That fusion helps on external data
- That multimodal fusion cannot work (it may, with fine-tuned encoders)

## Future work

- **Fold-specific fine-tuned video embeddings.** Retrain Week 9c video
  model with per-fold checkpoints saved inside the fold loop. Extract
  embeddings using each fold's model. Re-run feature fusion. ~1.5h
  compute + ~10 min extraction + ~10 min C2 re-run.
- **Scene-metadata baseline.** Train a classifier on coarse scene
  attributes (location if available, night, city) to establish how much
  of the video MLP's behavior is explainable by location identity alone.
- **Cross-attention fusion.** If feature-level MLPs are insufficient
  (as shown here), a transformer over modality-specific sequences might
  capture temporal interactions. Deferred until simpler approaches are
  fully exhausted.
- **Threshold-independent evaluation of TAU.** Compute bootstrap CIs on
  TAU for the three models to determine whether fusion's TAU lead is
  statistically distinguishable from audio-only.

## Reproduce

```powershell
# 1. Extract embeddings (needs Week 9c checkpoints)
uv run python scripts\extract_urbansas_embeddings.py --encoder-variant kinetics_only

# 2. Run C2b with clean embeddings
uv run python scripts\train_urbansas_feature_fusion.py --video-embedding-variant kinetics_only
```

## MLflow

- Experiment: `urbanintel-week9b-urbansas`
- No separate MLflow runs for the MLP training; results stored in JSON
  files alongside the model cards.

## Citation

% TODO: Replace with exact citations from CITATIONS.md before final portfolio polish.

```bibtex
@inproceedings{gong2021ast,
  title={AST: Audio Spectrogram Transformer},
  author={Gong, Yuan and Chung, Yu-An and Glass, James},
  booktitle={Interspeech},
  year={2021}
}

@inproceedings{tran2018r2plus1d,
  title={A Closer Look at Spatiotemporal Convolutions for Action Recognition},
  author={Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and Paluri, Manohar},
  booktitle={CVPR},
  year={2018}
}

@inproceedings{urbansas_2022,
  title  = {The Urbansas Dataset: Urban Scene Understanding from Paired Audio and Video},
  author = {...},
  year   = {2022},
  note   = {See https://zenodo.org/records/6658386}
}
```