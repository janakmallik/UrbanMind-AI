# Model Card: Urbansas Audio-Video Late Fusion (v1)

## What this model does

Combines two frozen single-modality baselines — a from-scratch 1D-CNN on
128-mel log-spectrograms (audio) and R(2+1)D-18 Kinetics-400 pretrained
(video) — at the probability level to predict `has_offscreen`: whether a
10-second urban clip's audio annotation set contains an `offscreen`
vehicle event.

Six fusion variants are evaluated on the same 5-fold leave-one-location-out
protocol:

| Name                  | Inputs                                          | Fitted parameters |
|-----------------------|-------------------------------------------------|-------------------|
| `audio`               | p_audio                                         | 0 (baseline)      |
| `video`               | p_video                                         | 0 (baseline)      |
| `mean`                | (p_audio + p_video) / 2                         | 0                 |
| `max`                 | max(p_audio, p_video)                           | 0                 |
| `logistic`            | LogisticRegression([p_audio, p_video])          | 3 (2 coef + 1 int) |
| `logistic_interaction`| LogisticRegression([p_audio, p_video, pa*pv])   | 4 (3 coef + 1 int) |

## What this model does NOT do

- **Not a feature-level or cross-attention fusion.** This is probability-level
  late fusion. Same-modality vision feature fusion (Week 9a) and audio-visual
  feature fusion (deferred) are different experiments. Late fusion discards
  most of the information learned inside each backbone.
- **Not a claim of generalization to unseen cities.** Best fusion transfer to
  TAU is max fusion at 0.2308 F1, roughly matching video-only (0.2362). The
  logistic fusion models collapse on TAU to 0.0230. Do not extrapolate.
- **Does not resolve the outer-fold leakage concern for logistic fusion
  heads** (see Known Limitations #1). The +0.079 logistic improvement over
  audio is promising but not definitive.
- **Does not prove the complementary-modality hypothesis.** The subclass
  results are consistent with complementary signals, but descriptive, not
  causal.
- **Not a claim that video "detects" offscreen vehicles.** Video cannot see
  offscreen vehicles by definition. Per the video baseline model card, the
  video model's signal is scene-context correlation with the annotation
  policy, not direct detection.

## Architecture

No single architecture. Six separate models:

- **`audio`, `video`**: direct use of frozen baseline probabilities. Each
  baseline is a separately-trained model with its own architecture (see
  `urbansas_audio_cnn_v1.md` and `urbansas_video_cnn_v1.md`).
- **`mean`, `max`**: trivial arithmetic on the two probabilities. Zero fitted
  parameters.
- **`logistic`, `logistic_interaction`**: `sklearn.linear_model.LogisticRegression`
  with `C=1.0`, `solver="lbfgs"`, `max_iter=1000`, `class_weight="balanced"`.
  Three or four fitted parameters.

**No backbones are trained here.** All base-model probabilities come from the
audio and video training scripts (see Reproduce).

## Training protocol

- **Outer evaluation**: 5-fold leave-one-location-out on MAVD. For fold `L`:
  - Fit fusion head on the 4 non-`L` locations' OOF probabilities (~432 clips).
  - Evaluate on `L`'s OOF probabilities (~108 clips).
- **Fixed threshold**: 0.5 everywhere. No threshold tuning on MAVD test folds
  or TAU.
- **No hyperparameter search**. Logistic regression is fit once per fold with
  fixed `C=1.0`.
- **No retraining of base models.** Audio and video OOF probabilities were
  produced in earlier experiments; each `p_audio` value is from an audio
  model that did not see that clip; each `p_video` value is from a video model
  that did not see that clip.
- **Class weighting**: `class_weight="balanced"` on the two logistic heads.
  Without this, both logistic models collapsed to predicting mostly-negative
  with F1 ≈ 0.09 on the pooled set. The unweighted results are preserved at
  `data/urbansas_results/fusion_results_unweighted.json` for reproducibility.

## Results — MAVD pooled OOF

| Model                  | F1@0.5 | P     | R     | ROC-AUC | PR-AUC |
|------------------------|-------:|------:|------:|--------:|-------:|
| audio                  | 0.3686 | 0.348 | 0.391 | 0.6237  | 0.3296 |
| video                  | 0.4643 | 0.358 | 0.659 | 0.6130  | 0.3063 |
| mean                   | 0.4167 | 0.400 | 0.435 | **0.6704** | **0.3841** |
| max                    | **0.4713** | 0.333 | **0.804** | 0.6176 | 0.3121 |
| logistic (weighted)    | 0.4480 | 0.326 | 0.717 | 0.6315  | 0.3480 |
| logistic_interaction (weighted) | 0.4510 | 0.329 | 0.717 | 0.6263 | 0.3439 |

Pooled OOF includes all 540 MAVD clips. Positive rate 0.256.

## Results — per-location F1 (5 LOLA folds)

| Model         | acevedo0103 (32 pos) | magallanes0308 (48 pos) | rivera0922 (2 pos) | rivera0923 (28 pos) | templete0103 (28 pos) |
|---------------|------:|------:|------:|------:|------:|
| audio         | 0.4138 | 0.2154 | 0.5000 | 0.4483 | 0.3797 |
| video         | 0.5067 | 0.4466 | 0.0000 | 0.3333 | **0.5846** |
| mean          | 0.4416 | 0.3871 | 0.0000 | 0.4091 | 0.4444 |
| max           | 0.4902 | 0.4466 | 0.3333 | 0.4444 | 0.5319 |
| logistic      | 0.4948 | 0.4402 | 0.0000 | 0.3556 | 0.4828 |
| logistic+I    | **0.5106** | 0.4381 | 0.0000 | 0.3636 | 0.4828 |

**Sign consistency (model F1 > audio-only F1, per location):**
- video: 3/5
- mean: 3/5
- max: 3/5
- logistic: 3/5
- logistic_interaction: 3/5

`rivera0922` has only 2 positives; its per-fold F1 is noise (0.5000 is 1 of 2
true positives). Do not interpret it as a reliable estimate.

## Results — per-subclass F1

| Model         | only_offscreen (n=66) | both (n=72) | only_visible (n=357) | empty (n=45) |
|---------------|------:|------:|------:|------:|
| audio         | 0.5957 | 0.5306 | 0.0000 | 0.0000 |
| video         | 0.9524 | 0.6019 | 0.0000 | 0.0000 |
| mean          | 0.7308 | 0.4681 | 0.0000 | 0.0000 |
| max           | **0.9767** | **0.8000** | 0.0000 | 0.0000 |
| logistic      | 0.9524 | 0.7027 | 0.0000 | 0.0000 |
| logistic+I    | 0.9524 | 0.7027 | 0.0000 | 0.0000 |

**Reading this table**: `only_visible` and `empty` subclasses have zero
positive labels. F1 is trivially 0 for all models — the metric cannot
distinguish false-positive behavior across models. **Use the pooled
precision/recall columns instead for those two subclasses.**

For `only_offscreen` (66 clips, all positive) and `both` (72 clips, all
positive), the fusion models substantially improve over audio-only. This is
consistent with a complementary-modality signal but does not prove causation.

## Results — per-fold mean ± std (across 5 LOLA folds)

| Model         | Mean F1 ± Std |
|---------------|------:|
| audio         | 0.3914 ± 0.0966 |
| video         | 0.3742 ± 0.2043 |
| mean          | 0.3364 ± 0.1696 |
| max           | **0.4493 ± 0.0663** |
| logistic      | 0.3547 ± 0.1839 |
| logistic+I    | 0.3590 ± 0.1863 |

**Max fusion has the highest mean F1 and the lowest variance.** This is
notable — the zero-parameter OR-logic produces a more stable result than
either baseline.

## Bootstrap CIs on F1 difference (1000 resamples, paired)

| Comparison                          | Resample unit | Mean diff | 95% CI                | P(fusion > audio) |
|-------------------------------------|---------------|----------:|-----------------------|------------------:|
| logistic (weighted) − audio         | clip          | +0.0800   | [+0.0158, +0.1444]    | 0.994 |
| logistic (weighted) − audio         | location      | +0.0770   | [−0.0258, +0.1653]    | 0.934 |
| logistic_interaction (weighted) − audio | clip      | +0.0831   | [+0.0169, +0.1487]    | 0.997 |
| logistic_interaction (weighted) − audio | location  | +0.0810   | [−0.0193, +0.1653]    | 0.937 |

**Interpretation:**
- **Clip-level** CI excludes zero for both weighted logistic variants.
- **Location-level** CI (5 resampling units) includes zero for both.
- The two CIs disagree because clip-level treats clips as independent while
  location-level treats locations as the independent unit. **With only 5
  locations, the location-level CI is the more honest one for claims about
  generalization to new locations.** The clip-level CI is more comparable to
  how the baselines were reported.

Max fusion was not tested against video-only in this table. Its pooled
advantage over video (+0.007) is small enough that a bootstrap CI would
likely include zero.

## TAU external (fusion head fit on all MAVD OOF probs, applied to TAU)

| Model                  | F1     | P     | R     |
|------------------------|-------:|------:|------:|
| audio                  | 0.0000 | 0.000 | 0.000 |
| video                  | **0.2362** | 0.326 | 0.185 |
| mean                   | 0.0000 | 0.000 | 0.000 |
| max                    | **0.2308** | 0.306 | 0.185 |
| logistic (weighted)    | 0.0230 | 0.167 | 0.012 |
| logistic_interaction (weighted) | 0.0230 | 0.167 | 0.012 |

TAU positive rate 0.150 (81/540). Threshold 0.5 fixed before TAU evaluation.

**Only video-only and max fusion transfer meaningfully to TAU.** All other
models collapse. The base audio model's TAU probabilities are near-zero
across the board — mean fusion (and logistic fusion, which inherits this)
collapse to near-zero predictions at threshold 0.5.

## Known limitations

1. **Leakage concern for logistic fusion heads — investigated by two-fold diagnostic pilot.**
   
   The fusion head's training probabilities for fold `L` come from base models
   that were trained on location `L`'s data in other folds. This is a subtle
   leakage channel.
   
   **Two-fold diagnostic pilot (2026-10-27).** Each pilot retrained base models
   on a leak-free inner split (different assignment per pilot) and trained a
   fresh logistic head on the resulting probabilities, then evaluated on the
   outer fold. Both pilots ran 15 epochs with the default `num_workers=4`:
   
   | Pilot | Outer fold | Standard head F1 | Nested head F1 | Delta |
   |---|---|---|---|---|
   | 1 | `acevedo0103` | 0.4948 | 0.5185 | **+0.0237** |
   | 2 | `magallanes0308` | 0.4402 | 0.4279 | **−0.0123** |
   | **Average** | | | | **≈ +0.0057** |
   
   The nested heads used substantially fewer training clips than the standard
   heads (265 vs 440 in pilot 1; 130 vs 365 in pilot 2). The standard head had
   ~2–3× the training data in both pilots.
   
   **Interpretation.** Two diagnostic folds with different inner splits show
   deltas within ±0.024 of zero. **This provides limited evidence against a
   large, consistently positive effect from the suspected leakage channel.
   It does not establish that leakage is negligible, and it does not rule out
   leakage as a contributor to the pooled +0.08 result.** The near-zero delta
   is consistent with several possibilities:
   
   - Leakage is small.
   - Leakage is real but offset by training-set-size effects.
   - Leakage is fold-dependent (small on these two folds, possibly larger on
     others).
   
   The comparison is further confounded by training-set size and location
   distribution. **The two pilots are diagnostic folds, not independent
   replications of the full experiment.**
   
   **Full nested CV remains future work.** Estimated cost: **12–18h** on this
   hardware (15-epoch basis, from measured pilot runtimes). Not run.
   
   **The +0.08 pooled improvement should be read as "promising, not proven."**
   The pilot provides diagnostic evidence, not validation. If the +0.08 ever
   becomes a headline claim, full nested CV is the mandatory validation step.
   
   The zero-parameter fusion models (`mean`, `max`) are unaffected — they fit
   no parameters and cannot leak.

2. **Max fusion's MAVD improvement over video-only is marginal.** +0.007
   pooled F1 (0.4713 vs 0.4643). Bootstrap CI not computed for this pair, but
   the point estimate is close enough that it should be treated as "roughly
   tied with video-only on pooled F1, better on per-fold stability (std 0.066
   vs 0.204), better on `both` subclass F1 (0.8000 vs 0.6019)."

3. **Mean fusion has the best ROC-AUC but not the best F1.** Threshold tuning
   on training data (not TAU) is a natural future experiment; deferred here.

4. **Logistic fusion improves MAVD in-domain but not definitively.** The
   clip-level CI excludes zero but the location-level CI includes zero, and
   the leakage concern (limitation #1) is unresolved.

5. **Logistic fusion collapses on TAU (0.0230).** The +0.08 MAVD improvement
   does not transfer. Any production use would need to be evaluated on
   target-domain data.

6. **Subclass improvements support complementary-signal hypothesis but do
   not prove it.** `only_offscreen` F1 goes 0.5957 → 0.9524 (logistic) and
   `both` goes 0.5306 → 0.7027. Consistent with complementary signals; not
   causal. **Post-hoc caveat:** the negative pool used in these subclass
   diagnostics (`only_visible` + `empty`, 402 clips) contains ~32 ambiguous
   clips — clips labelled `-1` where the annotator heard an
   unclassified vehicle sound (`non_identifiable_vehicle_sound = 1`). The
   `only_visible` subclass (357 clips) is a clean negative; the `empty`
   subclass is ambiguous. **The complementarity pattern against
   `only_visible` alone is clean and holds; the pooled negative-pool
   comparison is partly contaminated.** See `docs/dataset_cards/urbansas.md`
   for the audit.

7. **No threshold tuning on TAU.** Threshold 0.5 fixed before evaluation.
   TAU is a held-out external test and was not used to select any fusion
   model or tune any hyperparameter.

8. **`rivera0922` fold is unreliable.** 2 positives; per-fold F1 of 0.5000
   is 1 of 2 true positives. Report as noise.

9. **Post-hoc audit corrections (2026-10-27).** The `empty` subclass used in
   per-subclass diagnostics is ambiguous (57% carry
   `non_identifiable_vehicle_sound = 1`). The `only_visible` subclass is a
   clean negative and the complementarity pattern against `only_visible`
   alone survives. The pooled negative-pool comparison is partly
   contaminated. Additionally, the video model's TAU transfer may be
   substantially city-level (city-only baseline F1 0.2982 vs video F1
   0.2362 on TAU, different clip sets). See `docs/experiments.md` 10a
   entry for details.

## What this model demonstrates

- Probability-level late fusion of frozen baselines on a genuinely paired
  audio-visual dataset
- 5-fold leave-one-location-out evaluation with scene-leakage prevention
- Six fusion variants compared on five views (pooled, per-location,
  per-subclass, per-fold, TAU external)
- Bootstrap CIs at clip and location levels
- Explicit documentation of leakage limitation and its scope
- Honest reporting of the "in-domain helps, out-of-domain collapses" split
  between logistic and max fusion
- Preservation of unweighted logistic results for reproducibility

## What this model is NOT a demonstration of

- Feature-level or cross-attention fusion
- Generalization to unseen cities or datasets
- Causal attribution of gains to complementarity
- A resolved claim that audio and video provide independent information
- Production offscreen-vehicle detection

## Reproduce

```powershell
# 1. Train base models (see their model cards for full details)
uv run python scripts/train_urbansas_audio.py --epochs 15 --batch-size 16 --num-workers 4
uv run python scripts/precompute_urbansas_video_frames.py  # one-time, ~5 min
uv run python scripts/train_urbansas_video.py --epochs 15 --batch-size 4 --num-workers 4

# 2. Run fusion
uv run python scripts/train_urbansas_fusion.py
```

## MLflow

- Experiment: `urbanintel-week9b-urbansas`

## Citation

% TODO: Replace with exact citations from CITATIONS.md before final portfolio polish.

```bibtex
@inproceedings{urbansas_2022,
  title  = {The Urbansas Dataset: Urban Scene Understanding from Paired Audio and Video},
  author = {...},
  year   = {2022},
  note   = {See https://zenodo.org/records/6658386}
}

@inproceedings{tran2018r2plus1d,
  title={A Closer Look at Spatiotemporal Convolutions for Action Recognition},
  author={Tran, Du and Wang, Heng and Torresani, Lorenzo and Ray, Jamie and LeCun, Yann and Paluri, Manohar},
  booktitle={CVPR},
  year={2018}
}
```