# Model Card: Urbansas Offscreen-Vehicle Video Classifier (R(2+1)D-18 v1)

## What this model does

Given a 10-second Urbansas clip's video (16 uniformly-sampled RGB frames at
112×112), predicts a probability that the clip's audio annotations include
the `offscreen` vehicle label.

**Critical caveat — read this first:** the model does not directly detect
offscreen vehicles. It cannot — offscreen vehicles are by definition not in
the video. The ROC-AUC above chance (0.613 on MAVD) reflects **scene-level
correlation with the annotation policy**, not detection capability. See
"What this model actually learned" below.

## What this model does NOT do

- **Not an offscreen-vehicle detector.** The video model sees 16 stills at
  112×112 from a 10-second clip. There is no pixel corresponding to an
  offscreen vehicle. The ROC-AUC above chance reflects correlation between
  scene type and the offscreen annotation policy, not detection.
- **Not a visible-vehicle detector either.** A clip with a visible vehicle
  but no offscreen annotation (`only_visible`) is a *negative* under this
  task. The model is not trained to predict "is there a vehicle."
- **Not a cross-location generalizer.** Per-location MAVD ROC-AUC varies
  from 0.501 to 0.721. `magallanes0308` is at chance; `templete0103` reaches
  0.721. Cause not identified.
- **Not a cross-dataset generalizer in a strong sense.** TAU ROC-AUC 0.6343
  is better than audio-only's 0.5587 but the model's confident positive
  predictions are largely false positives on `only_visible` TAU clips.
- **Not evaluated at deployment conditions.** All evaluation on annotated
  Urbansas clips. Real camera feeds, varying quality, overlapping traffic
  are untested.

## What this model actually learned

The disagreement analysis (`scripts/diagnose_urbansas_audio.py` and follow-up
inspection) revealed: **55% of the video model's high-confidence false
positives are on `only_visible` clips** — clips that contain a visible
vehicle but no offscreen annotation. On these clips, the video model
confidently predicts positive (mean p ≈ 0.80) while audio correctly predicts
negative (mean p ≈ 0.08). On `empty` clips, the video model also fires
positive at p ≈ 0.73.

**Interpretation:** the video model has learned a *scene-context prior*.
"Empty road at a certain geometry" → predicts offscreen likely. "Visible
vehicle present" → predicts offscreen likely. This correlation exists in
the annotation policy (roads → vehicles → some offscreen), but is not the
task. The model is a weak scene-context classifier standing in for a task
it cannot actually solve.

**Two post-hoc corrections to this interpretation:**

1. **The `empty` "false positives" are not clean.** A later audit
   (2026-10-27, `docs/experiments.md` 10a) revealed that 57% of `empty` clips
   carry `non_identifiable_vehicle_sound = 1` — the annotator heard a
   vehicle-like sound but could not classify it. The video model firing
   positive on these clips may be detecting real offscreen vehicle events that
   the annotation pipeline could not label. **This is not confirmed** — we
   did not verify the acoustic content of those clips — but it means "video
   confidently wrong on empty clips" was an overstatement. Video may be
   partly right; the labels are ambiguous.

2. **The video model's TAU transfer may be substantially city-level.** A
   city-only logistic regression on TAU achieves F1 0.2982 ± 0.0525
   (5-fold stratified on 7 cities with ≥20 clips), while the video model on
   TAU achieves F1 0.2362 (on all 540 clips). The two use different clip
   sets; a strict comparison would evaluate the city baseline on the same
   540 clips. But the direction is clear: a non-trivial fraction of the
   video model's TAU signal is likely captured by city-level priors. The
   video model's contribution *beyond* city identity is not established.

**Combined honest statement:** the video model learns a scene-context prior
that correlates with the offscreen annotation policy. Part of this correlation
is city-level (TAU), part is scene-level (`only_visible`), and part may be
ambiguous `-1` clips that were labelled as empty but carry vehicle-sound
flags. None of these are direct offscreen-vehicle detection — the model cannot
see what is offscreen — but the "55% wrong on `only_visible`" framing
understates the ambiguity.

## Architecture

- **Backbone**: `torchvision.models.video.r2plus1d_18` pretrained on
  Kinetics-400 (`R2Plus1D_18_Weights.KINETICS400_V1`)
- **Total parameters**: 31,301,151
- **Head**: `nn.Linear(512, 2)` replacing the original 400-class head
- **Input**: `(B, 3, 16, 112, 112)` float32, Kinetics-normalized
  (mean `[0.43216, 0.394666, 0.37645]`, std `[0.22803, 0.22145, 0.216989]`)
- **Frame sampling**: 16 uniformly-spaced frames over the 10-second clip
  via decord, with cv2 fallback

## Training protocol

- **Data**: 540 MAVD clips (5 locations, 138 offscreen positives at 25.6%)
- **Protocol**: 5-fold leave-one-location-out on MAVD. Each fold holds out
  one entire Madrid location. Final model retrained on all 540 MAVD, then
  evaluated once on all 540 TAU clips at threshold 0.5 (fixed before
  evaluation).
- **Target**: `has_offscreen = "offscreen" in audio_annotations[filename]`
- **Loss**: `CrossEntropyLoss(weight=[1, n_neg/n_pos])` per fold
  (weights: 2.59–3.15 across folds; 2.91 for the final model)
- **Optimizer**: AdamW with differential LR — backbone at `0.3 × lr`,
  head at `lr`. Base `lr=3e-4`, `weight_decay=1e-4`.
- **Scheduler**: cosine annealing over 15 epochs
- **Batch size**: 4
- **Augmentation (train only)**: random crop pad=8, hflip 50%
- **AMP**: `torch.amp.autocast` + `GradScaler`
- **Random seed**: `42 + fold_idx` per fold, `142` for final model
- **Frame cache**: precomputed 2.60 GB memmap for fast loading

## Results — MAVD LOLA CV

| Fold | Location | F1 | P | R | n_pos |
|---|---:|---:|---:|---:|---:|
| 1 | acevedo0103 | 0.5067 | 0.442 | 0.594 | 32 |
| 2 | magallanes0308 | 0.4466 | 0.291 | 0.958 | 48 |
| 3 | rivera0922 | 0.0000 | 0.000 | 0.000 | 2 (noise) |
| 4 | rivera0923 | 0.3333 | 0.500 | 0.250 | 28 |
| 5 | templete0103 | 0.5846 | 0.514 | 0.679 | 28 |

**Mean F1**: 0.3742 ± 0.2043
**Pooled OOF**: F1 0.4643, P 0.358, R 0.659, ROC-AUC 0.6130, PR-AUC 0.3063

### Per-location ROC-AUC

| Location | n | Pos | ROC-AUC |
|---|---:|---:|---:|
| acevedo0103 | 100 | 32 | 0.641 |
| magallanes0308 | 175 | 48 | 0.501 |
| rivera0922 | 30 | 2 | 0.393 (n=2, noise) |
| rivera0923 | 145 | 28 | 0.650 |
| templete0103 | 90 | 28 | 0.721 |

**`magallanes0308` is at chance while `templete0103` reaches 0.721.** The
cause was not identified. Notably different from audio's bimodal pattern
(audio was at chance on `acevedo0103` and `templete0103`, useful on
`magallanes0308` and `rivera0923`). **This is the complementarity that
motivated the fusion experiment.**

### Per-subclass ROC-AUC (vs negative pool of `only_visible` + `empty`)

| Subclass | n | ROC-AUC |
|---|---:|---:|
| `only_offscreen` | 66 | **0.746** |
| `both` | 72 | 0.491 |

Video is strong on `only_offscreen` (0.746) — clips with offscreen events
but no visible vehicles — and near chance on `both`. This is the opposite
of audio's pattern (audio: 0.660 on `only_offscreen`, 0.590 on `both`).

## Results — TAU external

| Metric | Value |
|---|---:|
| Positives | 81 / 540 (15.0%) |
| ROC-AUC | **0.6343** |
| PR-AUC | 0.2509 |
| F1 @ 0.5 | 0.2362 |
| Precision @ 0.5 | 0.326 |
| Recall @ 0.5 | 0.185 |
| Accuracy @ 0.5 | 0.8204 |

**TAU ROC-AUC 0.6343 is meaningfully better than audio-only's 0.5587.**
The video model transfers across cities more gracefully than the from-scratch
audio model. This is likely because Kinetics-400 pretraining provides a
general visual representation that survives Madrid → European city shift,
while the from-scratch audio model learned Madrid-specific acoustic
signatures.

## Known limitations

1. **The model cannot detect offscreen vehicles.** It cannot see them. The
   ROC-AUC above chance reflects scene-context correlation with the
   annotation policy. **This is the most important limitation.** Any claim
   that this model "detects offscreen vehicles" is incorrect. Additionally,
   a post-hoc audit found that the model's `empty`-clip predictions may
   partly reflect real vehicle sounds the annotation pipeline could not
   classify (57% of `-1` clips carry `non_identifiable_vehicle_sound = 1`),
   and its TAU signal is not clearly beyond a trivial city-only classifier.
   The scene-context signal, while real, is confounded by city priors and
   label ambiguity.

2. **55% of high-confidence false positives are `only_visible` clips.** The
   model has learned "visible vehicle → predict offscreen." On clips with
   visible vehicles but no offscreen annotation, it confidently predicts
   positive (mean p 0.80) while audio correctly predicts negative. This is a
   genuine false-positive mode — `only_visible` is a clean negative class
   (357 clips, all confirmed visible-vehicle-only). The `empty` subclass
   (45 clips in MAVD, 130 in full Urbansas) is separately ambiguous (see
   limitation #1 above); false positives on `empty` should not be treated
   the same as false positives on `only_visible`.

3. **Per-location transfer is bimodal and unexplained.** 0.501 on
   `magallanes0308` vs 0.721 on `templete0103`. Cause not identified. Similar
   to audio's bimodality, but at different locations.

4. **TAU class-prior shift.** MAVD positive rate 25.6%, TAU 15.0%. A model
   trained at the MAVD prior under-predicts on TAU's lower base rate. This
   is partially addressed by the class-weighted loss during training, but
   still contributes to TAU's low recall at threshold 0.5 (0.185).

5. **From-scratch on 540 clips with 31M pretrained parameters.** The
   backbone is pretrained, but only 540 clips are available for fine-tuning.
   The model likely overfits to MAVD-specific visual patterns. Augmentation
   mitigates this partially but not fully.

6. **`rivera0922` fold is unreliable.** 2 positives. F1 0.0000 is noise.

7. **No temporal reasoning over events.** 16 uniformly-sampled frames over
   10s give ~0.66s temporal resolution. The model sees 16 stills, not
   continuous motion. Events that happen between sampled frames are invisible.

8. **`only_visible` and `empty` subclass F1 is trivially 0.** Both subclasses
   have zero positive labels. F1 cannot distinguish false-positive behavior
   across models on these subclasses.

9. **Post-hoc audit corrections (2026-10-27).** Two amendments made after the
   original model card:
   - The `empty` subclass is not a clean negative — 57% of `-1` clips carry
     `non_identifiable_vehicle_sound = 1`.
   - The TAU signal may be substantially city-level (city-only baseline F1
     0.2982 vs video F1 0.2362, different clip sets).
   See `docs/dataset_cards/urbansas.md` and the 10a experiments entry for
   details.

## What this model demonstrates

- Video fine-tuning with Kinetics-400 pretraining on a small paired dataset
- 5-fold LOLA evaluation with scene-leakage prevention
- Precomputed frame cache for fast training (6× speedup over cv2 seek-per-frame)
- Frame-level augmentation matching the R(2+1)D-18 training distribution
- Honest reporting of the "cannot detect what it cannot see" limitation
- Per-location and per-subclass diagnostics revealing scene-context shortcut

## What this model is NOT a demonstration of

- Offscreen vehicle detection
- Generalizable urban video understanding
- Robust performance across cities
- Understanding of motion patterns over time (only 16 sampled stills)
- Detection quality suitable for deployment

## Reproduce

```powershell
# one-time precompute: ~5 min, saves 2.6 GB
uv run python scripts/precompute_urbansas_video_frames.py

# train: ~1 hour on RTX 4050
uv run python scripts/train_urbansas_video.py --epochs 15 --batch-size 4 --num-workers 4

# diagnose: same script as audio diagnostics
uv run python scripts/diagnose_urbansas_audio.py \
    --oof data/urbansas_results/urbansas_video_oof.csv \
    --tau data/urbansas_results/urbansas_video_tau_external.csv
```

## MLflow

- Experiment: `urbanintel-week9b-urbansas`

## Citation

% TODO: Replace with exact citations from CITATIONS.md before final portfolio polish.

```bibtex
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