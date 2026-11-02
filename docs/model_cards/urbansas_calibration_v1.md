# Model Card: Urbansas Model Reliability (Week 10)

## What this covers

Week 10 evaluates three properties of the trained Urbansas classifiers from Week 9c:

- **Calibration** (W10.1): do predicted probabilities correspond to observed frequencies?
- **Temperature scaling** (W10.2): can post-hoc calibration improve probability reliability without changing ranking?
- **Confidence usefulness** (W10.3): does the model's confidence distinguish likely-correct from likely-wrong predictions?

These three properties are **distinct capabilities**, not three views of the same thing. A model can be well-calibrated but have no useful confidence signal (it predicts a constant), or usefully confident but badly calibrated (it ranks well but gives wrong probabilities). This card documents each separately.

For the underlying model results, see the Week 9c model cards:
`urbansas_audio_cnn_v1.md`, `urbansas_audio_pretrained_v1.md`,
`urbansas_video_cnn_v1.md`, `urbansas_fusion_v1.md`,
`urbansas_feature_fusion_v1.md`.

## Scope and provenance filtering

**MAVD (5-fold LOLA, in-domain):** Only 4 of 7 MAVD models are honest OOF:

| Model | Evaluated? | Reason |
|---|---|---|
| ast_audio | ✅ | AST encoder verified byte-identical to AudioSet (leak-free) |
| feature_fusion_mlp | ✅ | Kinetics-only video encoder, leak-free embeddings |
| feature_fusion_audio_only | ✅ | Same |
| feature_fusion_video_only | ✅ | Same |
| video_finetuned | ❌ | Encoder fine-tuned on all 540 MAVD (in-sample) |
| late_fusion_fromscratch | ❌ | Uses contaminated video predictions |
| late_fusion_ast | ❌ | Same; TAU file also missing |

**TAU (external test):** All 6 available models are honest (TAU never used for training). `late_fusion_ast` TAU is missing — `fusion_tau_external_kinetics.csv` was never regenerated after 9c-ext-1b.

## Part 1 — Calibration measurement (W10.1)

### MAVD (4 honest models)

| Model | ECE | Brier | LogLoss | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| ast_audio | **0.158** | **0.204** | **0.591** | 0.680 | 0.393 |
| feature_fusion_audio_only | 0.201 | 0.224 | 0.836 | **0.704** | **0.433** |
| feature_fusion_video_only | 0.274 | 0.300 | 0.956 | 0.579 | 0.315 |
| feature_fusion_mlp | 0.280 | 0.296 | 1.142 | 0.661 | 0.390 |

**AST audio is best-calibrated on MAVD** (ECE 0.158). But `feature_fusion_audio_only` has the **best discrimination** (ROC-AUC 0.704, AP 0.433) despite worse calibration (0.201). **Calibration and discrimination are independent.**

### TAU (6 honest models)

| Model | ECE | Brier | LogLoss | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|
| **video_finetuned** | **0.110** | **0.137** | **0.530** | 0.634 | 0.251 |
| feature_fusion_mlp | 0.321 | 0.281 | 1.056 | 0.616 | **0.266** |
| ast_audio | 0.328 | 0.257 | 0.710 | 0.557 | 0.180 |
| feature_fusion_audio_only | 0.331 | 0.316 | 1.237 | 0.606 | 0.232 |
| late_fusion_fromscratch | 0.338 | 0.261 | 0.721 | 0.577 | 0.187 |
| feature_fusion_video_only | 0.384 | 0.345 | 1.166 | 0.578 | 0.195 |
| late_fusion_ast | MISSING | — | — | — | — |

**`video_finetuned` is the best-calibrated model on TAU (ECE 0.110)** despite being contaminated on MAVD. Fine-tuning on MAVD improved the model's TAU probability reliability — the two effects are orthogonal.

**Every model degrades in calibration from MAVD to TAU** (ECE gap +0.04 to +0.17). This is the same domain shift that collapsed audio-only TAU F1 in Week 9c, now measured as calibration error.

## Part 2 — Temperature scaling (W10.2)

### Leakage-safe protocol

For each fold: fit temperature T on the 4 training folds' OOF predictions, apply to the held-out fold. Report per-fold and pooled calibrated ECE. For TAU: fit T on all MAVD OOF, apply to TAU. **Never fit on TAU labels.**

### MAVD per-fold improvements

| Model | Mean T | Range | ECE before | ECE after | Δ ECE |
|---|---:|---|---:|---:|---:|
| ast_audio | 0.87 | 0.70–1.11 | 0.158 | 0.154 | **−0.004** (tiny) |
| feature_fusion_mlp | 6.15 | 3.59–10.19 | 0.280 | 0.218 | **−0.061** (good) |
| feature_fusion_audio_only | 3.65 | 3.09–3.98 | 0.201 | **0.107** | **−0.094** (best) |
| feature_fusion_video_only | 5.14 | 2.41–7.74 | 0.274 | 0.238 | −0.037 |

**Temperature scaling helps most where the model was most miscalibrated.** `feature_fusion_audio_only` ECE dropped 47%. AST audio barely changed (T near 1, already well-calibrated). The MLPs were massively overconfident (T ≈ 4–6) and gained substantially.

**AST audio's T = 0.87 is unusual** — most overconfident models need T > 1 (softening). AST needs T < 1 (slight sharpening), meaning its MAVD probabilities are slightly under-dispersed.

### TAU transfer (MAVD-fit T applied to TAU)

| Model | T | ECE before | ECE after | Δ ECE |
|---|---:|---:|---:|---:|
| ast_audio | 0.86 | 0.328 | 0.325 | −0.003 |
| feature_fusion_mlp | 5.52 | 0.321 | 0.304 | −0.017 |
| feature_fusion_audio_only | 3.60 | 0.331 | 0.309 | −0.022 |
| feature_fusion_video_only | 4.65 | 0.384 | 0.355 | −0.030 |

**MAVD-fit temperature does not transfer to TAU.** All models improve by only 0.003–0.030 ECE. The residual TAU miscalibration reflects **base-rate shift** (MAVD 25.6% positive, TAU 15.0%) plus weak TAU discrimination. Temperature scaling fixes overconfidence; it cannot correct for a wrong base rate.

**AST audio is the diagnostic case:** its MAVD-fit T is 0.86 (near 1). Applied to TAU, ECE barely moves. **The model's TAU failure is a discrimination failure (ROC-AUC 0.557, near chance), not a calibration failure.** No probability transform fixes a near-random classifier.

### Sanity checks

- **Per-fold AUC/AP preserved** (asserted in code): temperature scaling is monotonic within a fold.
- **Pooled MAVD AUC/AP differs by 0.007–0.017** after per-fold calibration. This is the leakage-safe protocol effect (5 different T's across folds). Documented, not a bug.
- **TAU risk-coverage curves exactly preserved** (max accuracy diff = 0.0) under single-global temperature scaling. Confirms monotonicity.

### Three distinct calibration problems

The bin-level analysis (W10.1 diagnostic) revealed three separate miscalibration patterns:

1. **Uniform overconfidence** — AST audio MAVD (every bin gap negative, fixable by small T < 1).
2. **Bimodal over/under-confidence** — the MLPs (compressed probabilities: low bins underconfident, high bins overconfident; fixable by large T ≈ 4–6).
3. **Base-rate shift without discrimination** — AST audio TAU (mean_prob 0.478 vs prevalence 0.150, ROC-AUC 0.557 near chance). **Not fixable by temperature scaling.** Requires retraining or prior correction using target-domain labels (which we do not have).

## Part 3 — Confidence usefulness (W10.3)

### Correct definition

- `confidence = max(p, 1-p)` — confidence in the **predicted** class, not the positive class
- `predicted_class = (p >= 0.5)`
- `correct = (predicted_class == y_true)`
- **Bin by confidence, not by positive probability**

This distinction matters most for `video_finetuned` TAU, where predictions cluster near 0.02. **Those are high-confidence negative predictions** (confidence 0.98), not low-confidence predictions.

### The three statistically-supported confidence signals

Only three model/dataset combinations have gaps that are **both reliable** (n_high ≥ 20, n_low ≥ 20) **and** with CIs excluding zero:

| Model/Dataset | Variant | Gap | n_high | n_low | CI_clip | CI_video |
|---|---|---:|---:|---:|---|---|
| **feature_fusion_audio_only MAVD** | raw | +0.387 | 343 | 75 | [+0.265, +0.506] | [+0.293, +0.484] |
| **feature_fusion_audio_only MAVD** | calibrated | +0.353 | 59 | 263 | [+0.242, +0.447] | [+0.220, +0.462] |
| **video_finetuned TAU** | raw | +0.333 | 363 | 87 | [+0.222, +0.448] | — (no MAVD video IDs) |

**For the two MAVD findings**, the clip-level and video-level CIs agree;
the signals are robust to clustering; the signs do not flip. **The TAU
`video_finetuned` finding has clip-level evidence only** — TAU has no
source-video grouping, so no video-level bootstrap applies.

### The confidence-usefulness answer

**`feature_fusion_audio_only` on MAVD is the only model with both:**
- Reliable confidence usefulness (raw +0.387, calibrated +0.353, n_high≥59)
- Best MAVD discrimination (ROC-AUC 0.704)
- Good calibration after scaling (ECE 0.107)
- Signal preserved across both raw and calibrated variants

**This is the single model where all three properties hold with statistical support.**

**`video_finetuned` on TAU has the strongest confidence signal of any model** (+0.333, n_high=363) but no valid calibrator exists.

### Sample-inadequacy findings

Several apparent "confidence signals" are artifacts of small high-confidence groups:

| Model/Dataset | n_high | n_low | Gap | Status |
|---|---:|---:|---:|---|
| ast_audio MAVD raw | **1** | 340 | +0.412 | **n/a (single clip)** |
| ast_audio MAVD cal | 12 | 304 | +0.302 | n/a (12 clips) |
| feature_fusion_mlp MAVD cal | **2** | 384 | −0.076 | n/a (2 clips) |
| feature_fusion_video_only MAVD cal | **0** | 418 | n/a | empty group |
| late_fusion_fromscratch TAU | **2** | 386 | −0.536 | n/a (2 clips) |
| ast_audio TAU cal | 7 | 345 | +0.458 | n/a (7 clips) |
| feature_fusion_audio_only TAU cal | 14 | 310 | −0.048 | n/a (14 clips) |
| feature_fusion_mlp TAU cal | **0** | 421 | n/a | empty group |
| feature_fusion_video_only TAU cal | **0** | 462 | n/a | empty group |

**AST audio's MAVD "confidence signal" (+0.412) is a single clip.** The n_high=1 caveat is in the JSON. **This is why sample-count reporting matters** — without it, the +0.412 would look like a strong finding.

### Risk-coverage invariance

**For TAU (single global temperature):** raw and calibrated risk-coverage curves are **exactly equivalent** (max accuracy diff = 0.0 across all models). Confirms monotonicity.

**For MAVD (per-fold temperatures):** curves differ (max accuracy diff = 0.035–0.517). Different T's applied to different folds can rearrange cross-fold ranking. This is **a property of the leakage-safe per-fold protocol, not a bug**. Documented in the JSON as a fold-specific calibration ranking diagnostic.

**The correct interpretation:** risk-coverage measures confidence usefulness under a sorted-by-confidence criterion. TAU calibration preserves this property exactly. MAVD per-fold calibration changes it because the T varies across folds.

### The confidence-usefulness answer

**Only one model is simultaneously:**
- Well-calibrated (ECE ≤ 0.15)
- Well-discriminating (ROC-AUC ≥ 0.70)
- Usefully confident (reliable gap ≥ 0.30, CI excludes zero, n_high ≥ 20)

**That model is `feature_fusion_audio_only` on MAVD after temperature
scaling.** It is the deployment candidate **on the MAVD three-property
evaluation only**. Its weak TAU transfer (F1 0.3154, ROC-AUC 0.6158,
AP 0.2656) prevents a broader deployment claim.

## The complete three-property table

| Model | Cal ECE | Discrim ROC-AUC | Confidence gap (reliable?) |
|---|---:|---:|---|
| **feature_fusion_audio_only MAVD (cal)** | **0.107** | **0.704** | **+0.353 ✅** |
| video_finetuned TAU (raw) | 0.110 | 0.634 | +0.333 ✅ |
| ast_audio MAVD (cal) | 0.154 | 0.680 | +0.302 ❌ (n_high=12) |
| feature_fusion_mlp MAVD (raw) | 0.280 | 0.660 | +0.262 ✅ |
| feature_fusion_video_only MAVD (raw) | 0.274 | 0.579 | +0.159 ✅ |
| feature_fusion_mlp MAVD (cal) | 0.218 | 0.660 | −0.076 ❌ (n_high=2) |
| ast_audio TAU (raw) | 0.328 | 0.557 | n/a (n_high=0) |
| feature_fusion_audio_only TAU (raw) | 0.331 | 0.606 | +0.119 ✅ (CI barely) |
| feature_fusion_mlp TAU (raw) | 0.321 | 0.616 | +0.109 ❌ (CI crosses zero) |
| feature_fusion_video_only TAU (raw) | 0.384 | 0.578 | +0.034 ❌ (CI crosses zero) |

**Only one model row has all three columns in good shape.** `feature_fusion_audio_only` MAVD (calibrated) is the only model that wins on calibration, discrimination, and confidence usefulness with statistically-supported evidence.

## Known limitations

1. **Provenance contamination on 3 of 7 MAVD models.** `video_finetuned` encoder was fine-tuned on all 540 MAVD clips; two late-fusion models use its predictions as input. Their MAVD calibration/confidence results are excluded. Their TAU results are honest and reported.

2. **Missing evidence gap.** `late_fusion_ast` TAU file (`fusion_tau_external_kinetics.csv`) was never regenerated after 9c-ext-1b. Not reconstructed — no approximation from other models.

3. **Small high-confidence groups limit some claims.** Multiple models have n_high < 20 in at least one variant. Gaps and CIs are reported as `n/a` where sample is inadequate. **Reliable=False does not mean the gap is zero; it means the estimate is unstable.**

4. **MAVD-fit temperature does not transfer to TAU.** All models improve ECE by only 0.003–0.030 on TAU. The residual TAU miscalibration is base-rate shift (25.6% → 15.0%), not fixable by monotone probability scaling.

5. **Per-fold calibration is not equivalent to global calibration.** MAVD's per-fold temperatures change cross-fold ranking (accuracy diff 0.035–0.517). TAU's single global temperature preserves it exactly (0.0). Both are valid protocols; they answer different questions.

6. **Video-level bootstrap was added in W10.3.2** but only for MAVD (TAU has no source-video grouping — each clip is a distinct recording). Clip-level, video-level, and location-level CIs are all reported. The location-level CI with 5 clusters remains exploratory.

7. **Deferred to future work:** Platt scaling / isotonic regression (would need TAU-labeled calibration data we don't use), conformal prediction (requires exchangeability assumption that TAU violates), calibration on the METR-LA regressors (different problem).

## What this demonstrates

- **Three independent properties measured correctly.** Calibration, discrimination, and confidence usefulness are documented separately with proper definitions.
- **Confidence definition matters.** `max(p, 1-p)` vs `p` is not a cosmetic choice — `video_finetuned`'s 0.02 predictions are high-confidence negatives, not low-confidence predictions.
- **Sample adequacy is checked.** All gaps report `n_high`, `n_low`, and `gap_reliable`. Several apparent findings were refuted as small-sample artifacts (AST audio MAVD: n_high=1).
- **Monotonicity is verified.** TAU risk-coverage curves exactly preserved under global temperature scaling; MAVD per-fold behavior differs and is documented as expected.
- **Cluster-aware uncertainty.** Video-level bootstrap added (W10.3.2). All three CI levels reported. The significant findings hold under all three.
- **Honest negative findings.** The MAVD-fit temperature doesn't transfer to TAU. Several apparent confidence signals are sample-inadequate. AST audio's TAU failure is a discrimination failure, not a calibration failure.
- **Cross-dataset transfer is measured, not assumed.** MAVD↔TAU calibration gap quantified as 0.04–0.17 ECE. TAU remains untouched as external test (no fitting on TAU labels).

## What this model card is NOT a demonstration of

- That temperature scaling is a universal fix for miscalibration.
- That the model's confidence is useful on TAU after calibration.
- That the MAVD-fit temperature generalizes.
- That calibration on MAVD implies calibration on TAU.
- That the AST encoder's near-chance TAU discrimination can be fixed by probability rescaling.
- That the project's fusion story changes because of the calibration findings.

## Reproduce

```powershell
# W10.1 — Calibration measurement
uv run python scripts/evaluate_calibration.py

# W10.2 — Temperature scaling
uv run python scripts/calibrate_with_temperature.py

# W10.3 — Confidence usefulness (W10.3.1 + W10.3.2 combined)
uv run python scripts/analyze_confidence.py
```

**Outputs:**
- `data/urbansas_results/calibration_metrics.json`
- `data/urbansas_results/calibration_temperature_results.json`
- `data/urbansas_results/confidence_analysis.json`
- `data/urbansas_results/calibration_plots/` (11 W10.1 + 8 W10.2 + 36 W10.3 = 55 plots)

## MLflow

No new MLflow runs. All W10 analyses are read-only measurements of the Week 9c predictions; W10.2 fits a single scalar (temperature) per fold but does not produce new models.

## Citation

% TODO: Replace with exact citations from CITATIONS.md before final portfolio polish.

```bibtex
@inproceedings{guo2017calibration,
  title={On Calibration of Modern Neural Networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle={ICML},
  year={2017}
}
```