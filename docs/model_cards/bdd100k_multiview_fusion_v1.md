# Model Card: BDD100K Multi-View Fusion (v1)

## What this model does

Fuses three pretrained vision models that process the same image into a
single scene classifier:

| Feature group | Source model | Features |
|---|---|---|
| ResNet-50 probabilities | `bdd_resnet50_best.pt` (Week 5) | 3 class probs (city/highway/residential) |
| SegFormer-B0 pixel shares | `bdd_segformer_b0_best.pt` (Week 6) | 5 class shares (road/sidewalk/vehicle/person/other) |
| YOLOv8n detection counts | `yolov8n.pt` (COCO pretrained) | 6 class counts (person/car/bus/truck/traffic_light/stop_sign) |

**Task:** Predict urban scene class (`city street` / `highway` /
`residential`) from the 14 concatenated features via a LightGBM stacker.

**This is multi-view fusion on one modality (images).** Three different
architectures process the same RGB pixels. It is not cross-modal fusion.

## What this model does NOT do

- **Not cross-modal.** All three models process images only. No audio,
  text, or time series is fused.
- **Not better than the primary model on overall accuracy.** Fusion test
  accuracy (0.7372) is below the ResNet checkpoint alone (0.7819).
- **Not robust to missing modalities.** All three feature groups are
  required at inference time.
- **Not production-ready.** Trained on 4,727 images, tested on 4,745.
  Small dataset; results are exploratory.
- **Not evaluated on a truly untouched test set.** The test set was used
  to evaluate 7 fusion configurations × 3 seeds + 1 ResNet checkpoint = 22
  evaluations. Reporting mean ± std is honest, but this is not a clean
  held-out benchmark.

## Architecture

- **Type:** Feature-level fusion via LightGBM gradient-boosted classifier
- **Input:** 14 numeric features per image
- **Model:** `lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, class_weight="balanced")`
- **Output:** 3-class scene label

## Training

- **Training data:** 4,727 clean images from ResNet's validation split
  (BDD100K `labels/100k/val`, half held out by seed=42)
- **Test data:** 4,745 clean images from ResNet's test split (the other
  half of `labels/100k/val`)
- **Contamination filtering:** 187 images excluded from ResNet's test
  split because they were in SegFormer's training set (`seg/images/train`).
  205 images excluded from ResNet's val split for the same reason.
- **Feature provenance:** The 4,727 fusion-train images are from ResNet's
  *validation* split, not its training split. Overlap check against
  ResNet's training image IDs returned **zero matches**. ResNet
  probabilities on these images are out-of-sample (not memorized from
  training). Residual concern: ResNet's validation split influenced
  checkpoint selection (best-epoch), so features are not fully pristine.
- **Protocol:** 5-fold stratified CV on train for model comparison; single
  test evaluation per model configuration × 3 seeds (42, 123, 7). Mean ±
  std across seeds reported.
- **Class imbalance handling:** `class_weight="balanced"` in LightGBM.

## Results

### Overall comparison

| Model | Features | Test acc | Test macro-F1 |
|---|---:|---:|---:|
| ResNet-only stacker | 3 | 0.7043 ± 0.000 | 0.6662 ± 0.000 |
| SegFormer-only stacker | 5 | 0.5355 ± 0.000 | 0.4790 ± 0.000 |
| YOLO-only stacker | 6 | 0.4872 ± 0.000 | 0.4562 ± 0.000 |
| **Full fusion** | **14** | **0.7372 ± 0.000** | **0.6910 ± 0.000** |
| Ablation: no ResNet | 11 | 0.5903 ± 0.000 | 0.5325 ± 0.000 |
| Ablation: no SegFormer | 9 | 0.7216 ± 0.000 | 0.6804 ± 0.000 |
| Ablation: no YOLO | 8 | 0.7357 ± 0.000 | 0.6899 ± 0.000 |
| **Original ResNet (checkpoint)** | **—** | **0.7819** | **0.6986** |

### Per-class comparison (clean_test, n=4,745)

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| **Fusion** | city street | 0.85 | 0.73 | 0.79 | 2945 |
| **Fusion** | highway | 0.68 | 0.78 | 0.72 | 1209 |
| **Fusion** | residential | 0.48 | **0.67** | 0.56 | 591 |
| **ResNet** | city street | 0.80 | 0.88 | 0.84 | 2945 |
| **ResNet** | highway | 0.74 | 0.72 | 0.73 | 1209 |
| **ResNet** | residential | 0.74 | **0.41** | 0.53 | 591 |

### Bootstrap CI on macro-F1 difference

- **Point estimate (fusion − ResNet):** −0.0076
- **95% bootstrap CI:** [−0.0229, +0.0091]
- **P(fusion > ResNet):** 0.191
- **Method:** 1,000 bootstrap resamples, paired (same test images for both
  models), percentile interval

**Interpretation:** The 95% CI includes zero. Fusion and ResNet are not
statistically distinguishable in overall macro-F1 on this test set. The
point estimate favors ResNet slightly.

## Known limitations

1. **No overall improvement.** Fusion macro-F1 is −0.0076 below the
   original ResNet. Whether this reflects true inferiority or noise is
   unresolved by the current evidence (CI includes zero).

2. **Class trade-off.** Fusion raises residential recall (+0.26) but
   lowers overall accuracy (−0.045) and city-street recall (−0.15). The
   fusion model is more willing to predict residential, at the cost of
   city-street accuracy.

3. **Test set reuse.** 22 test evaluations (7 models × 3 seeds + ResNet).
   Results are exploratory. A truly untouched test evaluation would
   require a separate held-out split.

4. **Feature provenance caveat.** ResNet features are out-of-sample from
   training, but were influenced by checkpoint selection (val split). A
   fully disjoint triple-split would eliminate this, but requires more
   data than available.

5. **Ablation shows YOLO is redundant.** Removing YOLO changes F1 by
   −0.001 — indistinguishable from noise. YOLO's detection counts overlap
   with ResNet's learned visual features.

6. **SegFormer contributes marginally.** Removing SegFormer changes F1 by
   −0.011 — small. The residential-recall improvement appears to come
   primarily from SegFormer's pixel-share features nudging the decision
   boundary toward residential, but a formal attribution analysis would
   be needed to confirm.

7. **Same-modality fusion.** All three feature sources are images. Not
   cross-modal. Different from image + text or video + audio fusion.

## What this model demonstrates

- **Multi-view fusion infrastructure:** feature extraction from frozen
  models, clean split verification, contamination filtering, comparison
  against the original model and against ablations.
- **Honest negative result:** fusion does not improve overall accuracy.
  The residential-recall finding is a trade-off, not a win.
- **Ablation reasoning:** identifies which modality contributes and which
  is redundant.
- **Statistical rigor:** bootstrap CI, multi-seed averaging, per-class
  breakdown.
- **Accurate framing:** "multi-view fusion" is not "multimodal fusion."

## What this model does NOT demonstrate

- Cross-modal multimodal fusion
- Fusion improvement over the best unimodal model
- That the residential-recall trade-off is desirable (depends on the
  application's cost function)

## Reproduce

```powershell
# Session 1 — audit + split reconstruction
uv run python scripts/reconstruct_resnet_split.py
uv run python scripts/verify_resnet_split.py

# Session 2 — contamination filtering
uv run python scripts/prepare_fusion_data.py

# Session 3 — feature extraction (~20 min)
uv run python scripts/extract_fusion_features.py

# Session 4 — training + evaluation (~5 min)
uv run python scripts/train_bdd100k_fusion.py