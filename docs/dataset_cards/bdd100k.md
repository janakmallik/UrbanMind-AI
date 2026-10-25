# Dataset Card: BDD100K (Urban Scene Classification Subset)

## Identity

| Field | Value |
|---|---|
| Source | BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning |
| Original paper | Yu et al., CVPR 2020 |
| Official site | https://bdd-data.berkeley.edu/ |
| Kaggle mirror (images) | https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k |
| Local labels path | `<URBANINTEL_DATA_ROOT>/bdd100k/labels/` |
| Local images path | `<URBANINTEL_DATA_ROOT>/bdd100k/images/100k/` |
| License | UC Berkeley Regents License (research use) |

## Purpose

Classify urban driving scenes into one of three categories based on a
single keyframe image:
- **city street** — dense urban environment with buildings and traffic
- **highway** — multi-lane high-speed roads
- **residential** — low-density neighborhood streets

This is Week 5's image-modality task for UrbanMind-AI: fine-tuning
pretrained vision models on real, imbalanced urban imagery.

## Why Only 3 Classes

The full BDD100K scene attribute has 7 values, but the audit reveals
extreme class imbalance in the labeled keyframes:

| Class | Train | Val | % of Train | Usable? |
|---|---:|---:|---:|---|
| city street | 43,516 | 6,112 | 62.3% | ✅ |
| highway | 17,379 | 2,499 | 24.9% | ✅ |
| residential | 8,074 | 1,253 | 11.6% | ✅ |
| parking lot | 377 | 49 | 0.54% | ❌ Too few |
| undefined | 361 | 53 | 0.52% | ❌ Not a scene |
| tunnel | 129 | 27 | 0.18% | ❌ Too few |
| gas stations | 27 | 7 | 0.04% | ❌ Too few |

Only the top 3 classes have enough samples to train and evaluate a
classifier with meaningful per-class metrics. The others are documented
and excluded.

## Data Format

- **Images**: JPEG keyframes, 1280×720 pixels, extracted at the 10th
  second of each 40-second drive video
- **Labels**: JSON, one record per image, with scene/weather/timeofday
  attributes
- **Total in scope (3 classes)**: 68,969 train + 9,864 val

## Split (Week 5)

We use the official train/val split, then divide val 50/50 into val and
test to hold out a genuinely unseen evaluation set.

| Partition | Count | Source |
|---|---:|---|
| Train | 68,969 | Official train, filtered to 3 classes |
| Val | ~4,932 | Half of official val, stratified |
| Test | ~4,932 | Other half of official val, stratified |
| Random seed | 42 | |

## Class Distribution

- Imbalanced: 62% / 25% / 12%
- Evaluation must report **macro-F1** alongside accuracy
- Majority-class baseline: **62.3% accuracy**, poor macro-F1

## Baselines

Every model must beat:
1. **Majority class** (predict "city street" always) — establishes floor
2. **Frozen ResNet-50 features + linear classifier** — real baseline

## Planned Models

- ResNet-50 (ImageNet-pretrained), fine-tuned
- ViT-B/16 (optional, if ResNet pipeline succeeds)

## Known Challenges

1. **Class imbalance** — city street dominates
2. **Duplicate frames** — BDD100K keyframes are sampled from videos,
   so near-identical images may appear across splits
3. **Domain specificity** — models trained here may not transfer to other
   cities or lighting conditions
4. **No public test labels** — we hold out half of val

## Reproducibility

```powershell
uv run python scripts\inspect_bdd100k_labels.py
uv run python scripts\train_bdd100k_classifier.py
```

## Citation

```
Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., & Darrell, T. (2020).
BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning.
The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
```

## Changelog

| Date | Notes |
|---|---|
| 2026-10-17 | Initial dataset card. Scene label audit complete. |

---

## Week 9a — Derived fusion features

Week 9a extracted 14 numeric features per image from three pretrained
models processing the same BDD100K images:

| Feature group | Source model | Features | Preprocessing |
|---|---|---|---|
| ResNet probabilities | `bdd_resnet50_best.pt` | 3 class probs | Resize(256) → CenterCrop(224) → ImageNet norm |
| SegFormer pixel shares | `bdd_segformer_b0_best.pt` | 5 class shares | Resize(512×512) → ImageNet norm (custom numpy) |
| YOLO detection counts | `yolov8n.pt` (COCO pretrained) | 6 class counts | Ultralytics default |

**Extracted on:**
- 4,727 clean train images (ResNet val split, minus 205 contaminated)
- 4,745 clean test images (ResNet test split, minus 187 contaminated)

**Contamination filtering:** 187 test images and 205 val images were
excluded because they were in SegFormer's training set (`seg/images/train`).
This is documented to avoid upstream leakage (SegFormer features on those
images would be in-sample).

**Cached at:** `data/features/bdd100k_fusion/{train,test}.parquet`
(gitignored, ~1 MB each; reproducible via `scripts/extract_fusion_features.py`)

**Feature provenance:** the 4,727 train images are from ResNet's
*validation* split, not training. Overlap check against ResNet's training
IDs returned 0 matches. Residual concern: val split influenced checkpoint
selection (best-epoch); features are not fully pristine but are not
in-sample.

**Not committed:** the parquet files are gitignored (generated artifacts).
Only the extraction script and the derived metrics
(`data/fusion_results/session4_results.json`) are committed.