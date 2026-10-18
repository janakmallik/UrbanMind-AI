# Model Card: BDD100K 5-Class Segmentation (SegFormer-B0 v1)

## What this model does

Per-pixel semantic segmentation of urban driving images into 5 classes:
- **road** — drivable surface (BDD100K road + sidewalk)
- **sidewalk** — pedestrian walkway
- **vehicle** — car, truck, bus, train, motorcycle
- **person** — pedestrian, rider
- **other** — building, sky, vegetation, traffic signs, etc.

## Architecture

- **Model:** SegFormer-B0 (~3.7M parameters)
- **Pretraining:** ADE20k (150 classes)
- **Head:** new 5-class linear classifier (random init)
- **Input:** 512×512 RGB images
- **Output:** 512×512 per-pixel class predictions

## Training

- **Train:** 6,000 images (official BDD100K train, minus held-out test)
- **Val:** 1,000 images (official BDD100K val, for checkpoint selection)
- **Test:** 1,000 images (held out from official train)
- **Loss:** 0.5 × CE(class_weighted, ignore=255) + 0.5 × Dice
- **Class weights:** [1.0, 2.0, 1.5, 20.0, 0.5]
- **Optimizer:** AdamW, lr=6e-5, cosine annealing
- **Epochs:** 20
- **Batch size:** 8, mixed precision

## Results

### Overall

| Split | mIoU |
|---|---:|
| Validation (best epoch 19) | 0.7480 |
| Test | 0.7172 |

### Per-class test IoU

| Class | IoU | Pixel share |
|---|---:|---:|
| road | 0.9081 | 22.2% |
| sidewalk | 0.5377 | 1.6% |
| vehicle | 0.8538 | 9.6% |
| person | 0.3319 | 0.11% |
| other | 0.9547 | 46.5% |

## Limitations

- **Test split is held out from official train**, not the official BDD100K test (which has no public labels)
- **Person IoU (0.33) is the weakest** — the class is only 0.11% of pixels
- **Only 5 of BDD100K's 19 classes** are learned — the other 14 are merged into "other"
- **512×512 input resolution** — small objects (distant pedestrians, traffic signs) may be missed
- **Domain:** trained on US driving footage; not validated on other countries or conditions

## Reproduce

```powershell
uv run python scripts\train_bdd100k_segformer.py --epochs 20 --batch-size 8 --workers 4