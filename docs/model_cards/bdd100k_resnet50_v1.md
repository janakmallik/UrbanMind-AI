# Model Card: BDD100K Scene Classifier (ResNet-50 v1)

## What this model does

Classifies an urban driving keyframe image into one of three categories:
city street, highway, or residential.

## What this model does NOT do

- Does not classify from video frames in sequence (single-frame only)
- Does not handle the other 4 BDD100K scene classes (parking lot,
  undefined, tunnel, gas stations) — excluded for insufficient samples
- Not suitable for production — research prototype only

## Architecture

- Backbone: ResNet-50 (ImageNet-pretrained, IMAGENET1K_V2)
- Head: linear (2048 → 3)
- **Selective fine-tune:** only `layer4` and `fc` trainable
  (23,514,179 trainable parameters out of ~25.5M total)
- Earlier blocks (conv1, layer1, layer2, layer3) remain frozen

## Training

- Dataset: BDD100K scene subset (3 viable classes of 7)
- Train: 68,969 | Val: 4,932 | Test: 4,932
- Test split source: half of BDD100K's official *validation* annotations
- Optimizer: AdamW, lr=1e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR over 5 epochs
- Batch size 8, mixed precision (`torch.amp.autocast`)
- Augmentation: RandomResizedCrop(224, scale=(0.75, 1.0)),
  RandomHorizontalFlip
- Best-epoch checkpoint on validation macro-F1

## Results (test set)

| Model | Accuracy | F1 Macro |
|---|---:|---:|
| Majority class | 0.6196 | 0.2550 |
| Frozen ResNet-50 + LogReg | 0.7618 | 0.6780 |
| **Fine-tuned ResNet-50 (best epoch 2)** | **0.7800** | **0.6984** |

Delta over frozen baseline: +1.82 accuracy, +2.04 macro-F1.

Confusion matrix and per-class F1: see `urbanintel-week5` MLflow
experiment, run `bdd-resnet50-finetune`, artifact `classification_report.txt`.

## Limitations

- **Imbalanced classes** (63% city street / 25% highway / 12% residential)
- **Modest fine-tune gain** over frozen features
- **Test-set contamination:** the baseline script evaluated this test
  split first; metrics are a benchmark comparison, not clean
  out-of-sample
- **Not the official BDD100K test set** (no public scene labels exist)
- **Best epoch was epoch 2** — later epochs did not improve val F1
- **Not suitable for deployment** — research prototype

## Reproduce

```powershell
uv run python scripts/run_bdd100k_baselines.py
uv run python scripts/train_bdd100k_resnet50.py --epochs 5 --batch-size 8 --workers 4
```