"""Fine-tune SegFormer-B0 on BDD100K 5-class semantic segmentation.

Classes: road (0), sidewalk (1), vehicle (2), person (3), other (4), ignore=255.

Methodological notes
--------------------
- Train: 6,000 images (official train, minus held-out test)
- Val:   1,000 images (official val) — used for checkpoint selection
- Test:  1,000 images (held out from official train) — evaluated once at end
- Loss: 0.5 * CE(class_weighted, ignore=255) + 0.5 * DiceLoss(ignore=255)
- Best checkpoint selected by validation mIoU
- Uses torch.amp for mixed precision
- Best checkpoint saved to disk + logged to MLflow
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB_PATH}"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

from urbanintel.config import dataset_path
from urbanintel.data.bdd100k_segmentation import (
    IGNORE_INDEX,
    NUM_CLASSES,
    OUR_CLASSES,
    load_mask,
    prepare_segmentation,
)

EXPERIMENT_NAME = "urbanintel-week6"
SEED = 42
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SEG_ROOT = dataset_path("bdd100k", "seg")


class SegDataset(Dataset):
    """BDD100K segmentation dataset."""

    def __init__(self, df, resolution: int = 512, augment: bool = False):
        self.image_paths = df["image_path"].tolist()
        self.mask_paths = df["mask_path"].tolist()
        self.resolution = resolution
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = load_mask(Path(self.mask_paths[idx]))

        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img = mask_img.resize((self.resolution, self.resolution), Image.NEAREST)
        mask = np.array(mask_img, dtype=np.int64)

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = (img_np - MEAN) / STD
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()
        mask_t = torch.from_numpy(mask).long()

        if self.augment and torch.rand(1).item() < 0.5:
            img_t = torch.flip(img_t, dims=[2])
            mask_t = torch.flip(mask_t, dims=[1])

        return img_t, mask_t


def dice_loss_per_class(logits, target, num_classes, smooth=1.0):
    """Per-class Dice loss, ignoring target==255 pixels."""
    probs = F.softmax(logits, dim=1)
    valid = (target != IGNORE_INDEX).unsqueeze(1).float()

    # One-hot target, clamp ignore to 0 then mask out
    t_clamped = target.clamp(0, num_classes - 1)
    target_oh = F.one_hot(t_clamped, num_classes).permute(0, 3, 1, 2).float()
    target_oh = target_oh * valid

    probs = probs * valid

    intersection = (probs * target_oh).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + target_oh.sum(dim=(0, 2, 3))

    dice_per_class = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_per_class.mean()


def compute_iou(preds, targets, num_classes):
    """Compute per-class IoU and mean IoU, ignoring IGNORE_INDEX."""
    ious = []
    for c in range(num_classes):
        pred_c = preds == c
        true_c = targets == c
        valid = targets != IGNORE_INDEX
        pred_c = pred_c & valid
        true_c = true_c & valid
        intersection = (pred_c & true_c).sum().item()
        union = (pred_c | true_c).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return ious


def upsample_logits(logits, target_h, target_w):
    """SegFormer outputs downsampled logits; upsample to input resolution."""
    return F.interpolate(logits, size=(target_h, target_w), mode="bilinear", align_corners=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--person-weight", type=float, default=20.0)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nLoading BDD100K segmentation split ...")
    split = prepare_segmentation(SEG_ROOT)
    print(f"Classes: {split.label_names}")
    print(f"Num classes: {split.num_classes}")

    if split.num_classes != NUM_CLASSES:
        raise ValueError(
            f"Class mismatch: loader has {split.num_classes} classes, "
            f"but script expects {NUM_CLASSES}. "
            f"Update BDD_TO_OURS in bdd100k_segmentation.py."
        )

    train_ds = SegDataset(split.train_df, resolution=args.resolution, augment=True)
    val_ds = SegDataset(split.val_df, resolution=args.resolution, augment=False)
    test_ds = SegDataset(split.test_df, resolution=args.resolution, augment=False)

    loader_opts = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_opts)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_opts)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_opts)

    print(f"\nTrain: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    # ---- Model ----
    print("Loading SegFormer-B0 (pretrained on ADE20k) ...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    # ---- Loss: class-weighted CE + Dice ----
    # road=1.0, sidewalk=2.0, vehicle=1.5, person=20.0, other=0.5
    class_weights = torch.tensor([1.0, 2.0, 1.5, args.person_weight, 0.5], device=device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    with mlflow.start_run(run_name="bdd-segformer-b0"):
        mlflow.log_params(
            {
                "model": "SegFormer-B0 (ADE20k pretrained)",
                "num_classes": NUM_CLASSES,
                "class_names": str(OUR_CLASSES),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "resolution": args.resolution,
                "person_weight": args.person_weight,
                "loss": "0.5 * CE(class_weighted) + 0.5 * Dice",
                "train_rows": len(train_ds),
                "val_rows": len(val_ds),
                "test_rows": len(test_ds),
                "test_source": "held out from official train",
            }
        )

        best_val_miou = -1.0
        best_epoch = 0
        best_state = None
        best_path = _REPO_ROOT / "bdd_segformer_b0_best.pt"

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for imgs, masks in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        out = model(pixel_values=imgs)
                        logits = upsample_logits(out.logits, masks.shape[1], masks.shape[2])
                        loss_ce = ce_loss(logits, masks)
                        loss_dice = dice_loss_per_class(logits, masks, NUM_CLASSES)
                        loss = 0.5 * loss_ce + 0.5 * loss_dice
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(pixel_values=imgs)
                    logits = upsample_logits(out.logits, masks.shape[1], masks.shape[2])
                    loss_ce = ce_loss(logits, masks)
                    loss_dice = dice_loss_per_class(logits, masks, NUM_CLASSES)
                    loss = 0.5 * loss_ce + 0.5 * loss_dice
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss = total_loss / max(n_batches, 1)
            train_time = time.time() - t0

            # ---- Validate ----
            model.eval()
            iou_per_class_sum = np.zeros(NUM_CLASSES)
            iou_class_count = np.zeros(NUM_CLASSES)
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    out = model(pixel_values=imgs)
                    logits = upsample_logits(out.logits, masks.shape[1], masks.shape[2])
                    preds = logits.argmax(dim=1)
                    ious = compute_iou(preds, masks, NUM_CLASSES)
                    for c in range(NUM_CLASSES):
                        if not np.isnan(ious[c]):
                            iou_per_class_sum[c] += ious[c]
                            iou_class_count[c] += 1

            per_class_iou = np.where(
                iou_class_count > 0,
                iou_per_class_sum / np.maximum(iou_class_count, 1),
                0.0,
            )
            val_miou = float(per_class_iou.mean())

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_miou": val_miou,
                    "epoch_time_sec": train_time,
                    **{
                        f"val_iou_{name}": float(per_class_iou[i])
                        for i, name in enumerate(OUR_CLASSES)
                    },
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
                f"val_mIoU={val_miou:.4f}  ({train_time:.0f}s)"
            )
            print(
                "  per-class IoU: "
                + "  ".join(f"{OUR_CLASSES[i]}={per_class_iou[i]:.3f}" for i in range(NUM_CLASSES))
            )

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, best_path)
                print(f"  -> saved best checkpoint (val_mIoU={val_miou:.4f})")

        if best_state is None:
            raise RuntimeError("No checkpoint was saved — investigate training.")

        print(f"\nBest epoch: {best_epoch} (val_mIoU={best_val_miou:.4f})")
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_miou", best_val_miou)
        mlflow.log_artifact(str(best_path))

        # ---- Test (once, with best checkpoint) ----
        model.load_state_dict(torch.load(best_path, weights_only=True))
        model.eval()
        iou_per_class_sum = np.zeros(NUM_CLASSES)
        iou_class_count = np.zeros(NUM_CLASSES)
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                out = model(pixel_values=imgs)
                logits = upsample_logits(out.logits, masks.shape[1], masks.shape[2])
                preds = logits.argmax(dim=1)
                ious = compute_iou(preds, masks, NUM_CLASSES)
                for c in range(NUM_CLASSES):
                    if not np.isnan(ious[c]):
                        iou_per_class_sum[c] += ious[c]
                        iou_class_count[c] += 1

        per_class_iou = np.where(
            iou_class_count > 0,
            iou_per_class_sum / np.maximum(iou_class_count, 1),
            0.0,
        )
        test_miou = float(per_class_iou.mean())

        mlflow.log_metric("test_miou", test_miou)
        for i, name in enumerate(OUR_CLASSES):
            mlflow.log_metric(f"test_iou_{name}", float(per_class_iou[i]))

        print(f"\nTest mIoU: {test_miou:.4f}")
        print("Per-class IoU:")
        for i, name in enumerate(OUR_CLASSES):
            print(f"  {name:<10} {per_class_iou[i]:.4f}")


if __name__ == "__main__":
    main()
