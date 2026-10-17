"""Fine-tune ResNet-50 on BDD100K scene classification.

Methodological notes
--------------------
- Train split: used for training only.
- Validation split: used for checkpoint selection (best macro-F1).
- Test split: evaluated ONCE at the end, using the best checkpoint.
- The test split is half of BDD100K's official *validation* annotations;
  the official test set has no public scene labels and is not used.
- Baselines (majority class, frozen ResNet-50) were also evaluated on
  this test split. Their results were observed before fine-tuning.
  Therefore this is a consistent benchmark comparison, NOT a pristine
  out-of-sample evaluation.

Uses current PyTorch AMP API. Best checkpoint saved to disk and logged
to MLflow as an artifact.
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
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from urbanintel.config import dataset_path
from urbanintel.data.bdd100k import prepare_bdd100k

EXPERIMENT_NAME = "urbanintel-week5"
SEED = 42

LABEL_DIR = dataset_path("bdd100k", "labels")
IMAGES_ROOT = dataset_path("bdd100k", "images", "100k")

# Enable cuDNN autotuner for fixed-size convolutions.
# Safe when all images are the same size (224x224 after transforms).
torch.backends.cudnn.benchmark = True


class SceneDataset(Dataset):
    """Image dataset returning transformed image and label."""

    def __init__(self, df, transform):
        self.paths = df["image_path"].tolist()
        self.labels = df["label_id"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    """ResNet-50 with a fresh classification head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                param.requires_grad = False

    return model


def evaluate(model, loader, device):
    """Return (y_true, y_pred) arrays over the loader."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(labels), np.concatenate(preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nLoading BDD100K split ...")
    split = prepare_bdd100k(
        train_json=LABEL_DIR / "bdd100k_labels_images_train.json",
        val_json=LABEL_DIR / "bdd100k_labels_images_val.json",
        train_images_dir=IMAGES_ROOT / "train",
        val_images_dir=IMAGES_ROOT / "val",
    )
    class_names = split.label_names
    num_classes = len(class_names)

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = SceneDataset(split.train_df, train_tf)
    val_ds = SceneDataset(split.val_df, eval_tf)
    test_ds = SceneDataset(split.test_df, eval_tf)

    loader_opts = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    # drop_last=True on train: avoids BatchNorm instability on a 1-sample
    # final batch, at the cost of dropping up to batch_size-1 samples/epoch.
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_opts)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_opts)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_opts)

    model = build_model(num_classes, freeze_backbone=args.freeze_backbone).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    with mlflow.start_run(run_name="bdd-resnet50-finetune"):
        mlflow.log_params(
            {
                "model": "resnet50",
                "pretrained": "IMAGENET1K_V2",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "freeze_backbone": args.freeze_backbone,
                "trainable_params": trainable,
                "mixed_precision": use_amp,
                "train_rows": len(split.train_df),
                "val_rows": len(split.val_df),
                "test_rows": len(split.test_df),
                "label_names": str(class_names),
                "test_split_note": "half of BDD100K official val annotations",
                "test_contamination_note": (
                    "Baselines evaluated this test split first; "
                    "benchmark comparison only, not pristine out-of-sample"
                ),
            }
        )

        best_val_f1 = -1.0
        best_epoch = 0
        best_state = None
        best_path = _REPO_ROOT / "bdd_resnet50_best.pt"

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            n_batches = 0
            t0 = time.time()

            for imgs, labels in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        logits = model(imgs)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss = total_loss / max(n_batches, 1)
            train_time = time.time() - t0

            y_true, y_pred = evaluate(model, val_loader, device)
            val_acc = float(accuracy_score(y_true, y_pred))
            val_f1 = float(f1_score(y_true, y_pred, average="macro"))

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_accuracy": val_acc,
                    "val_f1_macro": val_f1,
                    "epoch_time_sec": train_time,
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  "
                f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  ({train_time:.0f}s)"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, best_path)
                print(f"  -> saved best checkpoint to {best_path}")

        if best_state is None:
            raise RuntimeError(
                "No checkpoint was saved. Every epoch had val_f1 <= -1 "
                "(should be impossible). Investigate before proceeding."
            )

        print(f"\nBest epoch: {best_epoch} (val_f1={best_val_f1:.4f})")
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_f1_macro", best_val_f1)
        mlflow.log_artifact(str(best_path))

        # Evaluate on test ONCE, with best checkpoint
        if not best_path.exists():
            raise FileNotFoundError(f"Expected checkpoint at {best_path}")
        model.load_state_dict(torch.load(best_path, weights_only=True))

        y_true, y_pred = evaluate(model, test_loader, device)
        test_acc = float(accuracy_score(y_true, y_pred))
        test_f1 = float(f1_score(y_true, y_pred, average="macro"))
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0,
        )
        mlflow.log_metrics({"test_accuracy": test_acc, "test_f1_macro": test_f1})
        mlflow.log_text(report, "classification_report.txt")

        print(f"\nTest accuracy:  {test_acc:.4f}")
        print(f"Test F1 macro:  {test_f1:.4f}")
        print("\n=== Comparison (test set) ===")
        print(f"{'Model':<30} {'Accuracy':>10} {'F1 Macro':>10}")
        print(f"{'Majority class':<30} {0.6196:>10.4f} {0.2550:>10.4f}")
        print(f"{'Frozen ResNet-50 + LogReg':<30} {0.7618:>10.4f} {0.6780:>10.4f}")
        print(f"{'Fine-tuned ResNet-50':<30} {test_acc:>10.4f} {test_f1:>10.4f}")


if __name__ == "__main__":
    main()
