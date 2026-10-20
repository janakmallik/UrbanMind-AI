"""Train a small 3D CNN on 15-class HMDB-51 urban behavior recognition.

15-class flat model with per-category evaluation.

Classes: stand, walk, sit, eat, drink, hug, run, ride_bike,
         ride_horse, fall_floor, push, hit, punch, shoot_gun, smoke

Categories (for reporting):
  normal: stand, walk, sit, eat, drink, hug
  mobility: run, ride_bike
  anomaly: ride_horse, fall_floor
  threat: push, hit, punch, shoot_gun
  restricted: smoke

Uses official split 1 (70 train / 30 test per class, val carved
from train 15%). Checkpoint selection: val macro-F1.
"""

from __future__ import annotations

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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from urbanintel.data.hmdb51 import CATEGORIES, CLASSES, NUM_CLASSES, prepare_hmdb51
from urbanintel.data.hmdb51_video import HMDB51VideoDataset

EXPERIMENT_NAME = "urbanintel-week7"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Tiny3DCNN(nn.Module):
    """Small 3D CNN: 4 conv blocks + global avg pool + linear head."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=16, C=3, H=112, W=112) -> permute to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        h = self.features(x)
        h = h.flatten(1)
        h = self.dropout(h)
        return self.head(h)


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            logits = model(frames)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def per_category_accuracy(y_true, y_pred):
    """Accuracy within each category, treating prediction as correct only
    if it matches the exact class (not just the category)."""
    results = {}
    for cat, class_names in CATEGORIES.items():
        class_idxs = [CLASSES.index(c) for c in class_names]
        mask = np.isin(y_true, class_idxs)
        if mask.sum() == 0:
            results[cat] = (0, 0, 0.0)
            continue
        correct = (y_true[mask] == y_pred[mask]).sum()
        total = mask.sum()
        results[cat] = (int(correct), int(total), correct / total)
    return results


import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nLoading HMDB-51 split (15 classes) ...")
    split = prepare_hmdb51(CLASSES, split_id=1)

    # ---- Baselines ----
    from collections import Counter

    print("\n=== Baselines ===")
    majority_class = Counter(split.train_df["label"]).most_common(1)[0][0]
    majority_preds = np.full(len(split.test_df), majority_class)
    majority_acc = accuracy_score(split.test_df["label"].values, majority_preds)
    majority_f1 = f1_score(split.test_df["label"].values, majority_preds, average="macro")
    print(f"Majority class ({CLASSES[majority_class]}):")
    print(f"  Test accuracy: {majority_acc:.4f}")
    print(f"  Test macro-F1: {majority_f1:.4f}")

    np.random.seed(args.seed)
    random_preds = np.random.randint(0, NUM_CLASSES, size=len(split.test_df))
    random_acc = accuracy_score(split.test_df["label"].values, random_preds)
    random_f1 = f1_score(split.test_df["label"].values, random_preds, average="macro")
    print(f"\nRandom guess:")
    print(f"  Test accuracy: {random_acc:.4f}")
    print(f"  Test macro-F1: {random_f1:.4f}")
    print()

    # ---- Datasets ----
    print("\nFiltering corrupt videos (one-time scan) ...")
    from urbanintel.data.hmdb51_video import filter_corrupt_videos

    train_df = filter_corrupt_videos(split.train_df)
    val_df = filter_corrupt_videos(split.val_df)
    test_df = filter_corrupt_videos(split.test_df)

    train_ds = HMDB51VideoDataset(train_df, is_train=True)
    val_ds = HMDB51VideoDataset(val_df, is_train=False)
    test_ds = HMDB51VideoDataset(test_df, is_train=False)

    loader_opts = {"num_workers": NUM_WORKERS, "pin_memory": (DEVICE == "cuda")}
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, **loader_opts
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_opts)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_opts)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = Tiny3DCNN(NUM_CLASSES).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Class weights for imbalance (each class ~59-60 in train, nearly balanced,
    # but weight anyway in case of slight imbalance)
    counts = np.array([(split.train_df["label"] == i).sum() for i in range(NUM_CLASSES)])
    assert (counts > 0).all(), f"Some classes have 0 training samples: {counts}"
    weights = (counts.sum() / (NUM_CLASSES * counts)).astype(np.float32)
    class_weights = torch.tensor(weights, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_path = _REPO_ROOT / "hmdb51_3dcnn_best.pt"

    with mlflow.start_run(run_name="hmdb51-3dcnn-15class"):
        mlflow.log_params(
            {
                "model": "Tiny3DCNN",
                "num_classes": NUM_CLASSES,
                "classes": str(CLASSES),
                "epochs": EPOCHS,
                "batch_size": 8,
                "lr": 3e-4,
                "n_frames": 16,
                "resolution": 112,
                "train_clips": len(train_ds),
                "val_clips": len(val_ds),
                "test_clips": len(test_ds),
                "n_params": n_params,
                "train_clips_after_filter": len(train_df),
                "val_clips_after_filter": len(val_df),
                "test_clips_after_filter": len(test_df),
                "corrupt_train_dropped": len(split.train_df) - len(train_df),
                "corrupt_val_dropped": len(split.val_df) - len(val_df),
                "corrupt_test_dropped": len(split.test_df) - len(test_df),
            }
        )
        mlflow.log_metrics(
            {
                "baseline_majority_accuracy": majority_acc,
                "baseline_majority_f1": majority_f1,
                "baseline_random_accuracy": random_acc,
                "baseline_random_f1": random_f1,
            }
        )

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss, n_batches = 0.0, 0
            t0 = time.time()

            for frames, labels in train_loader:
                frames = frames.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        logits = model(frames)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(frames)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss = total_loss / max(n_batches, 1)
            train_time = time.time() - t0

            y_val, p_val = evaluate(model, val_loader, DEVICE, NUM_CLASSES)
            val_acc = float(accuracy_score(y_val, p_val))
            val_f1 = float(f1_score(y_val, p_val, average="macro"))

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
                f"Epoch {epoch:2d}/{EPOCHS}  loss={train_loss:.4f}  "
                f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  ({train_time:.0f}s)"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, best_path)

        if best_state is None:
            raise RuntimeError("No checkpoint saved — investigate.")

        print(f"\nBest epoch: {best_epoch} (val_f1={best_val_f1:.4f})")
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_f1_macro", best_val_f1)
        mlflow.log_artifact(str(best_path))

        # Test with best checkpoint
        model.load_state_dict(torch.load(best_path, weights_only=True))
        y_test, p_test = evaluate(model, test_loader, DEVICE, NUM_CLASSES)
        test_acc = float(accuracy_score(y_test, p_test))
        test_f1 = float(f1_score(y_test, p_test, average="macro"))
        report = classification_report(
            y_test,
            p_test,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASSES,
            zero_division=0,
        )
        mlflow.log_metrics({"test_accuracy": test_acc, "test_f1_macro": test_f1})
        mlflow.log_text(report, "classification_report.txt")

        # Per-category accuracy
        print(f"\n=== Test results ===")
        print(f"Overall accuracy:  {test_acc:.4f}")
        print(f"Overall macro-F1:  {test_f1:.4f}")
        print(f"\nPer-category accuracy:")

        cat_results = per_category_accuracy(y_test, p_test)
        for cat, (correct, total, acc) in cat_results.items():
            print(f"  {cat:<12}  {correct:>3}/{total:<3} = {acc:.1%}")
            mlflow.log_metric(f"test_cat_{cat}_accuracy", acc)

        print(f"\nPer-class report saved to classification_report.txt")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
