"""Fine-tune R(2+1)D-18 (Kinetics-400 pretrained) on HMDB-51 15-class.

Compares against the Week 7 baseline (Tiny3DCNN, 25.33% test accuracy).

R(2+1)D-18 is a 3D ResNet variant: 2D convolutions for spatial features +
1D convolutions for temporal. Pretrained on Kinetics-400 (400 classes).
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
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from urbanintel.data.hmdb51 import CATEGORIES, CLASSES, NUM_CLASSES, prepare_hmdb51
from urbanintel.data.hmdb51_video import HMDB51VideoDataset, filter_corrupt_videos

EXPERIMENT_NAME = "urbanintel-week7"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def per_category_accuracy(y_true, y_pred):
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


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            logits = model(frames.permute(0, 2, 1, 3, 4))
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nLoading HMDB-51 split (15 classes) ...")
    split = prepare_hmdb51(CLASSES, split_id=1)

    print("\nFiltering corrupt videos ...")
    train_df = filter_corrupt_videos(split.train_df)
    val_df = filter_corrupt_videos(split.val_df)
    test_df = filter_corrupt_videos(split.test_df)

    train_ds = HMDB51VideoDataset(train_df, is_train=True, normalization="kinetics")
    val_ds = HMDB51VideoDataset(val_df, is_train=False, normalization="kinetics")
    test_ds = HMDB51VideoDataset(test_df, is_train=False, normalization="kinetics")

    loader_opts = {"num_workers": args.num_workers, "pin_memory": (DEVICE == "cuda")}
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_opts
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_opts)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_opts)

    print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # Load pretrained R(2+1)D-18, replace head
    print("Loading R(2+1)D-18 (Kinetics-400 pretrained) ...")
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    counts = np.array([(train_df["label"] == i).sum() for i in range(NUM_CLASSES)])
    assert (counts > 0).all(), f"Some classes have 0 training samples: {counts}"
    weights = (counts.sum() / (NUM_CLASSES * counts)).astype(np.float32)
    class_weights = torch.tensor(weights, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Lower LR for backbone, higher for head
    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.3},  # 30% of head LR
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_path = _REPO_ROOT / "hmdb51_r2plus1d_best.pt"

    with mlflow.start_run(run_name="hmdb51-r2plus1d-15class"):
        mlflow.log_params(
            {
                "model": "r2plus1d_18",
                "pretrained": "KINETICS400_V1",
                "num_classes": NUM_CLASSES,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "n_params": n_params,
                "train_clips": len(train_ds),
                "val_clips": len(val_ds),
                "test_clips": len(test_ds),
                "corrupt_train_dropped": len(split.train_df) - len(train_df),
                "corrupt_val_dropped": len(split.val_df) - len(val_df),
                "corrupt_test_dropped": len(split.test_df) - len(test_df),
            }
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, n_batches = 0.0, 0
            t0 = time.time()

            for frames, labels in train_loader:
                frames = frames.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        logits = model(frames.permute(0, 2, 1, 3, 4))  # (B,T,C,H,W) -> (B,C,T,H,W)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(frames.permute(0, 2, 1, 3, 4))
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss = total_loss / max(n_batches, 1)
            train_time = time.time() - t0

            y_val, p_val = evaluate(model, val_loader, DEVICE)

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
                f"Epoch {epoch:2d}/{args.epochs}  loss={train_loss:.4f}  "
                f"val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  ({train_time:.0f}s)"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, best_path)
                print(f"  -> saved best checkpoint (val_f1={val_f1:.4f})")

        if best_state is None:
            raise RuntimeError("No checkpoint saved.")

        print(f"\nBest epoch: {best_epoch} (val_f1={best_val_f1:.4f})")
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_f1_macro", best_val_f1)
        mlflow.log_artifact(str(best_path))

        # Load best checkpoint
        model.load_state_dict(torch.load(best_path, weights_only=True))

        # Single test evaluation
        y_test, p_test = evaluate(model, test_loader, DEVICE)
        test_acc = float(accuracy_score(y_test, p_test))
        test_f1 = float(f1_score(y_test, p_test, average="macro"))

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

        print(f"\n=== Test results ===")
        print(f"Overall accuracy:  {test_acc:.4f}")
        print(f"Overall macro-F1:  {test_f1:.4f}")
        print(f"\nPer-category accuracy:")

        cat_results = per_category_accuracy(y_test, p_test)
        for cat, (correct, total, acc) in cat_results.items():
            print(f"  {cat:<12}  {correct:>3}/{total:<3} = {acc:.1%}")
            mlflow.log_metric(f"test_cat_{cat}_accuracy", acc)

        print("\n=== Comparison with baseline ===")
        print(f"{'Model':<30} {'Test Acc':>10} {'Test F1':>10}")
        # Reference: Tiny3DCNN baseline from prior run (see docs/experiments.md Week 7)
        print(
            f"{'Tiny3DCNN (Week 7 baseline)':<30} {0.2533:>10.4f} {0.2364:>10.4f}  # from prior run"
        )
        print(f"{'R(2+1)D-18 (this run)':<30} {test_acc:>10.4f} {test_f1:>10.4f}")


if __name__ == "__main__":
    main()
