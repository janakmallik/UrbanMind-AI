"""Train 1D-CNN on UrbanSound8K with 10-fold CV.

Uses the predefined `fold` column (1-10), as recommended by the dataset
creators. Each fold is used as test once, with the remaining 9 used for
training (with 15% carved for validation).

Reports mean ± std accuracy and macro-F1 across 10 folds.
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
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from urbanintel.data.urbansound8k import (
    N_MELS,
    NUM_CLASSES,
    URBANSOUND_CLASSES,
    CachedAudioDataset,
    load_dataframe,
)

EXPERIMENT_NAME = "urbanintel-week7-6"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class AudioCNN(nn.Module):
    def __init__(self, n_mels: int = N_MELS, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        h = self.conv(x).squeeze(-1)
        h = self.dropout(h)
        return self.head(h)


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            preds.append(model(x).argmax(dim=-1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(preds)


def train_fold(
    df: pd.DataFrame,
    fold: int,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    val_fraction: float,
    seed: int,
):
    """Train and evaluate on one fold. Returns (test_acc, test_f1, best_val_f1)."""
    from sklearn.model_selection import train_test_split

    test_idx = df.index[df["fold"] == fold].tolist()
    train_pool_idx = df.index[df["fold"] != fold].tolist()

    train_labels = df.loc[train_pool_idx, "classID"].tolist()
    train_idx, val_idx = train_test_split(
        train_pool_idx,
        test_size=val_fraction,
        random_state=seed,
        stratify=train_labels,
    )

    train_ds = CachedAudioDataset(train_idx, is_train=True)
    val_ds = CachedAudioDataset(val_idx, is_train=False)
    test_ds = CachedAudioDataset(test_idx, is_train=False)

    loader_opts = {"num_workers": num_workers, "pin_memory": (DEVICE == "cuda")}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_opts)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_opts)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_opts)

    torch.manual_seed(seed + fold)
    model = AudioCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # Class-weighted loss to handle imbalanced classes (gun_shot, car_horn)
    from collections import Counter

    class_counts = Counter(df["classID"].tolist())
    total = sum(class_counts.values())
    weights = [total / (NUM_CLASSES * class_counts.get(i, 1)) for i in range(NUM_CLASSES)]
    class_weights = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    best_state = None
    patience, bad = 3, 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        y_val, p_val = evaluate(model, val_loader)
        val_f1 = float(f1_score(y_val, p_val, average="macro"))
        if val_f1 > best_val_f1 + 0.005:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
        train_loss = total_loss / max(n_batches, 1)
        print(
            f"  fold {fold} epoch {epoch:2d}  loss={train_loss:.4f}  val_f1={val_f1:.4f}  ({time.time() - t0:.0f}s)"
        )

    model.load_state_dict(best_state)
    y_test, p_test = evaluate(model, test_loader)
    test_acc = float(accuracy_score(y_test, p_test))
    test_f1 = float(f1_score(y_test, p_test, average="macro"))

    # Save best-epoch checkpoint so the model can be reused for inference.
    # Only save for single-fold runs (avoids 10-fold overwrite confusion).
    # Week 7.7 fix: original training script discarded best_state after test.
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"urbansound8k_audio_cnn_fold{fold}.pt"
    torch.save(
        {
            "model": best_state,
            "fold": fold,
            "best_val_f1": best_val_f1,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "classes": URBANSOUND_CLASSES,
            "n_mels": N_MELS,
            "num_classes": NUM_CLASSES,
        },
        ckpt_path,
    )
    print(f"  Saved checkpoint: {ckpt_path}")

    return test_acc, test_f1, best_val_f1, len(train_idx), len(val_idx), len(test_idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--single-fold",
        type=int,
        default=None,
        help="If set, run only this fold (1-10) for smoke testing",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading UrbanSound8K parquet dataframe ...")
    df = load_dataframe()
    print(f"Total clips: {len(df)}")
    print(f"Folds: {sorted(df['fold'].unique())}")
    print(f"Class distribution:\n{df['classID'].value_counts().sort_index().to_string()}")

    results = []
    with mlflow.start_run(run_name="urbansound8k-10fold"):
        mlflow.log_params(
            {
                "model": "AudioCNN",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "n_mels": N_MELS,
                "val_fraction": args.val_fraction,
                "cv": "10-fold (predefined)",
            }
        )

        folds_to_run = [args.single_fold] if args.single_fold else list(range(1, 11))
        for fold in folds_to_run:
            print(f"\n=== Fold {fold} / 10 ===")
            acc, f1, val_f1, n_tr, n_va, n_te = train_fold(
                df,
                fold,
                args.epochs,
                args.batch_size,
                args.lr,
                args.num_workers,
                args.val_fraction,
                args.seed,
            )
            print(f"  FOLD {fold}: test_acc={acc:.4f}  test_f1={f1:.4f}  best_val_f1={val_f1:.4f}")
            mlflow.log_metrics(
                {
                    f"fold{fold}_test_acc": acc,
                    f"fold{fold}_test_f1": f1,
                    f"fold{fold}_train_size": n_tr,
                    f"fold{fold}_val_size": n_va,
                    f"fold{fold}_test_size": n_te,
                }
            )
            results.append({"fold": fold, "acc": acc, "f1": f1})

        accs = [r["acc"] for r in results]
        f1s = [r["f1"] for r in results]
        mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs)) if len(accs) > 1 else 0.0
        mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s)) if len(f1s) > 1 else 0.0
        print(f"\n=== {len(accs)}-fold CV summary ===")
        print(f"Test accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Test macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
        mlflow.log_metrics(
            {
                "cv_mean_accuracy": mean_acc,
                "cv_std_accuracy": std_acc,
                "cv_mean_f1_macro": mean_f1,
                "cv_std_f1_macro": std_f1,
            }
        )

        # If all 10 folds ran, save a summary checkpoint of the best fold.
        # For single-fold runs, that checkpoint was already saved inside train_fold.
        if args.single_fold is None and results:
            best_fold = max(results, key=lambda r: r["f1"])["fold"]
            src = Path("checkpoints") / f"urbansound8k_audio_cnn_fold{best_fold}.pt"
            dst = Path("checkpoints") / "urbansound8k_audio_cnn_best.pt"
            if src.exists():
                dst.write_bytes(src.read_bytes())
                print(f"  Best fold: {best_fold} (F1={max(r['f1'] for r in results):.4f})")
                print(f"  Summary checkpoint: {dst}")


if __name__ == "__main__":
    main()
