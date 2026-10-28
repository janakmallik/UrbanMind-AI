"""Train audio-only baseline for offscreen vehicle detection.

Protocol (per other AI's corrections):
    1. Leave-One-Location-Out CV on 5 MAVD locations
       - per-fold pos_weight from that fold's training labels
       - per-fold F1, precision, recall, positive count
       - mean ± std across folds
    2. After CV completes, train a FRESH model on all 5 MAVD locations
    3. Evaluate that model ONCE on all 540 TAU clips (external test)
       - threshold fixed at 0.5 BEFORE TAU evaluation
       - no tuning on TAU

Saves:
    - OOF predictions for paired bootstrap
    - Final MAVD model checkpoint
    - TAU external test predictions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
import os

os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(_REPO_ROOT / 'mlflow.db').as_posix()}"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import (
    N_MELS,
    CachedUrbansasAudioDataset,
    prepare_urbansas,
)

EXPERIMENT_NAME = "urbanintel-week9b-urbansas"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model ---


class AudioCNN(nn.Module):
    """1D-CNN on log-mel spectrograms (same as Week 7.6)."""

    def __init__(self, n_mels: int = N_MELS):
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
        self.head = nn.Linear(256, 2)  # binary: 0/1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        h = self.conv(x).squeeze(-1)
        h = self.dropout(h)
        return self.head(h)


# --- Training per fold ---


def train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    seed: int,
) -> dict:
    """Train one fold, return predictions + metrics."""
    torch.manual_seed(seed)

    train_labels = train_df["has_offscreen"].astype(int).tolist()
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=DEVICE)
    print(f"  Train pos/neg: {n_pos}/{n_neg}, pos_weight={pos_weight.item():.2f}")

    train_ds = CachedUrbansasAudioDataset(train_df, is_train=True)
    test_ds = CachedUrbansasAudioDataset(test_df, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
    )

    model = AudioCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight.item()], device=DEVICE))

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            print(f"    epoch {epoch:2d}  loss={total_loss / n_batches:.4f}")

    # Evaluate on test fold
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(y.numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_test": len(y_true),
        "n_pos_test": int(y_true.sum()),
    }
    return metrics, y_true, y_pred, y_prob, model


# --- Main ---


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    root = urbansas_root()
    split = prepare_urbansas(root, val_fraction=0.0, test_fraction=0.0)

    # Because val_fraction=0 and test_fraction=0, everything goes to train_df
    all_df = split.train_df.copy()
    print(f"Total clips: {len(all_df)}")

    # Split by subset
    mavd_df = all_df[all_df["subset"] == "mavd"].reset_index(drop=True)
    tau_df = all_df[all_df["subset"] == "tau"].reset_index(drop=True)
    print(f"  MAVD: {len(mavd_df)} clips ({mavd_df['has_offscreen'].sum()} offscreen)")
    print(f"  TAU:  {len(tau_df)} clips ({tau_df['has_offscreen'].sum()} offscreen)")

    # MAVD locations for LeaveOneLocationOut
    mavd_locations = sorted(mavd_df["location_id"].unique())
    print(f"  MAVD locations: {mavd_locations}")
    print()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="audio-only-lolo-mavd"):
        mlflow.log_params(vars(args))

        # === Phase 1: Leave-One-Location-Out CV on MAVD ===
        print("=" * 70)
        print("PHASE 1: Leave-One-Location-Out CV on MAVD")
        print("=" * 70)
        fold_metrics = []
        oof_labels = []
        oof_probs = []
        oof_files = []

        for fold_idx, test_loc in enumerate(mavd_locations, 1):
            print(f"\n=== Fold {fold_idx}/{len(mavd_locations)}: test={test_loc} ===")
            test_df = mavd_df[mavd_df["location_id"] == test_loc].reset_index(drop=True)
            train_df = mavd_df[mavd_df["location_id"] != test_loc].reset_index(drop=True)
            print(
                f"  Train: {len(train_df)} clips, Test: {len(test_df)} clips "
                f"({test_df['has_offscreen'].sum()} offscreen)"
            )

            m, y_true, y_pred, y_prob, _ = train_and_eval(
                train_df,
                test_df,
                args.epochs,
                args.batch_size,
                args.lr,
                args.num_workers,
                args.seed + fold_idx,
            )
            m["location_id"] = test_loc
            fold_metrics.append(m)
            oof_labels.extend(y_true.tolist())
            oof_probs.extend(y_prob.tolist())
            oof_files.extend(test_df["filename"].tolist())

            print(
                f"  Fold F1: {m['f1']:.4f}  "
                f"P: {m['precision']:.4f}  R: {m['recall']:.4f}  "
                f"(n_pos={m['n_pos_test']})"
            )

            for k, v in m.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"fold{fold_idx}_{k}", v)

        # Aggregate
        f1s = [m["f1"] for m in fold_metrics]
        print(f"\n=== CV Summary ===")
        print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print("Per-location F1:")
        for m in fold_metrics:
            print(f"  {m['location_id']:25s}  F1={m['f1']:.4f}  (n_pos={m['n_pos_test']})")
        mlflow.log_metric("cv_mean_f1", float(np.mean(f1s)))
        mlflow.log_metric("cv_std_f1", float(np.std(f1s)))

        # Save OOF predictions
        results_dir = _REPO_ROOT / "data" / "urbansas_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        oof_df = pd.DataFrame({"filename": oof_files, "y_true": oof_labels, "y_prob": oof_probs})
        oof_df.to_csv(results_dir / "audio_only_oof.csv", index=False)
        print(f"\nOOF predictions saved to: {results_dir / 'audio_only_oof.csv'}")

        # === Phase 2: Train FINAL model on ALL MAVD, evaluate on TAU ===
        print("\n" + "=" * 70)
        print("PHASE 2: Final model on ALL MAVD, external test on TAU")
        print("=" * 70)
        final_metrics, tau_y_true, tau_y_pred, tau_y_prob, final_model = train_and_eval(
            mavd_df,
            tau_df,
            args.epochs,
            args.batch_size,
            args.lr,
            args.num_workers,
            args.seed + 100,
        )
        print(f"\nTAU external test:")
        print(f"  F1: {final_metrics['f1']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall: {final_metrics['recall']:.4f}")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  n_pos: {final_metrics['n_pos_test']}")

        for k, v in final_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"tau_external_{k}", v)

        # Save TAU predictions
        tau_df_out = pd.DataFrame(
            {
                "filename": tau_df["filename"].tolist(),
                "y_true": tau_y_true.tolist(),
                "y_pred": tau_y_pred.tolist(),
                "y_prob": tau_y_prob.tolist(),
            }
        )
        tau_df_out.to_csv(results_dir / "audio_only_tau_external.csv", index=False)

        # Save checkpoint
        torch.save(
            {"model": final_model.state_dict(), "config": vars(args)},
            results_dir / "audio_only_final.pt",
        )
        print(f"Final model saved to: {results_dir / 'audio_only_final.pt'}")


if __name__ == "__main__":
    main()
