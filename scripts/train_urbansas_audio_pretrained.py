"""Train pretrained audio baseline for Urbansas offscreen detection.

Compares frozen AST (AudioSet-pretrained) + linear head against the
from-scratch 1D-CNN baseline.

Protocol matches scripts/train_urbansas_audio.py:
    - 5-fold leave-one-location-out on MAVD
    - Per-fold class weight from that fold's training labels
    - Final model trained on ALL MAVD, evaluated once on TAU
    - Threshold fixed at 0.5, no tuning on TAU
    - Reports pooled OOF metrics (comparable to from-scratch card's 0.3686)
      AND mean per-fold metrics

Model: ASTClassifier (frozen encoder + linear head), imported from
urbanintel.models.ast_classifier.

Input: precomputed AST features from data/urbansas_ast_features/ast_features.npy,
loaded via CachedASTFeatureDataset (no per-batch feature extraction).
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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import (
    CachedASTFeatureDataset,
    prepare_urbansas,
)
from urbanintel.models.ast_classifier import MODEL_NAME, ASTClassifier

EXPERIMENT_NAME = "urbanintel-week9b-urbansas"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = _REPO_ROOT / "data" / "urbansas_ast_features"


def train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    seed: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_labels = train_df["has_offscreen"].astype(int).tolist()
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"  Train pos/neg: {n_pos}/{n_neg}, pos_weight={pos_weight:.2f}")

    train_ds = CachedASTFeatureDataset(train_df, cache_dir=CACHE_DIR)
    test_ds = CachedASTFeatureDataset(test_df, cache_dir=CACHE_DIR)
    loader_opts = {
        "num_workers": num_workers,
        "pin_memory": False,  # data is CPU-resident; no benefit from pinning
        "persistent_workers": (num_workers > 0),
    }
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_opts)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 4,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(f"  Loading AST (AudioSet-pretrained) ...")
    model = ASTClassifier().to(DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_trainable:,}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=DEVICE, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()  # disable encoder dropout; frozen anyway
        total_loss, n_batches = 0.0, 0
        t0 = time.time()
        for input_values, labels in train_loader:
            input_values = input_values.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(input_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"    epoch {epoch:2d}/{epochs}  loss={total_loss / n_batches:.4f}  "
                f"({time.time() - t0:.0f}s)"
            )

    model.eval()
    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for input_values, labels in test_loader:
            input_values = input_values.to(DEVICE, non_blocking=True)
            logits = model(input_values)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

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


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC with proper single-class guard (both classes required)."""
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Average precision, guarded for single-class."""
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not (CACHE_DIR / "ast_features.npy").exists():
        raise FileNotFoundError(
            f"AST feature cache not found at {CACHE_DIR}/ast_features.npy. "
            "Run scripts/precompute_urbansas_ast_features.py first."
        )

    root = urbansas_root()
    split = prepare_urbansas(root, val_fraction=0.0, test_fraction=0.0)
    all_df = split.train_df.copy()
    print(f"Total clips: {len(all_df)}")

    mavd_df = all_df[all_df["subset"] == "mavd"].reset_index(drop=True)
    tau_df = all_df[all_df["subset"] == "tau"].reset_index(drop=True)
    print(f"  MAVD: {len(mavd_df)} clips ({mavd_df['has_offscreen'].sum()} offscreen)")
    print(f"  TAU:  {len(tau_df)} clips ({tau_df['has_offscreen'].sum()} offscreen)")

    mavd_locations = sorted(mavd_df["location_id"].unique())
    print(f"  MAVD locations: {mavd_locations}")
    print()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="audio-pretrained-ast-lolo-mavd"):
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "encoder_frozen": True,
                "num_classes": 2,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "device": DEVICE,
                "cache": "16kHz",
            }
        )

        # ============ Phase 1: LOLA CV on MAVD ============
        print("=" * 70)
        print("PHASE 1: Leave-One-Location-Out CV on MAVD (AST pretrained)")
        print("=" * 70)
        fold_metrics = []
        oof_labels, oof_probs, oof_files = [], [], []

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
                f"  Fold F1: {m['f1']:.4f}  P: {m['precision']:.4f}  "
                f"R: {m['recall']:.4f}  (n_pos={m['n_pos_test']})"
            )
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"fold{fold_idx}_{k}", v)

        f1s = [m["f1"] for m in fold_metrics]
        oof_y = np.array(oof_labels, dtype=int)
        oof_p = np.array(oof_probs, dtype=float)
        oof_pred = (oof_p >= 0.5).astype(int)

        pooled_f1 = float(f1_score(oof_y, oof_pred, zero_division=0))
        pooled_precision = float(precision_score(oof_y, oof_pred, zero_division=0))
        pooled_recall = float(recall_score(oof_y, oof_pred, zero_division=0))
        pooled_roc_auc = _safe_auc(oof_y, oof_p)
        pooled_pr_auc = _safe_ap(oof_y, oof_p)

        print(f"\n=== CV Summary ===")
        print(f"Mean per-fold F1:   {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Pooled OOF F1:      {pooled_f1:.4f}  (from-scratch: 0.3686)")
        print(f"Pooled OOF Prec:    {pooled_precision:.4f}")
        print(f"Pooled OOF Recall:  {pooled_recall:.4f}")
        print(f"Pooled OOF ROC-AUC: {pooled_roc_auc:.4f}  (from-scratch: 0.6237)")
        print(f"Pooled OOF PR-AUC:  {pooled_pr_auc:.4f}  (from-scratch: 0.3296)")
        print()
        print("Per-location F1:")
        for m in fold_metrics:
            print(f"  {m['location_id']:22s}  F1={m['f1']:.4f}  (n_pos={m['n_pos_test']})")

        mlflow.log_metric("cv_mean_f1", float(np.mean(f1s)))
        mlflow.log_metric("cv_std_f1", float(np.std(f1s)))
        mlflow.log_metric("pooled_oof_f1", pooled_f1)
        mlflow.log_metric("pooled_oof_precision", pooled_precision)
        mlflow.log_metric("pooled_oof_recall", pooled_recall)
        mlflow.log_metric("pooled_oof_roc_auc", pooled_roc_auc)
        mlflow.log_metric("pooled_oof_pr_auc", pooled_pr_auc)

        results_dir = _REPO_ROOT / "data" / "urbansas_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        oof_df = pd.DataFrame({"filename": oof_files, "y_true": oof_labels, "y_prob": oof_probs})
        oof_df.to_csv(results_dir / "audio_pretrained_oof.csv", index=False)
        print(f"\nOOF predictions saved: {results_dir / 'audio_pretrained_oof.csv'}")

        # ============ Phase 2: Final model, TAU external ============
        print("\n" + "=" * 70)
        print("PHASE 2: Final model on ALL MAVD, external test on TAU (AST)")
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
        print(f"  F1:        {final_metrics['f1']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall:    {final_metrics['recall']:.4f}")
        print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"  n_pos:     {final_metrics['n_pos_test']}")

        for k, v in final_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"tau_external_{k}", v)

        tau_df_out = pd.DataFrame(
            {
                "filename": tau_df["filename"].tolist(),
                "y_true": tau_y_true.tolist(),
                "y_pred": tau_y_pred.tolist(),
                "y_prob": tau_y_prob.tolist(),
            }
        )
        tau_df_out.to_csv(results_dir / "audio_pretrained_tau_external.csv", index=False)

        torch.save(
            {"model": final_model.state_dict(), "config": vars(args)},
            results_dir / "audio_pretrained_final.pt",
        )
        print(f"\nFinal model saved: {results_dir / 'audio_pretrained_final.pt'}")


if __name__ == "__main__":
    main()
