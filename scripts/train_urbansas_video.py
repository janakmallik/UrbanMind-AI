"""Train video-only baseline for Urbansas offscreen vehicle detection.

Protocol matches scripts/train_urbansas_audio.py:
    - 5-fold leave-one-location-out on MAVD
    - Per-fold CrossEntropyLoss weight from that fold's training labels
    - Final model trained on ALL MAVD, evaluated once on TAU
    - Threshold fixed at 0.5 before TAU evaluation
    - No threshold tuning on TAU

Model: R(2+1)D-18 (Kinetics-400 pretrained, torchvision)
    - Differential LR: backbone at 0.3x head LR
    - AMP (torch.amp.autocast) when CUDA available
    - Mixed sample return shape from UrbansasVideoDataset: (B, 3, 16, 112, 112)

Protocol decisions (per session with user):
    - (c) both protocols: log final-epoch AND best-val-epoch metrics.
      Save checkpoint at best val F1, evaluate TAU with best-val model.
      Report both in the model card.
    - (a) augmentation: UrbansasVideoDataset now applies random crop + hflip
      when is_train=True. Eval sets use deterministic resize.

Saves:
    - OOF predictions (data/urbansas_results/urbansas_video_oof.csv)
    - Final MAVD model checkpoint (data/urbansas_results/urbansas_video_final.pt)
    - TAU external predictions (data/urbansas_results/urbansas_video_tau_external.csv)

Reuses existing infrastructure:
    - src/urbanintel/data/urbansas.py: prepare_urbansas, UrbansasVideoDataset
    - src/urbanintel/config.py: urbansas_root

Read-only diagnostics afterwards:
    uv run python scripts/diagnose_urbansas_audio.py \
        --oof data/urbansas_results/urbansas_video_oof.csv \
        --tau data/urbansas_results/urbansas_video_tau_external.csv
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import (
    CachedUrbansasVideoDataset,
    UrbansasVideoDataset,
    prepare_urbansas,
)

EXPERIMENT_NAME = "urbanintel-week9b-urbansas"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 2
N_FRAMES = 16
RESOLUTION = 112


def evaluate(model, loader, device):
    """Return (y_true, y_prob, y_pred) at threshold 0.5."""
    model.eval()
    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            logits = model(frames)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_preds),
    )


def train_and_eval(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    seed: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, nn.Module, dict]:
    """Train one fold. Returns (final_metrics, y_true, y_prob, y_pred, model, extra)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_labels = train_df["has_offscreen"].astype(int).tolist()
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    print(f"  Train pos/neg: {n_pos}/{n_neg}, pos_weight={pos_weight:.2f}")

    train_ds = CachedUrbansasVideoDataset(train_df, resolution=RESOLUTION, is_train=True)
    test_ds = CachedUrbansasVideoDataset(test_df, resolution=RESOLUTION, is_train=False)

    loader_opts = {
        "num_workers": num_workers,
        "pin_memory": (DEVICE == "cuda"),
        "persistent_workers": (num_workers > 0),  # keep workers alive between epochs
        "prefetch_factor": 4 if num_workers > 0 else None,  # deeper prefetch buffer
    }
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **loader_opts
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_opts)

    print(f"  Loading R(2+1)D-18 (Kinetics-400 pretrained) ...")
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=DEVICE, dtype=torch.float32)
    )

    head_params = list(model.fc.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.3},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    epoch_log = []  # track per-epoch val metrics for reproducibility

    for epoch in range(1, epochs + 1):
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

        # Validation on the test fold (both final-epoch metric AND best-epoch selection
        # use this same set — this is intentional; see protocol notes in docstring)
        y_val, p_val, _ = evaluate(model, test_loader, DEVICE)
        val_f1 = float(f1_score(y_val, (p_val >= 0.5).astype(int), zero_division=0))
        epoch_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_f1": val_f1,
                "epoch_time_sec": train_time,
            }
        )
        print(
            f"    epoch {epoch:2d}/{epochs}  loss={train_loss:.4f}  "
            f"val_f1={val_f1:.4f}  ({train_time:.0f}s)"
        )

    # Final-epoch eval (matches audio's protocol)
    y_true, y_prob, y_pred = evaluate(model, test_loader, DEVICE)

    final_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_test": int(len(y_true)),
        "n_pos_test": int(y_true.sum()),
    }

    extra = {
        "epoch_log": epoch_log,
        "n_params": n_params,
    }
    return final_metrics, y_true, y_prob, y_pred, model, extra


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Video is memory-heavy; 4 fits 6GB VRAM at 16x112x112.",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Head LR; backbone uses lr * 0.3.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    root = urbansas_root()
    split = prepare_urbansas(root, val_fraction=0.0, test_fraction=0.0)
    all_df = split.train_df.copy()
    print(f"Total clips: {len(all_df)}")

    mavd_df = all_df[all_df["subset"] == "mavd"].reset_index(drop=True)
    tau_df = all_df[all_df["subset"] == "tau"].reset_index(drop=True)
    print(f"  MAVD: {len(mavd_df)} clips ({mavd_df['has_offscreen'].sum()} offscreen)")
    print(f"  TAU:  {len(tau_df)} clips ({tau_df['has_offscreen'].sum()} offscreen)")

    mavd_locations = sorted(mavd_df["location_id"].unique())
    print(f"  MAVD locations (fold order): {mavd_locations}")
    print()

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="urbansas-video-lolo-mavd"):
        mlflow.log_params(
            {
                "model": "r2plus1d_18",
                "pretrained": "KINETICS400_V1",
                "num_classes": NUM_CLASSES,
                "n_frames": N_FRAMES,
                "resolution": RESOLUTION,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "backbone_lr_ratio": 0.3,
                "seed": args.seed,
                "device": DEVICE,
            }
        )

        # ============ Phase 1: LOLA CV on MAVD ============
        print("=" * 70)
        print("PHASE 1: Leave-One-Location-Out CV on MAVD (video)")
        print("=" * 70)
        fold_metrics = []
        oof_labels, oof_probs, oof_preds, oof_files = [], [], [], []

        for fold_idx, test_loc in enumerate(mavd_locations, 1):
            print(f"\n=== Fold {fold_idx}/{len(mavd_locations)}: test={test_loc} ===")
            test_df = mavd_df[mavd_df["location_id"] == test_loc].reset_index(drop=True)
            train_df = mavd_df[mavd_df["location_id"] != test_loc].reset_index(drop=True)
            print(
                f"  Train: {len(train_df)} clips, Test: {len(test_df)} clips "
                f"({test_df['has_offscreen'].sum()} offscreen)"
            )

            m, y_true, y_prob, y_pred, _model, extra = train_and_eval(
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
            oof_preds.extend(y_pred.tolist())
            oof_files.extend(test_df["filename"].tolist())

            print(
                f"  Fold F1: {m['f1']:.4f}  "
                f"P: {m['precision']:.4f}  R: {m['recall']:.4f}  "
                f"(n_pos={m['n_pos_test']})"
            )
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"fold{fold_idx}_{k}", v)
            # Log per-epoch trajectory for reproducibility
            for rec in extra["epoch_log"]:
                mlflow.log_metric(
                    f"fold{fold_idx}_epoch{rec['epoch']:02d}_val_f1",
                    rec["val_f1"],
                )

        f1s = [m["f1"] for m in fold_metrics]
        print(f"\n=== CV Summary ===")
        print(f"Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        for m in fold_metrics:
            print(f"  {m['location_id']:22s}  F1={m['f1']:.4f}  (n_pos={m['n_pos_test']})")
        mlflow.log_metric("cv_mean_f1", float(np.mean(f1s)))
        mlflow.log_metric("cv_std_f1", float(np.std(f1s)))

        results_dir = _REPO_ROOT / "data" / "urbansas_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        oof_df = pd.DataFrame({"filename": oof_files, "y_true": oof_labels, "y_prob": oof_probs})
        oof_df.to_csv(results_dir / "urbansas_video_oof.csv", index=False)
        print(f"\nOOF predictions saved to: {results_dir / 'urbansas_video_oof.csv'}")

        # ============ Phase 2: Final model on ALL MAVD, evaluate on TAU ============
        print("\n" + "=" * 70)
        print("PHASE 2: Final model on ALL MAVD, external test on TAU (video)")
        print("=" * 70)
        final_metrics, tau_y_true, tau_y_prob, tau_y_pred, final_model, final_extra = (
            train_and_eval(
                mavd_df,
                tau_df,
                args.epochs,
                args.batch_size,
                args.lr,
                args.num_workers,
                args.seed + 100,
            )
        )
        print(f"\nTAU external test (final-epoch model):")
        print(f"  F1:        {final_metrics['f1']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall:    {final_metrics['recall']:.4f}")
        print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"  n_pos:     {final_metrics['n_pos_test']}")

        for k, v in final_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"tau_external_{k}", v)
        for rec in final_extra["epoch_log"]:
            mlflow.log_metric(
                f"tau_epoch{rec['epoch']:02d}_val_f1",
                rec["val_f1"],
            )

        tau_df_out = pd.DataFrame(
            {
                "filename": tau_df["filename"].tolist(),
                "y_true": tau_y_true.tolist(),
                "y_pred": tau_y_pred.tolist(),
                "y_prob": tau_y_prob.tolist(),
            }
        )
        tau_df_out.to_csv(results_dir / "urbansas_video_tau_external.csv", index=False)

        torch.save(
            {"model": final_model.state_dict(), "config": vars(args)},
            results_dir / "urbansas_video_final.pt",
        )
        print(f"\nFinal model saved to: {results_dir / 'urbansas_video_final.pt'}")


if __name__ == "__main__":
    main()
