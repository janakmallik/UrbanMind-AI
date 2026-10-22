"""Train R(2+1)D-18 on UCF-Crime 4-category CLIP-LEVEL classification.

IMPORTANT — this is NOT temporal anomaly localization:

  UCF-Crime provides ONLY video-level labels (this video contains a
  burglary). A single video may have normal segments before/after the
  anomaly. Our labels apply the video's category to every clip extracted
  from that video — including clips that visually show "normal" activity.

  This is therefore best described as:
    "Weakly supervised clip-level anomaly classification with video-level labels"

  NOT "precise anomaly detection with temporal localization."

  Temporal localization would require annotation of anomaly start/end
  times, which UCF-Crime does not provide.
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
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

from urbanintel.data.ucf_crime import CATEGORIES, NUM_CATEGORIES
from urbanintel.data.ucf_crime_video import UCFCrimeClipDataset

EXPERIMENT_NAME = "urbanintel-week7-5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIPS_DIR = _REPO_ROOT / "data" / "ucf_crime_clips"


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device, non_blocking=True)
            logits = model(frames.permute(0, 2, 1, 3, 4))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def confidence_analysis(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Bucket predictions by confidence and report accuracy.

    IMPORTANT: Softmax confidence ≠ probability of correctness.
    These buckets show how accuracy correlates with confidence on THIS split.
    """
    top_conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = preds == labels

    buckets = {
        "conf_high_gt_070": top_conf >= 0.70,
        "conf_mid_040_070": (top_conf >= 0.40) & (top_conf < 0.70),
        "conf_low_lt_040": top_conf < 0.40,
    }

    results = {}
    for name, mask in buckets.items():
        if mask.sum() == 0:
            results[name] = {"count": 0, "accuracy": float("nan")}
        else:
            results[name] = {
                "count": int(mask.sum()),
                "accuracy": float(correct[mask].mean()),
            }
    return results


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute ECE — measures how well softmax confidence matches actual accuracy."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / total) * abs(bin_acc - bin_conf)
    return float(ece)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\nLoading clip DataFrames ...")
    train_clips = pd.read_parquet(CLIPS_DIR / "train_clips.parquet")
    val_clips = pd.read_parquet(CLIPS_DIR / "val_clips.parquet")
    test_clips = pd.read_parquet(CLIPS_DIR / "test_clips.parquet")
    official_test = pd.read_parquet(CLIPS_DIR / "official_test_clips.parquet")

    print(f"Train clips: {len(train_clips)}")
    print(f"Val clips:   {len(val_clips)}")
    print(f"Test clips:  {len(test_clips)}")
    print(f"Official test clips: {len(official_test)}")

    train_ds = UCFCrimeClipDataset(train_clips, is_train=True)
    val_ds = UCFCrimeClipDataset(val_clips, is_train=False)
    test_ds = UCFCrimeClipDataset(test_clips, is_train=False)
    official_ds = UCFCrimeClipDataset(official_test, is_train=False)

    from torch.utils.data import WeightedRandomSampler

    # Compute per-sample weights based on class frequency
    train_cat_counts = train_clips["category"].value_counts().to_dict()
    sample_weights = [1.0 / train_cat_counts[int(cat)] for cat in train_clips["category"].tolist()]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    loader_opts = {"num_workers": args.num_workers, "pin_memory": (DEVICE == "cuda")}
    official_loader_opts = {"num_workers": 0, "pin_memory": (DEVICE == "cuda")}
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
        **loader_opts,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_opts)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_opts)
    official_loader = DataLoader(
        official_ds, batch_size=args.batch_size, shuffle=False, **official_loader_opts
    )

    # Model
    print("\nLoading R(2+1)D-18 (Kinetics-400 pretrained) ...")
    model = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CATEGORIES)
    model = model.to(DEVICE)

    # Freeze backbone initially (only train head), unfreeze after 3 epochs
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
    n_params_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_params_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable (backbone frozen): {n_params_frozen:,} / {n_params_total:,}")

    UNFREEZE_EPOCH = 3

    # Class weights
    counts = np.array([(train_clips["category"] == i).sum() for i in range(NUM_CATEGORIES)])
    assert (counts > 0).all(), f"Zero training clips in some categories: {counts}"
    weights = (counts.sum() / (NUM_CATEGORIES * counts)).astype(np.float32)
    print(f"Class weights: {weights}")
    class_weights = torch.tensor(weights, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Differential learning rate
    head_ids = {id(p) for p in model.fc.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.3},
            {"params": list(model.fc.parameters()), "lr": args.lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(device="cuda") if use_amp else None

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    best_path = _REPO_ROOT / "ucf_crime_4class_best.pt"

    # Early stopping
    patience = 3
    epochs_no_improvement = 0
    min_delta = 0.005  # require at least 0.5% improvement to count

    with mlflow.start_run(run_name="ucf-crime-4class"):
        mlflow.log_params(
            {
                "model": "r2plus1d_18",
                "pretrained": "KINETICS400_V1",
                "num_classes": NUM_CATEGORIES,
                "category_names": str(CATEGORIES),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "n_params_total": n_params_total,
                "n_params_frozen_phase": n_params_frozen,
                "train_clips": len(train_clips),
                "val_clips": len(val_clips),
                "test_clips": len(test_clips),
                "official_test_clips": len(official_test),
            }
        )

        # Why: The head learns the task-specific mapping first (30 min, fast).
        # Then the backbone adapts at 10× lower LR (prevents catastrophic
        # forgetting + overfitting). This is the standard "warmup" fine-tuning
        # pattern.
        for epoch in range(1, args.epochs + 1):
            if epoch == UNFREEZE_EPOCH:
                print(f"\n>>> Unfreezing backbone at epoch {epoch} <<<")
                for name, param in model.named_parameters():
                    if not name.startswith("fc."):
                        param.requires_grad = True
                # Rebuild optimizer with unfrozen params
                head_ids = {id(p) for p in model.fc.parameters()}
                backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
                optimizer = torch.optim.AdamW(
                    [
                        {
                            "params": backbone_params,
                            "lr": args.lr * 0.1,
                        },  # 10x lower for unfrozen backbone
                        {"params": list(model.fc.parameters()), "lr": args.lr},
                    ],
                    weight_decay=1e-4,
                )

            model.train()
            total_loss, n_batches = 0.0, 0
            t0 = time.time()

            for frames, labels in train_loader:
                frames = frames.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast(device_type="cuda"):
                        logits = model(frames.permute(0, 2, 1, 3, 4))
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
            val_preds = p_val.argmax(axis=1)
            val_acc = float(accuracy_score(y_val, val_preds))
            val_f1 = float(f1_score(y_val, val_preds, average="macro"))

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

            if val_f1 > best_val_f1 + min_delta:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, best_path)
                epochs_no_improvement = 0
                print(f"  -> saved best (val_f1={val_f1:.4f})")
            else:
                epochs_no_improvement += 1
                print(f"  -> no improvement ({epochs_no_improvement}/{patience})")
                if epochs_no_improvement >= patience:
                    print(f"  -> early stopping at epoch {epoch}")
                    break

        if best_state is None:
            raise RuntimeError("No checkpoint saved.")

        print(f"\nBest epoch: {best_epoch} (val_f1={best_val_f1:.4f})")
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_f1_macro", best_val_f1)
        mlflow.log_artifact(str(best_path))

        # Test on held-out split
        model.load_state_dict(torch.load(best_path, weights_only=True))
        y_test, p_test = evaluate(model, test_loader, DEVICE)
        test_preds = p_test.argmax(axis=1)
        test_acc = float(accuracy_score(y_test, test_preds))
        test_f1 = float(f1_score(y_test, test_preds, average="macro"))
        mlflow.log_metrics({"test_accuracy": test_acc, "test_f1_macro": test_f1})

        print(f"\n=== Test results (held-out split) ===")
        print(f"Accuracy:  {test_acc:.4f}")
        print(f"Macro-F1:  {test_f1:.4f}")

        # Confidence analysis
        print("\n=== Confidence analysis (test set) ===")
        print("NOTE: softmax confidence is not calibrated probability of correctness.")
        conf = confidence_analysis(p_test, y_test)
        for name, stats in conf.items():
            print(f"  {name:<20}  n={stats['count']:>4}  accuracy={stats['accuracy']:.4f}")
            mlflow.log_metric(f"{name}_accuracy", stats["accuracy"])
            mlflow.log_metric(f"{name}_count", stats["count"])

        ece_test = expected_calibration_error(p_test, y_test)
        print(f"\nExpected Calibration Error (test): {ece_test:.4f}")
        print(f"  (0 = perfectly calibrated, higher = more overconfident)")
        mlflow.log_metric("test_ece", ece_test)

        # Official test evaluation
        print(f"\n=== Evaluation on official UCF-Crime Test ===")
        y_off, p_off = evaluate(model, official_loader, DEVICE)
        off_preds = p_off.argmax(axis=1)
        off_acc = float(accuracy_score(y_off, off_preds))
        off_f1 = float(f1_score(y_off, off_preds, average="macro"))
        print(f"Accuracy:  {off_acc:.4f}")
        print(f"Macro-F1:  {off_f1:.4f}")
        mlflow.log_metrics({"official_test_accuracy": off_acc, "official_test_f1": off_f1})

        # Per-class report
        report = classification_report(
            y_off,
            off_preds,
            labels=list(range(NUM_CATEGORIES)),
            target_names=[CATEGORIES[i] for i in range(NUM_CATEGORIES)],
            zero_division=0,
        )
        print("\nPer-category report on official test:")
        print(report)
        mlflow.log_text(report, "official_test_report.txt")

        # Confidence analysis on official test
        conf_off = confidence_analysis(p_off, y_off)
        print("\nOfficial test confidence analysis:")
        for name, stats in conf_off.items():
            print(f"  {name:<20}  n={stats['count']:>4}  accuracy={stats['accuracy']:.4f}")


if __name__ == "__main__":
    main()
