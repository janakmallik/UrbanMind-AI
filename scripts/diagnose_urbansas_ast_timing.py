"""Diagnostic: AST training timing measurement on fold 1.

This script measures per-clip training and eval cost with the precomputed
AST feature cache. Used during setup to verify the training pipeline runs
in acceptable time. Not part of the standard reproducibility pipeline
(use scripts/train_urbansas_audio_pretrained.py for actual training), but
retained as a diagnostic for pipeline validation.

Original context: it replaced an on-the-fly feature-extraction approach
that took ~10h for a full run. With caching, runtime is ~15 min. The
difference is documented in the model card.
"""

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import CachedASTFeatureDataset, prepare_urbansas
from urbanintel.models.ast_classifier import ASTClassifier

CACHE_DIR = _REPO_ROOT / "data" / "urbansas_ast_features"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Actual fold sizes from the audit (n_train, n_test) for epochs=10
FOLD_SIZES = [
    (440, 100),  # fold 1: acevedo0103
    (365, 175),  # fold 2: magallanes0308
    (510, 30),  # fold 3: rivera0922
    (395, 145),  # fold 4: rivera0923
    (450, 90),  # fold 5: templete0103
    (540, 540),  # final model + TAU eval
]


def main():
    print(f"Device: {DEVICE}")
    root = urbansas_root()
    split = prepare_urbansas(root, val_fraction=0.0, test_fraction=0.0)
    mavd = split.train_df[split.train_df["subset"] == "mavd"].reset_index(drop=True)

    locs = sorted(mavd["location_id"].unique())
    test_loc = locs[0]
    train_df = mavd[mavd["location_id"] != test_loc].reset_index(drop=True)
    test_df = mavd[mavd["location_id"] == test_loc].reset_index(drop=True)

    print(f"Fold 1 (measurement): train={len(train_df)}, test={len(test_df)}")

    print("Loading model ...")
    t0 = time.time()
    model = ASTClassifier().to(DEVICE)
    print(f"  loaded in {time.time() - t0:.1f}s")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_trainable:,}")

    train_ds = CachedASTFeatureDataset(train_df, cache_dir=CACHE_DIR)
    test_ds = CachedASTFeatureDataset(test_df, cache_dir=CACHE_DIR)
    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=False,  # pin_memory not needed for CPU-resident tensors
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print("\nMeasuring dataset throughput (10 batches)...")
    t0 = time.time()
    for i, (inputs, labels) in enumerate(train_loader):
        if i >= 9:  # 0..9 = 10 batches
            break
    dt = time.time() - t0
    print(f"  10 batches in {dt:.1f}s  = {dt / 10 * 1000:.0f}ms/batch")

    print("\nRunning 2 training epochs to measure per-epoch time...")
    pos_weight = (len(train_df) - train_df["has_offscreen"].sum()) / max(
        train_df["has_offscreen"].sum(), 1
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=DEVICE, dtype=torch.float32)
    )
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epoch_times = []
    epoch_train_clip_counts = []
    for epoch in range(1, 3):
        model.train()
        model.encoder.eval()
        total_loss, n_batches, n_clips = 0.0, 0, 0
        t0 = time.time()
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            n_clips += inputs.shape[0]
        scheduler.step()
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        epoch_train_clip_counts.append(n_clips)
        print(
            f"  epoch {epoch}: loss={total_loss / n_batches:.4f}  clips={n_clips}  time={epoch_time:.0f}s"
        )

    # Steady-state per-clip training cost (skip warmup epoch)
    steady_epoch = epoch_times[1]
    steady_clips = epoch_train_clip_counts[1]
    train_sec_per_clip = steady_epoch / steady_clips
    print(
        f"\n  steady-state: {steady_epoch:.0f}s / {steady_clips} clips = {train_sec_per_clip * 1000:.0f}ms per train clip"
    )

    # Eval
    print("\nMeasuring eval pass...")
    t0 = time.time()
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            print(f"  eval batch: inputs.shape={tuple(inputs.shape)}")
            inputs = inputs.to(DEVICE, non_blocking=True)
            logits = model(inputs)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    eval_time = time.time() - t0
    eval_sec_per_clip = eval_time / len(test_df)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(
        f"  eval time: {eval_time:.1f}s for {len(test_df)} clips = {eval_sec_per_clip * 1000:.0f}ms per eval clip"
    )
    print(f"  F1 (2 epochs, undertrained): {f1:.4f}")

    if DEVICE == "cuda":
        print(f"\n  GPU mem allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  GPU mem reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # --- Corrected full-run estimate ---
    print()
    print("=" * 70)
    print("FULL RUN ESTIMATE (fold-size aware)")
    print("=" * 70)
    n_epochs = 10
    total_train_time = 0.0
    total_eval_time = 0.0
    for i, (n_train, n_test) in enumerate(FOLD_SIZES, 1):
        run_train = n_epochs * n_train * train_sec_per_clip
        run_eval = n_test * eval_sec_per_clip
        total_train_time += run_train
        total_eval_time += run_eval
        print(
            f"  Run {i}: train={n_train} × {n_epochs} ep × {train_sec_per_clip * 1000:.0f}ms = {run_train:.0f}s, "
            f"eval={n_test} × {eval_sec_per_clip * 1000:.0f}ms = {run_eval:.0f}s"
        )
    total = total_train_time + total_eval_time
    print()
    print(f"  Sum training: {total_train_time / 60:.1f} min")
    print(f"  Sum eval:     {total_eval_time / 60:.1f} min")
    print(f"  Overhead (model load × 6, MLflow, IO): ~2 min")
    print(f"  ESTIMATE:     {(total + 120) / 60:.1f} min")
    print()

    if train_sec_per_clip * 440 > 2.0:  # fold-1 train-equivalent > 2s/clip
        print(
            "  WARNING: per-clip training cost is high. Consider reducing --epochs or batch-size."
        )
    else:
        print("  Timing looks reasonable. Proceed to full training if total is within budget.")


if __name__ == "__main__":
    main()
