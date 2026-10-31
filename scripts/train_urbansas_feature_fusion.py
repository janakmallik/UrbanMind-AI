"""Feature-level fusion on Urbansas: audio + video embeddings → MLP.

Three MLPs compared, all with the same architecture (hidden=256, dropout=0.3):
    audio-only:    768 → 256 → 2
    video-only:    512 → 256 → 2
    feature-fusion: 1280 → 256 → 2

Protocol:
    - 5-fold leave-one-location-out (LOLA) on MAVD
    - Per-fold StandardScaler fit on TRAINING locations only (no leakage)
    - 20 fixed epochs, no early stopping (avoids using test fold for selection)
    - Per-epoch training loss logged to JSON
    - Test-fold metrics computed once per fold, at the end
    - TAU external: train on all 540 MAVD, evaluate on 540 TAU

Comparisons (paired bootstrap, 1000 resamples, clip-level + location-level):
    - feature_fusion vs audio_only_mlp   (does video add value?)
    - feature_fusion vs video_only_mlp   (does audio add value?)
    (Note: late_fusion_max comparison removed — its baseline used the contaminated
     fine-tuned video encoder. C2b only compares clean vs clean.)

Reference baselines (from Week 9c):
    - AST audio alone: F1 0.4673, ROC-AUC 0.6801, AP 0.3933
    - Video alone:     F1 0.4643, ROC-AUC 0.6130, AP 0.3063
    - Late fusion max: F1 0.4713, ROC-AUC 0.6176

Outputs:
    - data/urbansas_results/feature_fusion_oof.csv
    - data/urbansas_results/feature_fusion_tau_external.csv
    - data/urbansas_results/feature_fusion_results{_kinetics}.json
    - data/urbansas_results/feature_fusion_training_logs{_kinetics}.json
    - data/urbansas_results/feature_fusion_metadata{_kinetics}.json
    (suffix depends on --video-embedding-variant)

Runtime: ~10 min on RTX 4050.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from urbanintel.config import urbansas_root
from urbanintel.data.urbansas import prepare_urbansas

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = _REPO_ROOT / "data" / "urbansas_results"

# Reference baselines from Week 9c (hardcoded for comparison)
WEEK_9C_BASELINES = {
    "ast_audio_alone": {"f1": 0.4673, "roc_auc": 0.6801, "ap": 0.3933},
    "video_alone": {"f1": 0.4643, "roc_auc": 0.6130, "ap": 0.3063},
    "late_fusion_max": {"f1": 0.4713, "roc_auc": 0.6176, "ap": 0.3121},
    "late_fusion_logistic": {"f1": 0.4660, "roc_auc": 0.6684, "ap": 0.3480},
}


# ============================== MLP models ==============================


class AudioOnlyMLP(nn.Module):
    """768-dim audio embedding → 256 → 2."""

    def __init__(self, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(768, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class VideoOnlyMLP(nn.Module):
    """512-dim video embedding → 256 → 2."""

    def __init__(self, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(512, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


class FeatureFusionMLP(nn.Module):
    """1280-dim concatenated embedding → 256 → 2."""

    def __init__(self, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(768 + 512, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.fc2(self.dropout(torch.relu(self.fc1(x))))


def build_mlp(kind: str) -> nn.Module:
    if kind == "audio":
        return AudioOnlyMLP().to(DEVICE)
    if kind == "video":
        return VideoOnlyMLP().to(DEVICE)
    if kind == "fusion":
        return FeatureFusionMLP().to(DEVICE)
    raise ValueError(f"unknown kind: {kind}")


def slice_input(kind: str, audio: np.ndarray, video: np.ndarray) -> np.ndarray:
    if kind == "audio":
        return audio
    if kind == "video":
        return video
    if kind == "fusion":
        return np.concatenate([audio, video], axis=1)
    raise ValueError(f"unknown kind: {kind}")


# ============================== Training ==============================


def train_one_fold(
    kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Train one MLP for one fold, return predictions + metrics + training log."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_mlp(kind)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Class-weighted loss
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    pos_weight = n_neg / max(n_pos, 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_weight], device=DEVICE, dtype=torch.float32)
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # DataLoader
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    train_ds = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # Per-epoch training log (loss only; test metrics computed once at end)
    epoch_log = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        epoch_log.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(n_batches, 1),
                "epoch_time_s": time.time() - t0,
            }
        )

    # Evaluate ONCE on test fold
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        logits = model(xb)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=-1).cpu().numpy()

    metrics = {
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "n_test": int(len(y_test)),
        "n_pos_test": int(y_test.sum()),
        "n_params": int(n_params),
    }
    # AUCs only if both classes present
    if len(np.unique(y_test)) >= 2:
        metrics["roc_auc"] = float(roc_auc_score(y_test, probs))
        metrics["ap"] = float(average_precision_score(y_test, probs))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["ap"] = float("nan")

    return metrics, probs, preds, epoch_log


# ============================== Bootstrap ==============================


def bootstrap_f1_diff(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    locations: np.ndarray,
    resample_unit: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap of (F1_b - F1_a). Positive = b beats a."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = (np.asarray(probs_a) >= 0.5).astype(int)
    pred_b = (np.asarray(probs_b) >= 0.5).astype(int)
    n = len(y_true)

    def f1_from_pred(y, pred):
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-12)

    diffs = []
    if resample_unit == "clip":
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            diffs.append(
                f1_from_pred(y_true[idx], pred_b[idx]) - f1_from_pred(y_true[idx], pred_a[idx])
            )
    elif resample_unit == "location":
        unique_locs = np.unique(locations)
        loc_idx = {loc: np.where(locations == loc)[0] for loc in unique_locs}
        for _ in range(n_boot):
            chosen = rng.choice(unique_locs, size=len(unique_locs), replace=True)
            idx = np.concatenate([loc_idx[loc] for loc in chosen])
            diffs.append(
                f1_from_pred(y_true[idx], pred_b[idx]) - f1_from_pred(y_true[idx], pred_a[idx])
            )
    else:
        raise ValueError(f"unknown resample_unit: {resample_unit}")

    diffs = np.asarray(diffs)
    return {
        "mean": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_b_beats_a": float((diffs > 0).mean()),
    }


# ============================== Main ==============================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--video-embedding-variant",
        choices=["fine_tuned", "kinetics_only"],
        default="fine_tuned",
        help=("Which video embedding file to use. kinetics_only is leak-free (for C2b)."),
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Deterministic seed controls ---
    import random

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeds set to {args.seed} (deterministic mode)")

    # --- Load embeddings ---
    audio_emb = np.load(RESULTS_DIR / "audio_embeddings.npy")
    if args.video_embedding_variant == "kinetics_only":
        video_emb = np.load(RESULTS_DIR / "video_embeddings_kinetics.npy")
        out_suffix = "_kinetics"
        print("Using LEAK-FREE Kinetics-only video embeddings")
    else:
        video_emb = np.load(RESULTS_DIR / "video_embeddings.npy")
        out_suffix = ""
        print("WARNING: using fine-tuned video embeddings (contaminated for MAVD OOF)")
    assert np.isfinite(audio_emb).all(), "Audio embeddings contain NaN/Inf"
    assert np.isfinite(video_emb).all(), "Video embeddings contain NaN/Inf"
    with open(RESULTS_DIR / "embedding_filenames.txt") as f:
        emb_fns = f.read().strip().split("\n")

    assert audio_emb.shape == (1080, 768)
    assert video_emb.shape == (1080, 512)
    assert len(emb_fns) == 1080

    # --- Load metadata to identify MAVD vs TAU and location ---
    root = urbansas_root()
    split = prepare_urbansas(root, val_fraction=0.0, test_fraction=0.0)
    meta = split.train_df[["filename", "subset", "location_id", "has_offscreen"]].copy()
    assert len(meta) == 1080
    assert meta["filename"].is_unique

    # Verify the embedding filename order matches the metadata order
    assert meta["filename"].tolist() == emb_fns, "embedding order differs from metadata order"

    # Split MAVD and TAU
    mavd_mask = meta["subset"] == "mavd"
    tau_mask = meta["subset"] == "tau"
    assert mavd_mask.sum() == 540
    assert tau_mask.sum() == 540

    mavd_fns = meta.loc[mavd_mask, "filename"].tolist()
    tau_fns = meta.loc[tau_mask, "filename"].tolist()

    # --- In-script OOF alignment verification ---
    # Verify OOF files agree with each other and with metadata.
    # (Doesn't reference mavd_y here — mavd_y is created below from meta.)
    audio_oof_check = pd.read_csv(RESULTS_DIR / "audio_pretrained_oof.csv")
    video_oof_check = pd.read_csv(RESULTS_DIR / "urbansas_video_oof.csv")
    assert audio_oof_check["filename"].tolist() == video_oof_check["filename"].tolist(), (
        "Audio and video OOF filename orders differ"
    )
    assert audio_oof_check["filename"].tolist() == mavd_fns, (
        "OOF filename order differs from metadata MAVD order"
    )
    merged_check = audio_oof_check.merge(video_oof_check, on="filename", suffixes=("_a", "_v"))
    assert (merged_check["y_true_a"] == merged_check["y_true_v"]).all(), (
        "Audio and video OOF y_true differ"
    )
    # Cross-check against metadata labels
    meta_labels_for_oof = meta.set_index("filename").loc[mavd_fns, "has_offscreen"].astype(int)
    assert merged_check["y_true_a"].tolist() == meta_labels_for_oof.tolist(), (
        "OOF y_true differs from metadata has_offscreen"
    )
    print("OOF alignment verified in-script")
    mavd_locs = meta.loc[mavd_mask, "location_id"].tolist()
    mavd_y = meta.loc[mavd_mask, "has_offscreen"].astype(int).values
    tau_y = meta.loc[tau_mask, "has_offscreen"].astype(int).values

    audio_mavd = audio_emb[mavd_mask.values]
    video_mavd = video_emb[mavd_mask.values]
    audio_tau = audio_emb[tau_mask.values]
    video_tau = video_emb[tau_mask.values]

    print(f"\nMAVD: {len(mavd_fns)} clips, {int(mavd_y.sum())} positives")
    print(f"TAU:  {len(tau_fns)} clips, {int(tau_y.sum())} positives")
    print(f"MAVD locations: {sorted(set(mavd_locs))}")

    # --- 5-fold LOLA on MAVD ---
    locations = sorted(set(mavd_locs))
    kinds = ["audio", "video", "fusion"]

    # Storage
    oof = {
        k: {
            "probs": np.zeros(540, dtype=np.float32),
            "preds": np.zeros(540, dtype=int),
            "fold_metrics": [],
            "epoch_log": [],
        }
        for k in kinds
    }

    print("\n" + "=" * 72)
    print(f"5-fold LOLA: {len(locations)} folds × {len(kinds)} MLPs × {args.epochs} epochs")
    print("=" * 72)

    for fold_idx, held_loc in enumerate(locations, 1):
        print(f"\n=== Fold {fold_idx}/{len(locations)}: test={held_loc} ===")
        train_mask = np.array([loc != held_loc for loc in mavd_locs])
        test_mask = ~train_mask
        assert train_mask.sum() + test_mask.sum() == 540

        # Per-fold scaler fit on TRAINING locations only
        scaler_audio = StandardScaler().fit(audio_mavd[train_mask])
        scaler_video = StandardScaler().fit(video_mavd[train_mask])
        audio_train_scaled = scaler_audio.transform(audio_mavd[train_mask]).astype(np.float32)
        audio_test_scaled = scaler_audio.transform(audio_mavd[test_mask]).astype(np.float32)
        video_train_scaled = scaler_video.transform(video_mavd[train_mask]).astype(np.float32)
        video_test_scaled = scaler_video.transform(video_mavd[test_mask]).astype(np.float32)

        y_train = mavd_y[train_mask]
        y_test = mavd_y[test_mask]

        for kind in kinds:
            X_tr = slice_input(kind, audio_train_scaled, video_train_scaled)
            X_te = slice_input(kind, audio_test_scaled, video_test_scaled)

            metrics, probs, preds, epoch_log = train_one_fold(
                kind,
                X_tr,
                y_train,
                X_te,
                y_test,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed + fold_idx,
            )
            metrics["location_id"] = held_loc
            oof[kind]["fold_metrics"].append(metrics)
            oof[kind]["epoch_log"].append(
                {"fold": fold_idx, "location_id": held_loc, "epochs": epoch_log}
            )
            test_indices = np.where(test_mask)[0]
            oof[kind]["probs"][test_indices] = probs
            oof[kind]["preds"][test_indices] = preds

            print(
                f"  {kind:8s}  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  AUC={metrics['roc_auc']:.4f}  "
                f"({metrics['n_params']:,} params)"
            )

    # --- Pooled OOF metrics per MLP ---
    print("\n" + "=" * 72)
    print("POOLED MAVD OOF METRICS")
    print("=" * 72)
    print(f"{'model':10s} {'F1':>8s} {'P':>8s} {'R':>8s} {'ROC-AUC':>8s} {'AP':>8s}")
    from sklearn.metrics import brier_score_loss, confusion_matrix

    pooled = {}
    for kind in kinds:
        probs = oof[kind]["probs"]
        y = mavd_y
        preds = (probs >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        m = {
            "f1": float(f1_score(y, preds, zero_division=0)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probs)),
            "ap": float(average_precision_score(y, probs)),
            "brier": float(brier_score_loss(y, probs)),
            "predicted_positive_rate": float((preds == 1).mean()),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }
        pooled[kind] = m
        print(
            f"{kind:10s} {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} "
            f"{m['roc_auc']:>8.4f} {m['ap']:>8.4f}  "
            f"brier={m['brier']:.4f}  pos_rate={m['predicted_positive_rate']:.3f}"
        )

    # Also report confusion matrices separately
    print("\nConfusion matrices (MAVD pooled OOF, threshold 0.5):")
    for kind in kinds:
        cm = pooled[kind]["confusion_matrix"]
        print(
            f"  {kind:10s}  TN={cm['tn']:3d}  FP={cm['fp']:3d}  FN={cm['fn']:3d}  TP={cm['tp']:3d}"
        )

    # --- Per-location F1 with sample counts ---
    print("\n" + "=" * 72)
    print("PER-LOCATION F1")
    print("=" * 72)
    # Print location header with (n, pos) counts
    for loc in locations:
        mask = np.array([loc_name == loc for loc_name in mavd_locs])
        n = int(mask.sum())
        pos = int(mavd_y[mask].sum())
        print(f"  {loc:20s}  n={n:3d}  pos={pos:3d}")
    print()
    print(f"{'model':10s} " + " ".join(f"{loc:>16s}" for loc in locations))
    per_loc = {kind: {} for kind in kinds}
    for kind in kinds:
        row = []
        for loc in locations:
            mask = np.array([loc_name == loc for loc_name in mavd_locs])
            f1 = f1_score(
                mavd_y[mask], (oof[kind]["probs"][mask] >= 0.5).astype(int), zero_division=0
            )
            row.append(f1)
            per_loc[kind][loc] = float(f1)
        print(f"{kind:10s} " + " ".join(f"{x:>16.4f}" for x in row))

    # --- TAU external (train on all 540 MAVD) ---
    print("\n" + "=" * 72)
    print("TAU EXTERNAL (train on all 540 MAVD)")
    print("=" * 72)
    scaler_audio_full = StandardScaler().fit(audio_mavd)
    scaler_video_full = StandardScaler().fit(video_mavd)
    audio_mavd_scaled = scaler_audio_full.transform(audio_mavd).astype(np.float32)
    video_mavd_scaled = scaler_video_full.transform(video_mavd).astype(np.float32)
    audio_tau_scaled = scaler_audio_full.transform(audio_tau).astype(np.float32)
    video_tau_scaled = scaler_video_full.transform(video_tau).astype(np.float32)

    print(f"{'model':10s} {'F1':>8s} {'P':>8s} {'R':>8s} {'ROC-AUC':>8s} {'AP':>8s}")
    tau_metrics = {}
    tau_probs_dict = {}
    for kind in kinds:
        X_tr = slice_input(kind, audio_mavd_scaled, video_mavd_scaled)
        X_te = slice_input(kind, audio_tau_scaled, video_tau_scaled)
        metrics, probs, preds, _ = train_one_fold(
            kind,
            X_tr,
            mavd_y,
            X_te,
            tau_y,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed + 100,
        )
        tau_metrics[kind] = metrics
        tau_probs_dict[kind] = probs
        print(
            f"{kind:10s} {metrics['f1']:>8.4f} {metrics['precision']:>8.4f} "
            f"{metrics['recall']:>8.4f} {metrics['roc_auc']:>8.4f} {metrics['ap']:>8.4f}"
        )

    # --- Paired bootstrap CIs ---
    print("\n" + "=" * 72)
    print("PAIRED BOOTSTRAP CIs (MAVD pooled OOF, 1000 resamples)")
    print("=" * 72)
    locs_arr = np.array(mavd_locs)

    # Load late fusion max probabilities from Week 9c
    late_fusion_oof = pd.read_csv(RESULTS_DIR / "fusion_oof.csv")
    # Column is p_max
    late_pmax = late_fusion_oof.set_index("filename").loc[mavd_fns, "p_max"].values

    comparisons = [
        ("audio", "feature_fusion vs audio_only_mlp"),     # clean vs clean
        ("video", "feature_fusion vs video_only_mlp"),     # clean vs clean (both Kinetics-only in C2b)
    ]
    # late_fusion_max REMOVED — contaminated reference

    bootstrap_results = {}
    for baseline_key, label in comparisons:
        # Baseline is A, feature fusion is B. Positive diff = fusion beats baseline.
        if baseline_key == "late_fusion_max":
            probs_a = late_pmax
        else:
            probs_a = oof[baseline_key]["probs"]
        probs_b = oof["fusion"]["probs"]

        for unit in ["clip", "location"]:
            r = bootstrap_f1_diff(mavd_y, probs_a, probs_b, locs_arr, resample_unit=unit)
            bootstrap_results[f"{label} [{unit}]"] = r
            print(f"\n{label}  ({unit}-level):")
            print(f"  mean diff = {r['mean']:+.4f}  (positive = fusion beats baseline)")
            print(f"  95% CI    = [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
            print(f"  P(fusion > baseline) = {r['p_b_beats_a']:.3f}")

    # --- Save artifacts ---
    # Derive fold number from location
    loc_to_fold = {loc: i + 1 for i, loc in enumerate(locations)}
    fusion_oof = pd.DataFrame(
        {
            "filename": mavd_fns,
            "y_true": mavd_y,
            "location_id": mavd_locs,
            "fold": [loc_to_fold[loc] for loc in mavd_locs],
            "p_audio_mlp": oof["audio"]["probs"],
            "p_video_mlp": oof["video"]["probs"],
            "p_fusion_mlp": oof["fusion"]["probs"],
        }
    )
    fusion_oof.to_csv(RESULTS_DIR / f"feature_fusion_oof{out_suffix}.csv", index=False)

    tau_df = pd.DataFrame(
        {
            "filename": tau_fns,
            "y_true": tau_y,
            "p_audio_mlp": tau_probs_dict["audio"],
            "p_video_mlp": tau_probs_dict["video"],
            "p_fusion_mlp": tau_probs_dict["fusion"],
        }
    )
    tau_df.to_csv(RESULTS_DIR / f"feature_fusion_tau_external{out_suffix}.csv", index=False)

    results = {
        "config": vars(args),
        "device": DEVICE,
        "week9c_baselines": WEEK_9C_BASELINES,
        "pooled_oof": pooled,
        "per_location": per_loc,
        "tau_external": tau_metrics,
        "bootstrap_cis": bootstrap_results,
    }
    with open(RESULTS_DIR / f"feature_fusion_results{out_suffix}.json", "w") as f:
        json.dump(results, f, indent=2)

    training_logs = {
        "epochs_per_fold": {kind: oof[kind]["epoch_log"] for kind in kinds},
        "hyperparams": vars(args),
    }
    with open(RESULTS_DIR / f"feature_fusion_training_logs{out_suffix}.json", "w") as f:
        json.dump(training_logs, f, indent=2)

    metadata = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "device": DEVICE,
        "embeddings": {
            "audio": "data/urbansas_results/audio_embeddings.npy (1080, 768)",
            "video": (f"data/urbansas_results/video_embeddings{out_suffix}.npy (1080, 512)"),
            "video_variant": args.video_embedding_variant,
        },
        "protocol": "5-fold LOLA on MAVD; per-fold StandardScaler on train only; 20 fixed epochs",
        "reference_baselines_week9c": WEEK_9C_BASELINES,
    }
    with open(RESULTS_DIR / f"feature_fusion_metadata{out_suffix}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 72)
    print("SAVED")
    print("=" * 72)
    print(f"  {RESULTS_DIR / f'feature_fusion_oof{out_suffix}.csv'}")
    print(f"  {RESULTS_DIR / f'feature_fusion_tau_external{out_suffix}.csv'}")
    print(f"  {RESULTS_DIR / f'feature_fusion_results{out_suffix}.json'}")
    print(f"  {RESULTS_DIR / f'feature_fusion_training_logs{out_suffix}.json'}")
    print(f"  {RESULTS_DIR / f'feature_fusion_metadata{out_suffix}.json'}")


if __name__ == "__main__":
    main()
