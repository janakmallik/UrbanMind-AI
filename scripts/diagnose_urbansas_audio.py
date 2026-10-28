"""Diagnose Urbansas audio-only OOF predictions.

Reproduces the per-location, per-subclass, night/day, and
vehicle-composition diagnostics discussed during the audio-only
experiment writeup.

Reads:
    --oof PATH       OOF predictions (default: data/urbansas_results/audio_only_oof.csv)
    --tau PATH       TAU external (default: data/urbansas_results/audio_only_tau_external.csv)
    --annotations    Annotation CSVs via urbansas_root()

Writes nothing to disk. Prints to stdout.

Assumptions (documented, not hidden):
    - Fold order in scripts/train_urbansas_audio.py (line 207) is
      `sorted(mavd_df["location_id"].unique())`, alphabetically.
    - Fold number = alphabetical position, 1-indexed (line 225).
    - The OOF CSV was produced by the same run that produced the
      audio_only_final.pt checkpoint. This was verified manually
      before this script was committed; the per-fold F1 match is
      recorded in docs/model_cards/urbansas_audio_cnn_v1.md.

Read-only. Does not modify any artifact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from urbanintel.config import urbansas_root

VEHICLE_LABELS = {"car", "bus", "motorbike", "truck"}


def _subclass(labels: set[str]) -> str:
    has_off = "offscreen" in labels
    has_veh = bool(labels & VEHICLE_LABELS)
    if labels == {"-1"}:
        return "empty"
    if has_off and has_veh:
        return "both"
    if has_off:
        return "only_offscreen"
    if has_veh:
        return "only_visible"
    return "other"


def _threshold_sweep(y: np.ndarray, p: np.ndarray, thresholds) -> None:
    print(f"{'threshold':>10s} {'n_pred':>7s} {'precision':>10s} {'recall':>7s} {'f1':>7s}")
    for t in thresholds:
        pred = (p >= t).astype(int)
        print(
            f"{t:>10.2f} {pred.sum():>7d} "
            f"{precision_score(y, pred, zero_division=0):>10.3f} "
            f"{recall_score(y, pred, zero_division=0):>7.3f} "
            f"{f1_score(y, pred, zero_division=0):>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oof",
        default=str(_REPO_ROOT / "data" / "urbansas_results" / "audio_only_oof.csv"),
    )
    parser.add_argument(
        "--tau",
        default=str(_REPO_ROOT / "data" / "urbansas_results" / "audio_only_tau_external.csv"),
    )
    args = parser.parse_args()

    root = urbansas_root()
    audio_ann = pd.read_csv(root / "annotations" / "audio_annotations.csv")
    video_ann = pd.read_csv(root / "annotations" / "video_annotations.csv")
    clip_meta = video_ann.drop_duplicates("filename")[
        ["filename", "location_id", "subset", "night"]
    ]

    label_sets = audio_ann.groupby("filename")["label"].agg(lambda x: set(x.astype(str)))
    clip_info = clip_meta.copy()
    clip_info["subclass"] = clip_info["filename"].map(label_sets).map(_subclass)
    clip_info["has_offscreen"] = (
        clip_info["filename"].map(label_sets).apply(lambda s: "offscreen" in s).astype(int)
    )

    # --- MAVD OOF ---
    oof = pd.read_csv(args.oof)
    mavd = clip_info[clip_info["subset"] == "mavd"].merge(oof, on="filename", how="inner")
    y = mavd["y_true"].astype(int).to_numpy()
    p = mavd["y_prob"].astype(float).to_numpy()

    print("=" * 72)
    print(f"MAVD OOF — {len(mavd)} clips, {int(y.sum())} positives")
    print("=" * 72)
    print(f"  ROC-AUC:            {roc_auc_score(y, p):.4f}")
    print(f"  Average precision:  {average_precision_score(y, p):.4f}")
    print(f"  Mean prob (neg):    {p[y == 0].mean():.4f}")
    print(f"  Mean prob (pos):    {p[y == 1].mean():.4f}")
    print()
    print("  Threshold sweep (MAVD):")
    _threshold_sweep(y, p, [0.1, 0.2, 0.3, 0.4, 0.5])
    print()

    # --- Per-location ---
    print("=" * 72)
    print("MAVD per-location ROC-AUC")
    print("=" * 72)
    print(f"{'location':22s} {'n':>4s} {'pos':>4s} {'ROC-AUC':>9s} {'note':>20s}")
    for loc, g in mavd.groupby("location_id"):
        if g["y_true"].nunique() < 2:
            note = "only one class"
            print(f"{loc:22s} {len(g):>4d} {int(g['y_true'].sum()):>4d} {'n/a':>9s} {note:>20s}")
            continue
        auc = roc_auc_score(g["y_true"], g["y_prob"])
        note = "chance (n<5 pos)" if g["y_true"].sum() < 5 else ""
        print(f"{loc:22s} {len(g):>4d} {int(g['y_true'].sum()):>4d} {auc:>9.3f} {note:>20s}")
    print()

    # --- Per-subclass vs negatives ---
    print("=" * 72)
    print("MAVD per-subclass ROC-AUC vs negative pool")
    print("=" * 72)
    neg = mavd[mavd["subclass"].isin(["only_visible", "empty"])]
    print(f"Negative pool (only_visible + empty): n={len(neg)}")
    for sub in ["only_offscreen", "both"]:
        pos = mavd[mavd["subclass"] == sub]
        if len(pos) == 0:
            continue
        combo = pd.concat([pos, neg])
        yy = combo["subclass"].isin([sub]).astype(int)
        auc = roc_auc_score(yy, combo["y_prob"])
        print(f"  {sub:16s} vs negatives:  n_pos={len(pos):>3d}  ROC-AUC={auc:.3f}")
    print()

    # --- Night / day ---
    print("=" * 72)
    print("MAVD night/day cross-tab")
    print("=" * 72)
    print(pd.crosstab(mavd["night"], mavd["location_id"]).to_string())
    print()

    # --- Vehicle composition ---
    print("=" * 72)
    print("MAVD per-location vehicle-label share (% of clips)")
    print("=" * 72)
    per_clip_labels = audio_ann.groupby("filename")["label"].agg(lambda x: set(x.astype(str)))
    mavd["labels_set"] = mavd["filename"].map(per_clip_labels)
    print(f"{'location':22s} {'car%':>6s} {'bus%':>6s} {'moto%':>6s} {'off%':>6s}")
    for loc, g in mavd.groupby("location_id"):
        n = len(g)
        car = g["labels_set"].apply(lambda s: "car" in s).mean() * 100
        bus = g["labels_set"].apply(lambda s: "bus" in s).mean() * 100
        moto = g["labels_set"].apply(lambda s: "motorbike" in s).mean() * 100
        off = g["labels_set"].apply(lambda s: "offscreen" in s).mean() * 100
        print(f"{loc:22s} {car:>6.1f} {bus:>6.1f} {moto:>6.1f} {off:>6.1f}")
    print()

    # --- TAU external ---
    tau = pd.read_csv(args.tau)
    tau["location_id"] = tau["filename"].str.rsplit("_", n=1).str[0]
    ty = tau["y_true"].astype(int).to_numpy()
    tp = tau["y_prob"].astype(float).to_numpy()
    print("=" * 72)
    print(f"TAU external — {len(tau)} clips, {int(ty.sum())} positives")
    print("=" * 72)
    print(f"  ROC-AUC:            {roc_auc_score(ty, tp):.4f}")
    print(f"  Average precision:  {average_precision_score(ty, tp):.4f}")
    print(f"  Mean prob (neg):    {tp[ty == 0].mean():.4f}")
    print(f"  Mean prob (pos):    {tp[ty == 1].mean():.4f}")
    print()
    print("  Threshold sweep (TAU):")
    _threshold_sweep(ty, tp, [0.1, 0.2, 0.3, 0.4, 0.5])
    print()
    print(f"  TAU offscreen prevalence:  {ty.mean() * 100:.1f}%")
    print(f"  MAVD offscreen prevalence: {mavd['y_true'].mean() * 100:.1f}%")
    print()
    print("Diagnostic complete. No files written.")


if __name__ == "__main__":
    main()
