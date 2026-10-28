"""Late fusion experiment for Urbansas offscreen vehicle detection.

Compares six models on the same 5-fold leave-one-location-out (LOLA) folds:
    (A) audio-only   [p_audio]
    (B) video-only   [p_video]
    (C) mean         (p_audio + p_video) / 2
    (D) max          max(p_audio, p_video)
    (E) logistic     LogisticRegression([p_audio, p_video])
    (F) logistic+I   LogisticRegression([p_audio, p_video, p_audio*p_video])

For each fold:
    - Fit fusion head (E, F) on the 4 training locations' OOF probabilities.
    - Evaluate on the held-out location's OOF probabilities.
    - Threshold fixed at 0.5.
    - No hyperparameter search inside fusion head (fixed C=1.0).

The base-model OOF probabilities come from scripts/train_urbansas_audio.py
and scripts/train_urbansas_video.py. Each base model was trained leave-one-
location-out; the held-out location's predictions are always from a base
model that never saw that location. The fusion head sees only these OOF
probabilities, never the labels of the held-out location.

Bootstrap CIs:
    - Location-level (5 resampling units) — PRIMARY, honest
    - Clip-level (540 resampling units)  — SECONDARY, matches baseline reporting

Reports:
    - Pooled MAVD OOF metrics for all 6 models
    - Per-location F1 for all 6 models
    - Per-subclass F1 for all 6 models
    - Per-fold mean ± std
    - Bootstrap CIs (both levels)
    - Sign consistency across locations
    - TAU external F1 for all 6 models

Saves:
    data/urbansas_results/fusion_oof.csv
    data/urbansas_results/fusion_tau_external.csv
    data/urbansas_results/fusion_results.json

No claims about outcome. All numbers reported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from urbanintel.config import urbansas_root

RESULTS = _REPO_ROOT / "data" / "urbansas_results"
VEH = {"car", "bus", "motorbike", "truck"}


# ---------------------------------------------------------------- helpers


def _subclass(labels_set: set[str]) -> str:
    has_off = "offscreen" in labels_set
    has_veh = bool(labels_set & VEH)
    if labels_set == {"-1"}:
        return "empty"
    if has_off and has_veh:
        return "both"
    if has_off:
        return "only_offscreen"
    if has_veh:
        return "only_visible"
    return "other"


def _load_subclasses():
    root = urbansas_root()
    audio_ann = pd.read_csv(root / "annotations" / "audio_annotations.csv")
    groups = audio_ann.groupby("filename")["label"].agg(lambda x: set(x.astype(str)))
    return {fn: _subclass(lbls) for fn, lbls in groups.items()}


def _loc_from_filename(stem: str) -> str:
    """Reconstruct MAVD location from filename prefix.

    MAVD files are {location}_{run:02d}_{clip}. Splitting from the right
    on '_' twice gives the location prefix. TAU files use different
    naming; this helper is only used on MAVD.
    """
    return stem.rsplit("_", 2)[0]  # positional, no `n=`


def _metrics(y_true, y_prob, threshold=0.5):
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "n": int(len(y_true)),
        "n_pos": int(np.sum(y_true)),
    }


def _full_metrics(y_true, y_prob):
    """Includes AUCs (undefined if only one class present)."""
    out = _metrics(y_true, y_prob)
    if len(np.unique(y_true)) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


# ---------------------------------------------------------------- fusion models


def _fusion_fit_predict(train_a, train_v, train_y, test_a, test_v, kind: str):
    """Fit fusion model on train (a, v, y) and predict on test (a, v)."""
    train_a = np.asarray(train_a, dtype=float)
    train_v = np.asarray(train_v, dtype=float)
    test_a = np.asarray(test_a, dtype=float)
    test_v = np.asarray(test_v, dtype=float)

    if kind == "mean":
        return (test_a + test_v) / 2.0
    if kind == "max":
        return np.maximum(test_a, test_v)
    if kind == "logistic":
        X_tr = np.column_stack([train_a, train_v])
        X_te = np.column_stack([test_a, test_v])
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced")
        clf.fit(X_tr, train_y)
        return clf.predict_proba(X_te)[:, 1]
    if kind == "logistic_interaction":
        X_tr = np.column_stack([train_a, train_v, train_a * train_v])
        X_te = np.column_stack([test_a, test_v, test_a * test_v])
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced")
        clf.fit(X_tr, train_y)
        return clf.predict_proba(X_te)[:, 1]
    raise ValueError(f"unknown fusion kind: {kind}")


# ---------------------------------------------------------------- bootstrap


def _bootstrap_diff_ci(y_true, prob_a, prob_b, resample_unit, n_boot=1000, seed=42):
    """Paired bootstrap of (model_b_f1 - model_a_f1)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_a = (np.asarray(prob_a) >= 0.5).astype(int)
    pred_b = (np.asarray(prob_b) >= 0.5).astype(int)
    n = len(y_true)

    # Per-sample correct/incorrect arrays (paired across models)
    def f1_from_pred(y, pred):
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-12)

    diffs = []
    if resample_unit == "clip":
        idx_pool = np.arange(n)
        for _ in range(n_boot):
            idx = rng.choice(idx_pool, size=n, replace=True)
            f1_a = f1_from_pred(y_true[idx], pred_a[idx])
            f1_b = f1_from_pred(y_true[idx], pred_b[idx])
            diffs.append(f1_b - f1_a)
    elif resample_unit == "location":
        # Need location column; caller passes it via a parallel list
        # We rely on a module-level _loc_ids array set before call
        locs = _loc_ids  # set by caller
        unique_locs = sorted(set(locs))
        loc_idx_map = {loc: np.where(locs == loc)[0] for loc in unique_locs}
        for _ in range(n_boot):
            chosen = rng.choice(unique_locs, size=len(unique_locs), replace=True)
            idx = np.concatenate([loc_idx_map[loc] for loc in chosen])
            f1_a = f1_from_pred(y_true[idx], pred_a[idx])
            f1_b = f1_from_pred(y_true[idx], pred_b[idx])
            diffs.append(f1_b - f1_a)
    else:
        raise ValueError(f"unknown resample_unit: {resample_unit}")

    diffs = np.asarray(diffs)
    return {
        "mean": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "p_b_gt_a": float((diffs > 0).mean()),
    }


# ---------------------------------------------------------------- main


_loc_ids = None  # populated in main(); required for location-level bootstrap


def main():
    global _loc_ids

    # Load base OOF predictions and align
    a_df = pd.read_csv(RESULTS / "audio_pretrained_oof.csv").rename(columns={"y_prob": "p_audio"})
    v_df = pd.read_csv(RESULTS / "urbansas_video_oof.csv").rename(columns={"y_prob": "p_video"})
    m = a_df.merge(v_df[["filename", "p_video"]], on="filename", how="inner")
    assert len(m) == 540, f"expected 540 MAVD clips, got {len(m)}"
    assert (a_df.set_index("filename")["y_true"] == v_df.set_index("filename")["y_true"]).all()

    m["location"] = m["filename"].apply(_loc_from_filename)
    locs_sorted = sorted(m["location"].unique())
    print(f"MAVD clips: {len(m)}, locations: {locs_sorted}")
    print(f"Positive rate: {m['y_true'].mean():.3f}\n")

    # Load subclasses (for per-subclass breakdown)
    subclasses = _load_subclasses()
    m["subclass"] = m["filename"].map(subclasses).fillna("other")

    _loc_ids = m["location"].values  # for location-level bootstrap

    # Fitted models on all MAVD, applied to TAU, need OOF probs of TAU from
    # both base models. TAU OOF probs come from a model trained on all MAVD.
    tau_a = pd.read_csv(RESULTS / "audio_pretrained_tau_external.csv").rename(
        columns={"y_prob": "p_audio"}
    )
    tau_v = pd.read_csv(RESULTS / "urbansas_video_tau_external.csv").rename(
        columns={"y_prob": "p_video"}
    )
    tau = tau_a.merge(tau_v[["filename", "p_video"]], on="filename", how="inner")
    assert len(tau) == 540, f"expected 540 TAU clips, got {len(tau)}"
    print(f"TAU clips: {len(tau)}, positive rate: {tau['y_true'].mean():.3f}\n")

    # ------------------------------------------------ MAVD LOLA evaluation
    models = ["audio", "video", "mean", "max", "logistic", "logistic_interaction"]

    # Store per-fold predictions per model: probs per clip
    fold_preds = {name: {} for name in models}  # name -> {(fold, idx): prob}
    per_fold_metrics = {name: [] for name in models}

    for held_loc in locs_sorted:
        train = m[m["location"] != held_loc]
        test = m[m["location"] == held_loc]

        print(f"=== Held-out: {held_loc}  ({len(test)} clips, {int(test['y_true'].sum())} pos) ===")

        # Baselines: direct use of OOF probs (already from models that didn't
        # see this location — OOF probs are produced by the base model
        # trained without this location's fold, by construction)
        preds_this_fold = {}
        preds_this_fold["audio"] = test["p_audio"].values
        preds_this_fold["video"] = test["p_video"].values

        # Fusion: fit on training-location OOF probs, predict held-out
        for kind in ["mean", "max", "logistic", "logistic_interaction"]:
            preds_this_fold[kind] = _fusion_fit_predict(
                train["p_audio"].values,
                train["p_video"].values,
                train["y_true"].values,
                test["p_audio"].values,
                test["p_video"].values,
                kind,
            )

        for name in models:
            per_fold_metrics[name].append(_metrics(test["y_true"].values, preds_this_fold[name]))
            fold_preds[name].update(
                {fn: prob for fn, prob in zip(test["filename"].values, preds_this_fold[name])}
            )
            print(f"  {name:22s} F1={per_fold_metrics[name][-1]['f1']:.4f}")
        print()

    # ------------------------------------------------ pooled OOF
    y_all = m["y_true"].values
    print("=" * 72)
    print("POOLED MAVD OOF METRICS")
    print("=" * 72)
    print(f"{'model':22s} {'F1@0.5':>8s} {'P':>6s} {'R':>6s} {'ROC-AUC':>8s} {'PR-AUC':>8s}")
    pooled = {}
    for name in models:
        probs = np.array([fold_preds[name][fn] for fn in m["filename"].values])
        met = _full_metrics(y_all, probs)
        pooled[name] = met
        print(
            f"{name:22s} {met['f1']:>8.4f} {met['precision']:>6.3f} "
            f"{met['recall']:>6.3f} {met['roc_auc']:>8.4f} {met['pr_auc']:>8.4f}"
        )
    print()

    # ------------------------------------------------ per-location
    print("=" * 72)
    print("PER-LOCATION F1")
    print("=" * 72)
    print(f"{'model':22s} " + " ".join(f"{loc:>16s}" for loc in locs_sorted))
    per_loc = {}
    for name in models:
        probs = np.array([fold_preds[name][fn] for fn in m["filename"].values])
        row = []
        per_loc[name] = {}
        for loc in locs_sorted:
            mask = m["location"].values == loc
            met = _metrics(y_all[mask], probs[mask])
            row.append(met["f1"])
            per_loc[name][loc] = met["f1"]
        print(f"{name:22s} " + " ".join(f"{x:>16.4f}" for x in row))
    print()

    # Sign consistency vs audio
    print("Sign consistency (model F1 > audio-only F1, per location):")
    for name in models:
        if name == "audio":
            continue
        wins = sum(1 for loc in locs_sorted if per_loc[name][loc] > per_loc["audio"][loc])
        print(f"  {name:22s} {wins}/5 locations")
    print()

    # ------------------------------------------------ per-subclass
    print("=" * 72)
    print("PER-SUBCLASS F1 (evaluated on clips of that subclass only)")
    print("=" * 72)
    print(
        f"{'model':22s} {'only_offscreen':>16s} {'both':>10s} {'only_visible':>14s} {'empty':>10s}"
    )
    subclass_order = ["only_offscreen", "both", "only_visible", "empty"]
    per_sub = {}
    for name in models:
        probs = np.array([fold_preds[name][fn] for fn in m["filename"].values])
        row = []
        per_sub[name] = {}
        for sub in subclass_order:
            mask = m["subclass"].values == sub
            if mask.sum() == 0:
                row.append(float("nan"))
                per_sub[name][sub] = float("nan")
                continue
            met = _metrics(y_all[mask], probs[mask])
            row.append(met["f1"])
            per_sub[name][sub] = met["f1"]
        print(f"{name:22s} " + " ".join(f"{x:>16.4f}" for x in row))
    print("Note: per-subclass F1 is dominated by the class balance within each subclass.")
    print("For 'only_visible' and 'empty' there are no positive clips (F1 is trivially 0")
    print("we still report it to show the false-positive rate is not zero).")
    print()

    # ------------------------------------------------ per-fold mean ± std
    print("=" * 72)
    print("PER-FOLD F1 MEAN ± STD (across 5 LOLA folds)")
    print("=" * 72)
    fold_stats = {}
    for name in models:
        f1s = [met["f1"] for met in per_fold_metrics[name]]
        fold_stats[name] = {"mean": float(np.mean(f1s)), "std": float(np.std(f1s))}
        print(f"{name:22s} {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print()

    # ------------------------------------------------ bootstrap CIs
    print("=" * 72)
    print("BOOTSTRAP CI ON F1 DIFFERENCE (1000 resamples, paired)")
    print("=" * 72)
    probs_all = {
        name: np.array([fold_preds[name][fn] for fn in m["filename"].values]) for name in models
    }
    boot = {}
    for compare_name in ["logistic", "logistic_interaction"]:
        boot[compare_name] = {}
        for unit in ["clip", "location"]:
            ci = _bootstrap_diff_ci(
                y_all, probs_all["audio"], probs_all[compare_name], resample_unit=unit
            )
            boot[compare_name][unit] = ci
            print(f"{compare_name:22s} vs audio  ({unit}-level resample):")
            print(f"  mean diff = {ci['mean']:+.4f}")
            print(f"  95% CI    = [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
            print(f"  P({compare_name} > audio) = {ci['p_b_gt_a']:.3f}")
        print()

    # ------------------------------------------------ TAU external
    print("=" * 72)
    print("TAU EXTERNAL (base models trained on ALL MAVD; fusion head fit on ALL MAVD OOF)")
    print("=" * 72)
    # For TAU: base models' TAU probabilities come from models trained on all
    # MAVD. Fusion head fit on all 540 MAVD OOF probs, applied to TAU probs.
    tau_y = tau["y_true"].values
    tau_preds = {}
    tau_preds["audio"] = tau["p_audio"].values
    tau_preds["video"] = tau["p_video"].values
    for kind in ["mean", "max", "logistic", "logistic_interaction"]:
        tau_preds[kind] = _fusion_fit_predict(
            m["p_audio"].values,
            m["p_video"].values,
            m["y_true"].values,
            tau["p_audio"].values,
            tau["p_video"].values,
            kind,
        )

    print(f"{'model':22s} {'F1':>8s} {'P':>6s} {'R':>6s}")
    tau_metrics = {}
    for name in models:
        met = _metrics(tau_y, tau_preds[name])
        tau_metrics[name] = met
        print(f"{name:22s} {met['f1']:>8.4f} {met['precision']:>6.3f} {met['recall']:>6.3f}")
    print()

    # ------------------------------------------------ save artifacts
    fusion_oof = m[["filename", "y_true"]].copy()
    for name in models:
        fusion_oof[f"p_{name}"] = [fold_preds[name][fn] for fn in m["filename"].values]
    fusion_oof.to_csv(RESULTS / "fusion_oof.csv", index=False)

    tau_out = tau[["filename", "y_true"]].copy()
    for name in models:
        tau_out[f"p_{name}"] = tau_preds[name]
    tau_out.to_csv(RESULTS / "fusion_tau_external.csv", index=False)

    results_json = {
        "mavd_pooled": pooled,
        "mavd_per_location": per_loc,
        "mavd_per_subclass": per_sub,
        "mavd_per_fold_mean_std": fold_stats,
        "bootstrap_ci": boot,
        "tau_external": tau_metrics,
    }
    with open(RESULTS / "fusion_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Saved: {RESULTS / 'fusion_oof.csv'}")
    print(f"Saved: {RESULTS / 'fusion_tau_external.csv'}")
    print(f"Saved: {RESULTS / 'fusion_results.json'}")
    print()
    print("Experiment complete. All 6 models reported on all views.")


if __name__ == "__main__":
    main()
