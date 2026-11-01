"""Temperature scaling for Urbansas calibration (W10.2).

Leakage-safe per-fold protocol.

For each MAVD honest model, for each fold L held out:
    - Fit T on the OOF predictions from the 4 non-L folds
    - Apply T to L's predictions
    - Evaluate calibrated ECE on L
Report per-fold and pooled calibrated ECE.

For TAU:
    - Fit T on all honest MAVD OOF for the same model
    - Apply T to TAU predictions
    - Evaluate calibrated ECE on TAU
    - Never use TAU labels for fitting or threshold selection

Models calibrated (per W10.1 bin-level analysis):
    - ast_audio: MAVD + TAU
    - feature_fusion_mlp: MAVD + TAU
    - feature_fusion_audio_only: MAVD + TAU
    - feature_fusion_video_only: MAVD + TAU
    - video_finetuned: diagnostic-only on TAU (see below)

video_finetuned:
    MAVD predictions are contaminated (encoder trained on all 540 MAVD
    clips). Cannot fit a calibrator for MAVD. Cannot fit a calibrator
    for TAU from MAVD source data. Diagnostic-only: fit T using the
    contaminated MAVD predictions and report what T would be, but do
    NOT adopt the calibrated predictions. If fitted T is close to 1.0,
    that suggests the model's TAU calibration is not accidental. If
    T is far from 1.0, that suggests the TAU calibration is a fragile
    coincidence. This is diagnostic evidence, not a deployment
    recommendation.

Late-fusion models (late_fusion_fromscratch, late_fusion_ast):
    Excluded. Their MAVD predictions are contaminated (video input).
    late_fusion_ast also has a missing TAU file.

Methodological notes:
    - Temperature scaling is monotonic on logits: ROC-AUC and AP must
      be identical before/after (sanity assertion in code).
    - F1@0.5 may change due to threshold-placement effect — this is
      NOT a discrimination improvement.
    - The MAVD per-fold calibrator is fit on cross-fitted predictions,
      so the calibration distribution and final-model prediction
      distribution may differ. Documented as a caveat.

Outputs:
    data/urbansas_results/calibration_temperature_results.json
    data/urbansas_results/calibration_plots/w10_2/{model}_{dataset}_{before,after}.png
    data/urbansas_results/calibration_plots/w10_2/{model}_{dataset}_before_after.png

Runtime: ~1 min (one logistic regression per model per fold; fast).

Runs:
    uv run python scripts/calibrate_with_temperature.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

RESULTS_DIR = _REPO_ROOT / "data" / "urbansas_results"
PLOTS_DIR = RESULTS_DIR / "calibration_plots" / "w10_2"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Logit helpers ---

def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Clip probabilities and return logit."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    """Temperature-scaled probability: sigmoid(logit(p) / T).

    T > 1 softens (probabilities toward 0.5). T < 1 sharpens.
    """
    return _sigmoid(_logit(p) / T)


def _fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    """Fit temperature T minimizing NLL via gradient descent in PyTorch.

    Equivalent to logistic regression on the single feature logit(p),
    without an intercept. Standard "temperature scaling" approach.

    Returns T (positive scalar).
    """
    logits = torch.tensor(_logit(p), dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32)

    # Parameterize T via exp(log_T) so T is always positive
    log_T = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_T)
        scaled_logits = logits / T
        loss = nn.functional.binary_cross_entropy_with_logits(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_value = float(torch.exp(log_T).item())
    return T_value


# --- ECE helpers (same as W10.1) ---

def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15, strategy: str = "uniform"
) -> float:
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    total = len(y_prob)
    ece = 0.0
    for bin_i in range(len(bin_edges) - 1):
        mask = bin_indices == bin_i
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += (n_in_bin / total) * abs(avg_acc - avg_conf)
    return ece


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    preds = (p >= 0.5).astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    out = {
        "n": int(len(y)),
        "n_pos": n_pos,
        "prevalence": float(y.mean()),
        "mean_prob": float(p.mean()),
        "mean_prob_minus_prevalence": float(p.mean() - y.mean()),
        "ece_uniform": _expected_calibration_error(y, p, n_bins=15, strategy="uniform"),
        "ece_adaptive": _expected_calibration_error(y, p, n_bins=15, strategy="quantile"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) >= 2 else float("nan"),
        "ap": float(average_precision_score(y, p)) if n_pos > 0 else float("nan"),
        "f1_at_0_5": float(f1_score(y, preds, zero_division=0)),
        "precision_at_0_5": float((preds[y == 1]).mean()) if n_pos > 0 and (preds == 1).sum() > 0 else 0.0,
        "recall_at_0_5": float(((preds == 1) & (y == 1)).sum() / max(n_pos, 1)),
    }
    return out


def _plot_before_after(y: np.ndarray, p_before: np.ndarray, p_after: np.ndarray,
                       title: str, out_path: Path, n_bins: int = 15):
    """Reliability diagram with before (blue) and after (orange) curves."""
    def bin_curve(y, p):
        edges = np.linspace(0, 1, n_bins + 1)
        idxs = np.digitize(p, edges[1:-1])
        confs, accs = [], []
        for i in range(n_bins):
            m = idxs == i
            if m.sum() == 0:
                continue
            confs.append(p[m].mean())
            accs.append(y[m].mean())
        return confs, accs

    c_b, a_b = bin_curve(y, p_before)
    c_a, a_a = bin_curve(y, p_after)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(c_b, a_b, "o-", color="#1f77b4", label="Before")
    ax.plot(c_a, a_a, "s-", color="#ff7f0e", label="After")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


# --- Data loading ---

def _load_mavd_predictions() -> dict:
    """Load the 4 honest MAVD prediction files with fold info."""
    out = {}

    # ast_audio: audio_pretrained_oof.csv
    ast = pd.read_csv(RESULTS_DIR / "audio_pretrained_oof.csv")
    ast["location_id"] = ast["filename"].str.rsplit("_", n=2).str[0]
    out["ast_audio"] = {"y": ast["y_true"].astype(int).to_numpy(),
                        "p": ast["y_prob"].astype(float).to_numpy(),
                        "loc": ast["location_id"].to_numpy()}

    # feature fusion C2b: 3 columns from feature_fusion_oof_kinetics.csv
    ff = pd.read_csv(RESULTS_DIR / "feature_fusion_oof_kinetics.csv")
    for name, col in [
        ("feature_fusion_mlp", "p_fusion_mlp"),
        ("feature_fusion_audio_only", "p_audio_mlp"),
        ("feature_fusion_video_only", "p_video_mlp"),
    ]:
        out[name] = {"y": ff["y_true"].astype(int).to_numpy(),
                     "p": ff[col].astype(float).to_numpy(),
                     "loc": ff["location_id"].to_numpy()}

    return out


def _load_tau_predictions() -> dict:
    """Load TAU predictions for models that can be calibrated or diagnosed."""
    out = {}

    ast = pd.read_csv(RESULTS_DIR / "audio_pretrained_tau_external.csv")
    out["ast_audio"] = {"y": ast["y_true"].astype(int).to_numpy(),
                        "p": ast["y_prob"].astype(float).to_numpy()}

    ff = pd.read_csv(RESULTS_DIR / "feature_fusion_tau_external_kinetics.csv")
    for name, col in [
        ("feature_fusion_mlp", "p_fusion_mlp"),
        ("feature_fusion_audio_only", "p_audio_mlp"),
        ("feature_fusion_video_only", "p_video_mlp"),
    ]:
        out[name] = {"y": ff["y_true"].astype(int).to_numpy(),
                     "p": ff[col].astype(float).to_numpy()}

    vid = pd.read_csv(RESULTS_DIR / "urbansas_video_tau_external.csv")
    out["video_finetuned"] = {"y": vid["y_true"].astype(int).to_numpy(),
                              "p": vid["y_prob"].astype(float).to_numpy()}

    return out


# --- Main calibration workflows ---

def calibrate_mavd(name: str, data: dict) -> dict:
    """Per-fold MAVD calibration (leakage-safe)."""
    y = data["y"]
    p = data["p"]
    loc = data["loc"]

    before_metrics = _metrics(y, p)

    locations = sorted(np.unique(loc))
    per_fold = []
    pooled_before = np.zeros(len(y))
    pooled_after = np.zeros(len(y))

    for fold_idx, held in enumerate(locations, 1):
        train_mask = loc != held
        test_mask = loc == held

        # Fit T on the 4 training folds
        T = _fit_temperature(p[train_mask], y[train_mask])

        # Apply to held-out fold
        p_cal = _apply_temperature(p[test_mask], T)

        fold_metrics = {
            "fold": fold_idx,
            "held_location": held,
            "n_test": int(test_mask.sum()),
            "n_pos_test": int(y[test_mask].sum()),
            "temperature": T,
            "before": _metrics(y[test_mask], p[test_mask]),
            "after": _metrics(y[test_mask], p_cal),
        }
        per_fold.append(fold_metrics)

        pooled_before[test_mask] = p[test_mask]
        pooled_after[test_mask] = p_cal

    pooled_before_metrics = _metrics(y, pooled_before)
    pooled_after_metrics = _metrics(y, pooled_after)

    # Sanity: per-fold AUC/AP must be identical (temperature scaling is monotonic
    # within a fold). Pooled AUC/AP may differ because folds fit different T's —
    # that's a consequence of the leakage-safe protocol, not a bug.
    sanity_ok = True
    fold_auc_diffs = []
    fold_ap_diffs = []
    for fold in per_fold:
        if np.isnan(fold["before"]["roc_auc"]):
            continue
        fold_auc_diff = abs(fold["before"]["roc_auc"] - fold["after"]["roc_auc"])
        fold_ap_diff = abs(fold["before"]["ap"] - fold["after"]["ap"])
        fold_auc_diffs.append(fold_auc_diff)
        fold_ap_diffs.append(fold_ap_diff)
        if fold_auc_diff > 1e-6 or fold_ap_diff > 1e-6:
            sanity_ok = False
            print(
                f"  WARNING: fold {fold['fold']} ({fold['held_location']}) "
                f"AUC diff={fold_auc_diff:.2e}, AP diff={fold_ap_diff:.2e}"
            )

    if not sanity_ok:
        raise RuntimeError(
            f"SANITY CHECK FAILED for {name}: per-fold AUC/AP not preserved. "
            f"Temperature scaling is monotonic within each fold; if this fails "
            f"the implementation is buggy."
        )

    # Report pooled AUC/AP change as informational (not a sanity failure)
    pooled_auc_diff = abs(pooled_before_metrics["roc_auc"] - pooled_after_metrics["roc_auc"])
    pooled_ap_diff = abs(pooled_before_metrics["ap"] - pooled_after_metrics["ap"])
    print(
        f"    Pooled AUC diff: {pooled_auc_diff:.4f}  "
        f"AP diff: {pooled_ap_diff:.4f}  "
        f"(expected nonzero — per-fold T's)"
    )

    # Plot before/after
    plot_path = PLOTS_DIR / f"mavd_{name}_before_after.png"
    _plot_before_after(y, pooled_before, pooled_after,
                       f"MAVD: {name} (temperature scaling per fold)", plot_path)

    return {
        "model": name,
        "dataset": "mavd",
        "per_fold": per_fold,
        "pooled_before": pooled_before_metrics,
        "pooled_after": pooled_after_metrics,
        "per_fold_auc_ap_preserved": sanity_ok,
        "pooled_auc_diff": pooled_auc_diff,
        "pooled_ap_diff": pooled_ap_diff,
        "temperature_range": [f["temperature"] for f in per_fold],
        "temperature_min": float(min(f["temperature"] for f in per_fold)),
        "temperature_max": float(max(f["temperature"] for f in per_fold)),
        "reliability_plot": str(plot_path.relative_to(_REPO_ROOT)),
        "note": (
            "Pooled AUC/AP may differ from before because each fold uses a "
            "different temperature T. This is a consequence of the leakage-safe "
            "per-fold protocol, not a bug. Per-fold AUC/AP are asserted identical."
        ),
    }


def calibrate_tau(name: str, mavd_data: dict, tau_data: dict) -> dict:
    """Fit T on all honest MAVD OOF, apply to TAU."""
    y_mavd = mavd_data["y"]
    p_mavd = mavd_data["p"]
    y_tau = tau_data["y"]
    p_tau = tau_data["p"]

    # Fit T on all MAVD
    T = _fit_temperature(p_mavd, y_mavd)

    # Apply to TAU
    p_cal = _apply_temperature(p_tau, T)

    before_metrics = _metrics(y_tau, p_tau)
    after_metrics = _metrics(y_tau, p_cal)

    # Sanity check
    auc_diff = abs(before_metrics["roc_auc"] - after_metrics["roc_auc"])
    ap_diff = abs(before_metrics["ap"] - after_metrics["ap"])
    sanity_ok = auc_diff < 1e-6 and ap_diff < 1e-6
    if not sanity_ok:
        raise RuntimeError(
            f"SANITY CHECK FAILED for TAU {name}: AUC diff={auc_diff:.2e}, "
            f"AP diff={ap_diff:.2e}"
        )

    # Plot
    plot_path = PLOTS_DIR / f"tau_{name}_before_after.png"
    _plot_before_after(y_tau, p_tau, p_cal,
                       f"TAU: {name} (T={T:.3f} fit on MAVD)", plot_path)

    return {
        "model": name,
        "dataset": "tau",
        "temperature_fit_on_mavd": T,
        "before": before_metrics,
        "after": after_metrics,
        "sanity_auc_ap_unchanged": sanity_ok,
        "reliability_plot": str(plot_path.relative_to(_REPO_ROOT)),
    }


def diagnostic_video_finetuned_tau(mavd_contaminated: dict, tau_data: dict) -> dict:
    """Diagnostic only — do NOT adopt calibrated predictions.

    Fits T on the contaminated MAVD predictions and reports the fitted T.
    This tests whether video_finetuned's TAU calibration is a coincidence
    (T far from 1.0) or genuine (T close to 1.0).
    """
    T = _fit_temperature(mavd_contaminated["p"], mavd_contaminated["y"])
    y_tau = tau_data["y"]
    p_tau = tau_data["p"]

    return {
        "model": "video_finetuned",
        "dataset": "tau",
        "note": "DIAGNOSTIC ONLY. Calibrator fitted on contaminated MAVD predictions. Do not adopt.",
        "temperature_fit_on_contaminated_mavd": T,
        "interpretation": (
            "T near 1.0 suggests the TAU calibration is not accidental; "
            "T far from 1.0 suggests it is a fragile coincidence that a "
            "properly fit calibration would break."
        ),
        "before": _metrics(y_tau, p_tau),
        # Do not report "after" — the calibrator isn't trustworthy
    }


# --- Main ---

def main():
    print(f"Outputs: {PLOTS_DIR}")

    mavd_preds = _load_mavd_predictions()
    tau_preds = _load_tau_predictions()

    models_to_calibrate = [
        "ast_audio",
        "feature_fusion_mlp",
        "feature_fusion_audio_only",
        "feature_fusion_video_only",
    ]

    all_results = {"mavd": [], "tau": [], "diagnostic": []}

    # MAVD per-fold calibration
    print("\n=== MAVD: per-fold temperature scaling ===")
    for name in models_to_calibrate:
        print(f"  Calibrating {name} ...")
        result = calibrate_mavd(name, mavd_preds[name])
        all_results["mavd"].append(result)
        temps = [f["temperature"] for f in result["per_fold"]]
        print(
            f"    Temperatures: {temps}  "
            f"ECE: {result['pooled_before']['ece_uniform']:.4f} → "
            f"{result['pooled_after']['ece_uniform']:.4f}  "
            f"AUC: {result['pooled_before']['roc_auc']:.4f} → "
            f"{result['pooled_after']['roc_auc']:.4f}"
        )

    # TAU transfer (fit on MAVD, apply to TAU)
    print("\n=== TAU: temperature transfer (fit on MAVD) ===")
    for name in models_to_calibrate:
        print(f"  Transferring {name} ...")
        result = calibrate_tau(name, mavd_preds[name], tau_preds[name])
        all_results["tau"].append(result)
        print(
            f"    T = {result['temperature_fit_on_mavd']:.4f}  "
            f"ECE: {result['before']['ece_uniform']:.4f} → "
            f"{result['after']['ece_uniform']:.4f}  "
            f"AUC: {result['before']['roc_auc']:.4f} → "
            f"{result['after']['roc_auc']:.4f}"
        )

    # video_finetuned diagnostic (contaminated MAVD → TAU)
    print("\n=== Diagnostic: video_finetuned TAU (contaminated MAVD fit) ===")
    vid_mavd_contaminated = {
        "y": pd.read_csv(RESULTS_DIR / "urbansas_video_oof.csv")["y_true"].astype(int).to_numpy(),
        "p": pd.read_csv(RESULTS_DIR / "urbansas_video_oof.csv")["y_prob"].astype(float).to_numpy(),
    }
    diag = diagnostic_video_finetuned_tau(vid_mavd_contaminated, tau_preds["video_finetuned"])
    all_results["diagnostic"].append(diag)
    print(f"    Diagnostic T = {diag['temperature_fit_on_contaminated_mavd']:.4f}")
    print(f"    TAU ECE before: {diag['before']['ece_uniform']:.4f}  (no 'after' reported)")

    # Save
    out_path = RESULTS_DIR / "calibration_temperature_results.json"

    def _sanitize(obj):
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        if obj is Ellipsis:
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    sanitized = _sanitize(all_results)

    def _sanitize(obj):
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        if obj is Ellipsis:
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_sanitize(all_results), f, indent=2)

    print(f"\nSaved: {out_path}")
    print(f"Plots: {PLOTS_DIR}")
    print(f"  ({len(all_results['mavd']) + len(all_results['tau'])} before/after plots)")


if __name__ == "__main__":
    main()
