"""Evaluate calibration of Urbansas classification models.

For each prediction CSV, produces:
    - Reliability diagram (PNG, with sample-count histogram)
    - ECE (15 equal-width bins)
    - Adaptive ECE (15 equal-size target bins, actual bins reported)
    - Brier score, log loss, ROC-AUC, AP
    - Per-bin sample counts

Read-only. No fitting, no training, no threshold tuning.

PROVENANCE CAVEAT (must be observed):
    Some MAVD OOF predictions are contaminated by encoder provenance.
    - urbansas_video_oof.csv: fine-tuned video encoder trained on all 540
      MAVD clips (verified in 9c-ext-2 audit — 222 backbone weight
      mismatches vs Kinetics-pretrained). MAVD calibration NOT reported.
    - fusion_oof.csv, fusion_oof_kinetics.csv: late fusion uses the above
      contaminated video predictions as input. MAVD calibration NOT reported.

    Clean MAVD models:
    - audio_pretrained_oof.csv (AST encoder verified byte-identical to
      AudioSet-pretrained — 0 mismatches)
    - feature_fusion_*_kinetics.csv (Kinetics-only embeddings, never
      fine-tuned on Urbansas)

    TAU external predictions are all honest (TAU never used in training
    for any model) — all included.

CALIBRATION vs DISCRIMINATION:
    - ROC-AUC and AP measure discrimination (ranking quality)
    - Brier score and log loss measure probability quality
    - ECE measures binned calibration error
    - Temperature scaling preserves binary ranking (positive scalar T
      is monotonic on logits) and therefore preserves ROC-AUC and AP.
      It does NOT create better discrimination. Any F1 change at threshold
      0.5 after calibration is a threshold-placement effect, not a
      discrimination improvement.

Usage:
    uv run python scripts/evaluate_calibration.py
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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

RESULTS_DIR = _REPO_ROOT / "data" / "urbansas_results"
PLOTS_DIR = RESULTS_DIR / "calibration_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Each entry: (name, path, probability_column, provenance_valid)
# provenance_valid=False means MAVD calibration evaluation is NOT reported
# (predictions are in-sample with respect to the encoder).
MAVD_MODELS = [
    ("ast_audio", RESULTS_DIR / "audio_pretrained_oof.csv", "y_prob", True),
    ("video_finetuned", RESULTS_DIR / "urbansas_video_oof.csv", "y_prob", False),
    ("late_fusion_fromscratch", RESULTS_DIR / "fusion_oof.csv", "p_max", False),
    ("late_fusion_ast", RESULTS_DIR / "fusion_oof_kinetics.csv", "p_max", False),
    ("feature_fusion_mlp", RESULTS_DIR / "feature_fusion_oof_kinetics.csv", "p_fusion_mlp", True),
    (
        "feature_fusion_audio_only",
        RESULTS_DIR / "feature_fusion_oof_kinetics.csv",
        "p_audio_mlp",
        True,
    ),
    (
        "feature_fusion_video_only",
        RESULTS_DIR / "feature_fusion_oof_kinetics.csv",
        "p_video_mlp",
        True,
    ),
]

# For TAU, all models are honest (TAU never used in training).
TAU_MODELS = [
    ("ast_audio", RESULTS_DIR / "audio_pretrained_tau_external.csv", "y_prob"),
    ("video_finetuned", RESULTS_DIR / "urbansas_video_tau_external.csv", "y_prob"),
    ("late_fusion_fromscratch", RESULTS_DIR / "fusion_tau_external.csv", "p_max"),
    ("late_fusion_ast", RESULTS_DIR / "fusion_tau_external_kinetics.csv", "p_max"),
    (
        "feature_fusion_mlp",
        RESULTS_DIR / "feature_fusion_tau_external_kinetics.csv",
        "p_fusion_mlp",
    ),
    (
        "feature_fusion_audio_only",
        RESULTS_DIR / "feature_fusion_tau_external_kinetics.csv",
        "p_audio_mlp",
    ),
    (
        "feature_fusion_video_only",
        RESULTS_DIR / "feature_fusion_tau_external_kinetics.csv",
        "p_video_mlp",
    ),
]


def _validate_predictions(y: np.ndarray, p: np.ndarray) -> tuple[bool, str]:
    """Return (is_valid, reason). Checks the audit-requested conditions."""
    if len(y) == 0:
        return False, "empty dataset"
    if len(y) != len(p):
        return False, f"length mismatch: y={len(y)}, p={len(p)}"
    unique_y = set(np.unique(y).tolist())
    if unique_y - {0, 1}:
        return False, f"y_true contains values outside {{0,1}}: {unique_y}"
    if len(unique_y) < 2:
        return False, "only one class present in y_true"
    if not np.isfinite(p).all():
        return False, f"non-finite probabilities ({int((~np.isfinite(p)).sum())} values)"
    if p.min() < 0 or p.max() > 1:
        return False, f"probabilities outside [0,1]: min={p.min()}, max={p.max()}"
    if pd.isna(p).any():
        return False, "missing values in probabilities"
    return True, "ok"


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int, strategy: str
) -> tuple[float, dict]:
    """ECE with uniform or quantile binning. Reports actual bins used."""
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)  # collapse degenerate edges
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    total = len(y_prob)
    ece = 0.0
    bins_used = 0
    bin_info = []
    for bin_i in range(len(bin_edges) - 1):
        mask = bin_indices == bin_i
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue
        bins_used += 1
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        contribution = (n_in_bin / total) * abs(avg_acc - avg_conf)
        ece += contribution
        bin_info.append(
            {
                "bin_low": float(bin_edges[bin_i]),
                "bin_high": float(bin_edges[bin_i + 1]),
                "n": n_in_bin,
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "gap": avg_acc - avg_conf,
            }
        )
    return ece, {
        "strategy": strategy,
        "n_bins_requested": n_bins,
        "n_bins_used": bins_used,
        "bins": bin_info,
    }


def _plot_reliability(
    y_true: np.ndarray, y_prob: np.ndarray, title: str, out_path: Path, n_bins: int = 15
):
    """Save a reliability diagram with a sample-count histogram below."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    avg_confs, avg_accs, counts = [], [], []
    for bin_i in range(n_bins):
        mask = bin_indices == bin_i
        if mask.sum() == 0:
            continue
        avg_confs.append(y_prob[mask].mean())
        avg_accs.append(y_true[mask].mean())
        counts.append(int(mask.sum()))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_top.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax_top.plot(avg_confs, avg_accs, "o-", label="Observed")
    ax_top.set_xlabel("Mean predicted probability")
    ax_top.set_ylabel("Observed positive rate")
    ax_top.set_title(title)
    ax_top.legend()
    ax_top.grid(alpha=0.3)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax_bot.bar(bin_centers[: len(counts)], counts, width=1 / n_bins, alpha=0.6)
    ax_bot.set_xlabel("Predicted probability")
    ax_bot.set_ylabel("Sample count")
    ax_bot.set_xlim(0, 1)
    ax_bot.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def _evaluate_one(name: str, path: Path, prob_col: str, dataset_label: str) -> dict:
    """Compute calibration metrics for one model on one dataset."""
    if not path.exists():
        return {"name": name, "dataset": dataset_label, "error": f"file not found: {path}"}
    df = pd.read_csv(path)
    if prob_col not in df.columns:
        return {
            "name": name,
            "dataset": dataset_label,
            "error": f"missing column '{prob_col}' (available: {df.columns.tolist()})",
        }
    y = df["y_true"].astype(int).to_numpy()
    p = df[prob_col].astype(float).to_numpy()

    valid, reason = _validate_predictions(y, p)
    if not valid:
        return {"name": name, "dataset": dataset_label, "error": f"validation failed: {reason}"}

    ece_uniform, ece_u_info = _expected_calibration_error(y, p, n_bins=15, strategy="uniform")
    ece_adaptive, ece_a_info = _expected_calibration_error(y, p, n_bins=15, strategy="quantile")

    plot_path = PLOTS_DIR / f"{dataset_label}_{name}_reliability.png"
    _plot_reliability(y, p, f"{dataset_label}: {name}", plot_path)

    return {
        "name": name,
        "dataset": dataset_label,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "ece_uniform": ece_uniform,
        "ece_uniform_bins": ece_u_info,
        "ece_adaptive": ece_adaptive,
        "ece_adaptive_bins": ece_a_info,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
        "roc_auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "mean_prob": float(p.mean()),
        "mean_prob_neg": float(p[y == 0].mean()),
        "mean_prob_pos": float(p[y == 1].mean()),
        "mean_prob_minus_prevalence": float(p.mean() - y.mean()),
        "reliability_plot": str(plot_path.relative_to(_REPO_ROOT)),
    }


def _print_table(results: list[dict], title: str):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(
        f"{'model':<30s} {'n':>5s} {'pos':>4s} {'prev':>6s} {'ECE_u':>7s} "
        f"{'ECE_a':>7s} {'Bins_a':>7s} {'Brier':>7s} {'LogLoss':>8s} "
        f"{'AUC':>7s} {'AP':>7s}"
    )
    for r in results:
        if "error" in r:
            print(f"{r['name']:<30s}  ERROR: {r['error']}")
            continue
        print(
            f"{r['name']:<30s} {r['n']:>5d} {r['n_pos']:>4d} "
            f"{r['prevalence']:>6.3f} {r['ece_uniform']:>7.4f} "
            f"{r['ece_adaptive']:>7.4f} {r['ece_adaptive_bins']['n_bins_used']:>7d} "
            f"{r['brier']:>7.4f} {r['log_loss']:>8.4f} "
            f"{r['roc_auc']:>7.4f} {r['ap']:>7.4f}"
        )


def main():
    print(f"Outputs: {PLOTS_DIR}")

    # --- MAVD (only provenance-valid models evaluated; others recorded as excluded) ---
    mavd_included = []
    mavd_excluded = []
    for name, path, col, valid in MAVD_MODELS:
        if not valid:
            mavd_excluded.append(
                {
                    "name": name,
                    "reason": "encoder provenance contaminated (fine-tuned on all 540 MAVD clips) — MAVD OOF is in-sample",
                }
            )
            print(f"[EXCLUDED from MAVD] {name} — provenance contaminated")
            continue
        print(f"Evaluating MAVD: {name} ({path.name}, col={col})")
        mavd_included.append(_evaluate_one(name, path, col, "mavd"))

    _print_table(mavd_included, "MAVD CALIBRATION METRICS (honest models only)")
    if mavd_excluded:
        print("\nMAVD models EXCLUDED due to provenance contamination:")
        for e in mavd_excluded:
            print(f"  - {e['name']}: {e['reason']}")

    # --- TAU (all honest) ---
    tau_included = []
    for name, path, col in TAU_MODELS:
        print(f"Evaluating TAU: {name} ({path.name}, col={col})")
        tau_included.append(_evaluate_one(name, path, col, "tau"))

    _print_table(tau_included, "TAU CALIBRATION METRICS (all honest — TAU never used in training)")

    print("\n" + "=" * 92)
    print("SUMMARY")
    print("=" * 92)
    mavd_evaluated = [r for r in mavd_included if "error" not in r]
    mavd_missing = [r for r in mavd_included if "error" in r]
    tau_evaluated = [r for r in tau_included if "error" not in r]
    tau_missing = [r for r in tau_included if "error" in r]
    print(
        f"MAVD: {len(mavd_evaluated)} evaluated, {len(mavd_excluded)} excluded (provenance), {len(mavd_missing)} missing file"
    )
    print(f"TAU:  {len(tau_evaluated)} evaluated, {len(tau_missing)} missing file")
    if mavd_missing:
        for r in mavd_missing:
            print(f"  MAVD missing: {r['name']} — {r['error']}")
    if tau_missing:
        for r in tau_missing:
            print(f"  TAU missing: {r['name']} — {r['error']}")

    # --- Save ---
    out = {
        "mavd_evaluated": mavd_included,
        "mavd_excluded": mavd_excluded,
        "tau_evaluated": tau_included,
        "notes": {
            "mavd_excluded_reason": "Encoder-provenance contamination — fine-tuned video "
            "encoder trained on all 540 MAVD clips, so urbansas_video_oof.csv and both "
            "late-fusion files (which use video predictions as input) are in-sample for MAVD. "
            "See 9c-ext-2 audit for verification.",
            "calibration_vs_discrimination": "ROC-AUC and AP measure ranking; Brier and "
            "log loss measure probability quality; ECE measures binned calibration error. "
            "Temperature scaling preserves binary ranking and therefore preserves ROC-AUC/AP.",
        },
    }
    out_path = RESULTS_DIR / "calibration_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved metrics: {out_path}")
    print(f"Saved plots: {PLOTS_DIR}")
    print(f"  ({len(mavd_included) + len(tau_included)} reliability diagrams)")
    print(f"  ({len(mavd_excluded)} MAVD models excluded due to provenance)")


if __name__ == "__main__":
    main()
