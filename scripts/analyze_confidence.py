"""Confidence-usefulness analysis for Urbansas models (W10.3).

The question: when a model is confident about its prediction (whether
positive or negative), is it actually more accurate than when uncertain?

This is DIFFERENT from calibration.
    Calibration:          Do predicted probabilities match observed frequencies?
    Confidence usefulness: Does predicted confidence correlate with correctness?
    Discrimination:       Can the model rank positives above negatives?

A model can be calibrated but uselessly confident (mean predictor at
prevalence). A model can be miscalibrated but usefully confident (ranking
right, probabilities wrong).

Methodology (per the audit's corrections):
    - confidence = max(p, 1-p)  [confidence in the PREDICTED class]
    - predicted_class = (p >= 0.5)
    - correct = (predicted_class == y_true)
    - Bin by CONFIDENCE, not positive probability

  This distinction matters most for video_finetuned TAU, where predictions
  cluster near zero. Those are HIGH-CONFIDENCE NEGATIVE predictions,
  not low-confidence predictions.

Both raw and calibrated predictions analyzed:
    - RAW: original prediction CSV
    - CALIBRATED (where available): per-fold MAVD temperature applied to
      held-out fold; MAVD-fitted temperature applied to TAU (no TAU fitting)
    - Raw-only for video_finetuned TAU (no valid calibrator)

Per-fold temperature mapping (MAVD):
    Reconstructed via filename -> location -> fold in W10.2 artifacts.
    Each fold's held-out location has a corresponding temperature.
    Location extraction validated against the 5 known MAVD locations.

Deliverables per (model, dataset, raw|calibrated):
    - Mean confidence
    - Per-confidence-bin accuracy, n, n_actual_positive, n_predicted_positive
    - Confidence monotonicity
    - High-vs-low confidence accuracy gap:
        low-confidence  = confidence < 0.70
        high-confidence = confidence >= 0.90
    - Bootstrap 95% CI on the gap (clip-level; location-level for MAVD marked
      exploratory because 5 clusters)
    - Risk-coverage curve (unique confidence thresholds; ties handled)

Read-only. No training, no temperature fitting on held-out data, no threshold
tuning. Filename/label validated against expected counts (540 rows, 138 positives
for MAVD; 540 rows, 81 positives for TAU). Required columns, missing/duplicate
filenames, probability range, and label values checked. NOT validated against
a stored canonical manifest (no manifest file exists); alignment between
files relies on the OOF-pair alignment verified in W10.1.

Outputs:
    data/urbansas_results/confidence_analysis.json
    data/urbansas_results/calibration_plots/w10_3/{model}_{dataset}_{raw|cal}_confidence.png
    data/urbansas_results/calibration_plots/w10_3/{model}_{dataset}_{raw|cal}_risk_coverage.png

Runtime: ~30-60 sec.

Runs:
    uv run python scripts/analyze_confidence.py
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

RESULTS_DIR = _REPO_ROOT / "data" / "urbansas_results"
PLOTS_DIR = RESULTS_DIR / "calibration_plots" / "w10_3"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

KNOWN_MAVD_LOCATIONS = [
    "acevedo0103",
    "magallanes0308",
    "rivera0922",
    "rivera0923",
    "templete0103",
]

CONFIDENCE_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
LOW_CONF_MAX = 0.70
HIGH_CONF_MIN = 0.90

MAVD_EXPECTED_N = 540
MAVD_EXPECTED_POS = 138
TAU_EXPECTED_N = 540
TAU_EXPECTED_POS = 81

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


# --- Correction 1: validated location extraction ---


def _extract_location(filename: str) -> str:
    """Match filename against known MAVD locations; raise if not exactly one."""
    matches = [loc for loc in KNOWN_MAVD_LOCATIONS if filename.startswith(loc + "_")]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely identify location for filename: {filename}")
    return matches[0]


def _extract_video_id(filename: str) -> str:
    """MAVD filename is {location}_{run:02d}_{clip:02d}.

    The source video ID is {location}_{run:02d} — clips from the same
    video share it. Used for video-level cluster bootstrap.
    """
    parts = filename.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}"
    return filename  # fallback for unexpected filename format


# --- Correction 2: alignment validation ---


def _load_and_validate(
    path: Path,
    prob_col: str,
    expected_n: int,
    expected_pos: int,
    need_locations: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load CSV, validate schema/alignment, return (y, p, filenames, locations)."""
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"filename", "y_true", prob_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    if df["filename"].isna().any():
        raise ValueError(f"{path.name}: null filenames present")
    if df["filename"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate filenames present")
    if len(df) != expected_n:
        raise ValueError(f"{path.name}: expected {expected_n} rows, got {len(df)}")
    if int(df["y_true"].sum()) != expected_pos:
        raise ValueError(
            f"{path.name}: expected {expected_pos} positives, got {int(df['y_true'].sum())}"
        )

    y = df["y_true"].astype(int).to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(
            f"{path.name}: y_true contains values outside {{0,1}}: "
            f"{set(np.unique(y).tolist()) - {0, 1}}"
        )
    p = df[prob_col].astype(float).to_numpy()
    fns = df["filename"].tolist()

    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError(f"{path.name}: invalid probability values")

    locations = None
    if need_locations:
        locations = np.array([_extract_location(fn) for fn in fns])
        # Ensure every known location appears
        present = set(locations.tolist())
        if present != set(KNOWN_MAVD_LOCATIONS):
            raise ValueError(
                f"{path.name}: expected all MAVD locations, missing "
                f"{set(KNOWN_MAVD_LOCATIONS) - present}"
            )

    return y, p, fns, locations


# --- Confidence binning (audit correction #3) ---


def _bin_accuracy_by_confidence(y: np.ndarray, p: np.ndarray) -> dict:
    pred_class = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1.0 - p)
    correct = (pred_class == y).astype(int)

    bins = []
    for i in range(len(CONFIDENCE_BINS) - 1):
        lo = CONFIDENCE_BINS[i]
        hi = CONFIDENCE_BINS[i + 1]
        if i < len(CONFIDENCE_BINS) - 2:
            mask = (confidence >= lo) & (confidence < hi)
        else:
            mask = (confidence >= lo) & (confidence <= hi)
        n = int(mask.sum())
        if n == 0:
            bins.append(
                {
                    "bin_low": lo,
                    "bin_high": hi,
                    "n": 0,
                    "accuracy": None,
                    "n_correct": 0,
                    "n_actual_positive": 0,
                    "n_predicted_positive": 0,
                }
            )
            continue
        bins.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "n": n,
                "accuracy": float(correct[mask].mean()),
                "n_correct": int(correct[mask].sum()),
                "n_actual_positive": int(y[mask].sum()),
                "n_predicted_positive": int(pred_class[mask].sum()),
            }
        )

    return {
        "bins": bins,
        "mean_confidence": float(confidence.mean()),
        "overall_accuracy": float(correct.mean()),
    }


def _high_low_gap(y: np.ndarray, p: np.ndarray) -> dict:
    pred_class = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1.0 - p)
    correct = (pred_class == y).astype(int)

    low_mask = confidence < LOW_CONF_MAX
    high_mask = confidence >= HIGH_CONF_MIN

    n_low = int(low_mask.sum())
    n_high = int(high_mask.sum())

    acc_low = float(correct[low_mask].mean()) if n_low > 0 else None
    acc_high = float(correct[high_mask].mean()) if n_high > 0 else None

    if acc_low is not None and acc_high is not None:
        gap = float(acc_high - acc_low)
    else:
        gap = None

    # Sufficiency flags (20 is a reasonable minimum for stable estimate)
    MIN_GROUP = 20
    n_low_sufficient = n_low >= MIN_GROUP
    n_high_sufficient = n_high >= MIN_GROUP
    gap_reliable = (gap is not None) and n_low_sufficient and n_high_sufficient

    if gap is None:
        gap_caveat = f"undefined: high-conf group has {n_high} clips, low-conf {n_low}"
    elif not gap_reliable:
        gap_caveat = f"exploratory: n_high={n_high}, n_low={n_low} (min {MIN_GROUP} recommended)"
    else:
        gap_caveat = None

    return {
        "low_confidence_threshold": LOW_CONF_MAX,
        "high_confidence_threshold": HIGH_CONF_MIN,
        "n_low": n_low,
        "n_high": n_high,
        "n_low_sufficient": n_low_sufficient,
        "n_high_sufficient": n_high_sufficient,
        "accuracy_low": acc_low,
        "accuracy_high": acc_high,
        "gap": gap,
        "gap_reliable": gap_reliable,
        "gap_caveat": gap_caveat,
    }


# --- Bootstrap (audit correction #6) ---


def _bootstrap_gap_ci(y, p, locations, video_ids=None, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    pred_class = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1.0 - p)
    correct = (pred_class == y).astype(int)

    def gap_from_idx(idx):
        low = idx[confidence[idx] < LOW_CONF_MAX]
        high = idx[confidence[idx] >= HIGH_CONF_MIN]
        if len(low) == 0 or len(high) == 0:
            return None
        return float(correct[high].mean() - correct[low].mean())

    results = {}
    n = len(y)
    gaps = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        g = gap_from_idx(idx)
        if g is not None:
            gaps.append(g)

    # If fewer than 90% of bootstraps produce a gap, the CI is unreliable
    if len(gaps) < 0.9 * n_boot:
        results["clip_level"] = {
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "n_valid_bootstraps": int(len(gaps)),
            "n_boot_requested": int(n_boot),
            "reason": "insufficient valid bootstraps (empty group in too many resamples)",
        }
    else:
        gaps_arr = np.array(gaps)
        results["clip_level"] = {
            "mean": float(gaps_arr.mean()),
            "ci_low": float(np.percentile(gaps_arr, 2.5)),
            "ci_high": float(np.percentile(gaps_arr, 97.5)),
            "n_valid_bootstraps": int(len(gaps)),
            "n_boot_requested": int(n_boot),
            "reason": None,
        }

    # Location-level (if provided)
    if locations is not None:
        uniq = np.unique(locations)
        loc_idx = {loc: np.where(locations == loc)[0] for loc in uniq}
        gaps_loc = []
        for _ in range(n_boot):
            chosen = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([loc_idx[loc] for loc in chosen])
            g = gap_from_idx(idx)
            if g is not None:
                gaps_loc.append(g)
        if len(gaps_loc) < 0.9 * n_boot:
            results["location_level"] = {
                "mean": None,
                "ci_low": None,
                "ci_high": None,
                "n_valid_bootstraps": int(len(gaps_loc)),
                "n_boot_requested": int(n_boot),
                "label": "location-level (EXPLORATORY — only 5 clusters)",
                "reason": "insufficient valid bootstraps",
            }
        else:
            gaps_loc_arr = np.array(gaps_loc)
            results["location_level"] = {
                "mean": float(gaps_loc_arr.mean()),
                "ci_low": float(np.percentile(gaps_loc_arr, 2.5)),
                "ci_high": float(np.percentile(gaps_loc_arr, 97.5)),
                "n_valid_bootstraps": int(len(gaps_loc)),
                "n_boot_requested": int(n_boot),
                "label": "location-level (EXPLORATORY — only 5 clusters)",
                "reason": None,
            }

    # Video-level (if provided) — clusters of clips from same source video
    if video_ids is not None:
        uniq_vids = np.unique(video_ids)
        vid_idx = {v: np.where(video_ids == v)[0] for v in uniq_vids}
        gaps_vid = []
        for _ in range(n_boot):
            chosen = rng.choice(uniq_vids, size=len(uniq_vids), replace=True)
            idx = np.concatenate([vid_idx[v] for v in chosen])
            g = gap_from_idx(idx)
            if g is not None:
                gaps_vid.append(g)
        if len(gaps_vid) < 0.9 * n_boot:
            results["video_level"] = {
                "mean": None,
                "ci_low": None,
                "ci_high": None,
                "n_valid_bootstraps": int(len(gaps_vid)),
                "n_boot_requested": int(n_boot),
                "label": "video-level (clusters of clips from same source video; assumes independence between videos)",
                "reason": "insufficient valid bootstraps",
            }
        else:
            gaps_vid_arr = np.array(gaps_vid)
            results["video_level"] = {
                "mean": float(gaps_vid_arr.mean()),
                "ci_low": float(np.percentile(gaps_vid_arr, 2.5)),
                "ci_high": float(np.percentile(gaps_vid_arr, 97.5)),
                "n_valid_bootstraps": int(len(gaps_vid)),
                "n_boot_requested": int(n_boot),
                "label": "video-level (clusters of clips from same source video; assumes independence between videos)",
                "reason": None,
            }

    return results


# --- Risk-coverage with unique thresholds (audit correction #5) ---


def _risk_coverage(y: np.ndarray, p: np.ndarray, n_points: int = 20) -> dict:
    """Sort by confidence descending; use UNIQUE confidence thresholds for coverage.

    Groups predictions with identical confidence, so intermediate coverage
    points are stable across runs.
    """
    pred_class = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1.0 - p)
    correct = (pred_class == y).astype(int)

    unique_confs = np.unique(confidence)[::-1]  # descending
    if len(unique_confs) > n_points:
        idx_samples = np.linspace(0, len(unique_confs) - 1, n_points).astype(int)
        threshold_sample = unique_confs[idx_samples]
    else:
        threshold_sample = unique_confs

    coverages, accuracies = [], []
    for thr in threshold_sample:
        mask = confidence >= thr
        n_cover = int(mask.sum())
        if n_cover == 0:
            continue
        coverages.append(n_cover / len(y))
        accuracies.append(float(correct[mask].mean()))

    return {
        "coverage": coverages,
        "accuracy": accuracies,
        "overall_accuracy": float(correct.mean()),
    }


def _compare_risk_coverage(rc_raw: dict, rc_cal: dict) -> dict:
    """Check risk-coverage equivalence. Compares both coverage grid and accuracy.

    For single-global-temperature calibration (e.g., TAU), curves should
    match exactly (monotonic transformation preserves ranking). For per-fold-
    temperature calibration (e.g., MAVD), curves may differ because different
    temperatures applied to different folds can change cross-fold ordering.

    Both coverage grid AND accuracy are compared.
    """
    covs_raw = np.array(rc_raw["coverage"])
    covs_cal = np.array(rc_cal["coverage"])
    accs_raw = np.array(rc_raw["accuracy"])
    accs_cal = np.array(rc_cal["accuracy"])

    if len(covs_raw) != len(covs_cal) or len(accs_raw) != len(accs_cal):
        return {
            "max_abs_coverage_diff": None,
            "max_abs_accuracy_diff": None,
            "curves_equivalent": None,
            "note": "coverage grid length differs (unexpected)",
        }

    max_cov_diff = float(np.max(np.abs(covs_raw - covs_cal)))
    max_acc_diff = float(np.max(np.abs(accs_raw - accs_cal)))

    return {
        "max_coverage_difference": max_cov_diff,
        "max_accuracy_difference": max_acc_diff,
        "curves_equivalent": bool(max_acc_diff < 1e-6),
        "note": (
            "Diagnostic, not an invariance test. For single-global-temperature "
            "calibration (TAU), ranking is preserved → curves should match exactly. "
            "For per-fold-temperature calibration (MAVD), different temperatures "
            "per fold can change cross-fold ranking → accuracy may differ. This "
            "is a property of the per-fold protocol, not a bug."
        ),
    }


# --- Plots ---


def _plot_confidence(bins: list, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    valid_bins = [b for b in bins if b["n"] > 0]
    centers = [(b["bin_low"] + b["bin_high"]) / 2 for b in valid_bins]
    accs = [b["accuracy"] for b in valid_bins]
    ns = [b["n"] for b in valid_bins]
    ax.plot(centers, accs, "o-", label="Accuracy per confidence bin")
    for c, a, n in zip(centers, accs, ns):
        ax.annotate(f"n={n}", (c, a), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xlabel("Confidence in predicted class")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def _plot_risk_coverage(rc: dict, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rc["coverage"], rc["accuracy"], "o-", label="Accuracy at coverage")
    ax.axhline(
        rc["overall_accuracy"],
        color="k",
        linestyle="--",
        label=f"Overall accuracy ({rc['overall_accuracy']:.3f})",
    )
    ax.set_xlabel("Coverage (fraction of clips accepted by confidence threshold)")
    ax.set_ylabel("Accuracy on accepted clips")
    ax.set_title(title)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


# --- Analysis driver ---


def _analyze_model(name, y, p, locations, dataset, variant, fns, video_ids=None):
    """Compute all confidence-usefulness metrics for one (model, dataset, variant)."""
    bins = _bin_accuracy_by_confidence(y, p)
    gap = _high_low_gap(y, p)
    ci = _bootstrap_gap_ci(y, p, locations, video_ids=video_ids)
    rc = _risk_coverage(y, p)

    _plot_confidence(
        bins["bins"],
        f"{dataset.upper()} / {name} / {variant}",
        PLOTS_DIR / f"{dataset}_{name}_{variant}_confidence.png",
    )
    _plot_risk_coverage(
        rc,
        f"{dataset.upper()} / {name} / {variant} (Accuracy vs Coverage)",
        PLOTS_DIR / f"{dataset}_{name}_{variant}_risk_coverage.png",
    )

    return {
        "model": name,
        "dataset": dataset,
        "variant": variant,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "mean_confidence": bins["mean_confidence"],
        "overall_accuracy": bins["overall_accuracy"],
        "per_confidence_bin": bins["bins"],
        "high_vs_low_gap": gap,
        "gap_bootstrap_ci": ci,
        "risk_coverage": rc,
        "risk_coverage_raw_vs_calibrated": None,
    }


# --- Calibration application helpers ---


def _load_w10_2_results() -> dict:
    path = RESULTS_DIR / "calibration_temperature_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _build_fold_t_map(temp_results: dict, model_name: str) -> dict:
    """Return {held_location: temperature} from W10.2 results."""
    for r in temp_results.get("mavd", []):
        if r.get("model") != model_name:
            continue
        return {f["held_location"]: f["temperature"] for f in r.get("per_fold", [])}
    return {}


def _build_tau_t(temp_results: dict, model_name: str) -> float | None:
    """Return the single TAU transfer temperature from W10.2 results."""
    for r in temp_results.get("tau", []):
        if r.get("model") == model_name:
            return r.get("temperature_fit_on_mavd")
    return None


def _apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-7
    p_clip = np.clip(p, eps, 1 - eps)
    logit = np.log(p_clip / (1 - p_clip))
    return 1.0 / (1.0 + np.exp(-logit / T))


# --- Main ---


def main():
    print(f"Outputs: {PLOTS_DIR}")

    w10_2 = _load_w10_2_results()
    if not w10_2:
        print(
            "WARNING: calibration_temperature_results.json missing — calibrated "
            "variant will be skipped"
        )

    mavd_results, tau_results = [], []

    print("\n=== MAVD (honest models only) ===")
    for name, path, col, valid in MAVD_MODELS:
        if not valid:
            print(f"  [EXCLUDED] {name} — provenance contaminated")
            continue
        print(f"\n  --- {name} ---")
        try:
            y, p, fns, locations = _load_and_validate(
                path,
                col,
                MAVD_EXPECTED_N,
                MAVD_EXPECTED_POS,
                need_locations=True,
            )
        except Exception as e:
            print(f"    VALIDATION FAILED: {e}")
            continue

        # Raw
        # Extract video IDs for MAVD (from filenames)
        video_ids = np.array([_extract_video_id(fn) for fn in fns])
        raw = _analyze_model(name, y, p, locations, "mavd", "raw", fns, video_ids=video_ids)
        mavd_results.append(raw)
        g = raw["high_vs_low_gap"]
        ci_clip = raw["gap_bootstrap_ci"].get("clip_level", {})
        gap_str = f"{g['gap']:+.3f}" if g["gap"] is not None else "n/a"
        ci_clip = raw["gap_bootstrap_ci"].get("clip_level", {})
        ci_vid = raw["gap_bootstrap_ci"].get("video_level", {})
        ci_str = (
            f"[{ci_clip['ci_low']:+.3f}, {ci_clip['ci_high']:+.3f}]"
            if ci_clip.get("ci_low") is not None and ci_clip.get("ci_high") is not None
            else "n/a"
        )
        ci_vid_str = (
            f"[{ci_vid['ci_low']:+.3f}, {ci_vid['ci_high']:+.3f}]"
            if ci_vid.get("ci_low") is not None and ci_vid.get("ci_high") is not None
            else "n/a"
        )
        print(
            f"    raw:  mean_conf={raw['mean_confidence']:.3f}  "
            f"gap={gap_str}  "
            f"CI_clip={ci_str}  "
            f"CI_video={ci_vid_str}  "
            f"(n_high={g['n_high']}, n_low={g['n_low']}, "
            f"reliable={g.get('gap_reliable')})"
        )

        # Calibrated (if per-fold T available)
        fold_t = _build_fold_t_map(w10_2, name)
        if fold_t:
            p_cal = np.zeros_like(p)
            for loc in KNOWN_MAVD_LOCATIONS:
                if loc not in fold_t:
                    print(f"    WARNING: no temperature for {loc}, using raw")
                    p_cal[locations == loc] = p[locations == loc]
                    continue
                p_cal[locations == loc] = _apply_temperature(p[locations == loc], fold_t[loc])
            cal = _analyze_model(
                name, y, p_cal, locations, "mavd", "calibrated", fns, video_ids=video_ids
            )

            # Risk-coverage invariance check (temperature scaling is monotonic)
            rc_comparison = _compare_risk_coverage(raw["risk_coverage"], cal["risk_coverage"])
            raw["risk_coverage_raw_vs_calibrated"] = rc_comparison
            cal["risk_coverage_raw_vs_calibrated"] = rc_comparison

            mavd_results.append(cal)
            g_cal = cal["high_vs_low_gap"]
            ci_cal = cal["gap_bootstrap_ci"].get("clip_level", {})
            gap_str_cal = f"{g_cal['gap']:+.3f}" if g_cal["gap"] is not None else "n/a"
            ci_cal = cal["gap_bootstrap_ci"].get("clip_level", {})
            ci_vid_cal = cal["gap_bootstrap_ci"].get("video_level", {})
            ci_str = (
                f"[{ci_cal['ci_low']:+.3f}, {ci_cal['ci_high']:+.3f}]"
                if ci_cal.get("ci_low") is not None
                else "n/a"
            )
            ci_vid_str_cal = (
                f"[{ci_vid_cal['ci_low']:+.3f}, {ci_vid_cal['ci_high']:+.3f}]"
                if ci_vid_cal.get("ci_low") is not None
                else "n/a"
            )
            print(
                f"    cal:  mean_conf={cal['mean_confidence']:.3f}  "
                f"gap={gap_str_cal}  "
                f"CI_clip={ci_str}  "
                f"CI_video={ci_vid_str_cal}  "
                f"(n_high={g_cal['n_high']}, n_low={g_cal['n_low']}, "
                f"reliable={g_cal.get('gap_reliable')})"
            )
            if rc_comparison:
                print(
                    f"    risk-coverage diagnostic (fold-specific calibration ranking): "
                    f"max_cov_diff={rc_comparison.get('max_coverage_difference')}, "
                    f"max_acc_diff={rc_comparison.get('max_accuracy_difference')}, "
                    f"equivalent={rc_comparison.get('curves_equivalent')}"
                )
                if not rc_comparison.get("curves_equivalent"):
                    print(
                        f"      (MAVD uses per-fold temperatures — cross-fold "
                        f"ranking can change; not a bug)"
                    )
        else:
            print(f"    calibrated: skipped (no W10.2 per-fold T for this model)")

    print("\n=== TAU (all honest) ===")
    for name, path, col in TAU_MODELS:
        print(f"\n  --- {name} ---")
        try:
            y, p, fns, _ = _load_and_validate(
                path,
                col,
                TAU_EXPECTED_N,
                TAU_EXPECTED_POS,
                need_locations=False,
            )
        except Exception as e:
            print(f"    VALIDATION FAILED: {e}")
            continue

        raw = _analyze_model(name, y, p, None, "tau", "raw", fns)
        tau_results.append(raw)
        g = raw["high_vs_low_gap"]
        ci_clip = raw["gap_bootstrap_ci"].get("clip_level", {})
        gap_str = f"{g['gap']:+.3f}" if g["gap"] is not None else "n/a"
        ci_str = (
            f"[{ci_clip['ci_low']:+.3f}, {ci_clip['ci_high']:+.3f}]"
            if ci_clip.get("ci_low") is not None and ci_clip.get("ci_high") is not None
            else "n/a"
        )
        print(
            f"    raw:  mean_conf={raw['mean_confidence']:.3f}  "
            f"gap={gap_str}  "
            f"CI_clip={ci_str}  "
            f"(n_high={g['n_high']}, n_low={g['n_low']}, "
            f"reliable={g.get('gap_reliable')})"
        )

        tau_t = _build_tau_t(w10_2, name)
        if tau_t is not None:
            p_cal = _apply_temperature(p, tau_t)
            cal = _analyze_model(name, y, p_cal, None, "tau", "calibrated", fns)

            # Risk-coverage invariance check (temperature scaling is monotonic)
            rc_comparison = _compare_risk_coverage(raw["risk_coverage"], cal["risk_coverage"])
            raw["risk_coverage_raw_vs_calibrated"] = rc_comparison
            cal["risk_coverage_raw_vs_calibrated"] = rc_comparison

            tau_results.append(cal)
            g_cal = cal["high_vs_low_gap"]
            ci_cal = cal["gap_bootstrap_ci"].get("clip_level", {})
            gap_str_cal = f"{g_cal['gap']:+.3f}" if g_cal["gap"] is not None else "n/a"
            ci_str = (
                f"[{ci_cal['ci_low']:+.3f}, {ci_cal['ci_high']:+.3f}]"
                if ci_cal.get("ci_low") is not None
                else "n/a"
            )
            print(
                f"    cal:  mean_conf={cal['mean_confidence']:.3f}  "
                f"gap={gap_str_cal}  "
                f"CI_clip={ci_str}  "
                f"(T={tau_t:.3f}, n_high={g_cal['n_high']}, n_low={g_cal['n_low']}, "
                f"reliable={g_cal.get('gap_reliable')})"
            )
            if rc_comparison:
                print(
                    f"    risk-coverage diagnostic (single-temperature ranking): "
                    f"max_cov_diff={rc_comparison.get('max_coverage_difference')}, "
                    f"max_acc_diff={rc_comparison.get('max_accuracy_difference')}, "
                    f"equivalent={rc_comparison.get('curves_equivalent')}"
                )
        else:
            print(f"    calibrated: skipped (no W10.2 TAU transfer T for this model)")

    out = {
        "mavd": mavd_results,
        "tau": tau_results,
        "mavd_excluded": [
            {
                "model": "video_finetuned",
                "reason": "encoder provenance contaminated (fine-tuned on all 540 MAVD clips)",
            },
            {
                "model": "late_fusion_fromscratch",
                "reason": "uses contaminated video predictions as input",
            },
            {
                "model": "late_fusion_ast",
                "reason": "uses contaminated video predictions; TAU file also missing",
            },
        ],
        "tau_excluded": [
            {
                "model": "late_fusion_ast",
                "reason": "fusion_tau_external_kinetics.csv missing (documented evidence gap from W10.1)",
            },
        ],
        "notes": {
            "concept_distinction": "Calibration, discrimination, and confidence "
            "usefulness are three distinct properties.",
            "confidence_definition": "confidence = max(p, 1-p) — confidence in "
            "the PREDICTED class, not the positive class.",
            "high_low_definitions": {
                "low": f"confidence < {LOW_CONF_MAX}",
                "high": f"confidence >= {HIGH_CONF_MIN}",
            },
            "gap_definition": "gap = accuracy_high − accuracy_low",
            "gap_reliability": "gap_reliable=False when n_high<20 or n_low<20 "
            "(sample inadequate for stable estimate). Reliable does NOT mean "
            "statistically significant — check the CI.",
            "ci_validity": "CI is None when <90% of bootstrap resamples produce "
            "a gap (empty high/low group in too many resamples).",
            "calibrated_variant": "MAVD: per-fold temperature applied to held-out "
            "fold. TAU: MAVD-fitted temperature applied to TAU (no TAU fitting). "
            "Fold-to-temperature mapping reconstructed via filename -> location "
            "in W10.2 artifacts.",
            "location_bootstrap": "EXPLORATORY — only 5 MAVD clusters.",
            "mavd_risk_coverage_caveat": "MAVD applies a different temperature "
            "per fold. Per-fold temperatures are not a single global monotonic "
            "transform — cross-fold ranking can change. The MAVD risk-coverage "
            "difference is a property of the per-fold protocol, not a bug. "
            "For TAU (single global T), curves are exactly equivalent, "
            "confirming monotonicity.",
            "late_fusion_ast_excluded": "fusion_tau_external_kinetics.csv missing "
            "(documented evidence gap from W10.1). Not regenerated.",
        },
    }
    out_path = RESULTS_DIR / "confidence_analysis.json"
    with open(out_path, "w") as f:
        json.dump(
            out, f, indent=2, default=lambda o: None if isinstance(o, float) and o != o else str(o)
        )
    print(f"\nSaved: {out_path}")
    print(f"Plots: {PLOTS_DIR}")
    total_plots = len([1 for _ in PLOTS_DIR.glob("*.png")])
    print(f"  ({total_plots} plots)")


if __name__ == "__main__":
    main()
