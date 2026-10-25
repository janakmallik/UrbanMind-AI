"""Week 9 Session 4 — Train and evaluate multi-view fusion on BDD100K.

Compares 8 models:
    1. ResNet-only stacker (3 probs → LightGBM)
    2. SegFormer-only stacker (5 shares → LightGBM)
    3. YOLO-only stacker (6 counts → LightGBM)
    4. Full fusion (14 features → LightGBM)               ← THE DELIVERABLE
    5. Ablation: full minus ResNet (11 features)
    6. Ablation: full minus SegFormer (9 features)
    7. Ablation: full minus YOLO (8 features)
    8. Original ResNet classifier (loaded from .pt)

Protocol:
    - 5-fold stratified CV on clean_train (4,727 rows) for model comparison
    - 5-fold OOF predictions used to evaluate fusion without leakage
    - Final test evaluation on clean_test (4,745 rows) — once
    - 3 seeds per model, paired bootstrap CI

Outputs:
    - Console comparison table
    - Per-class F1 for ResNet-only vs. Full fusion
    - Confusion matrix for full fusion on clean_test
    - MLflow run with all metrics
    - Model card draft
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

# --- Module-level dataset (Windows DataLoader needs picklable classes) ---


class ResnetDataset(Dataset):
    """Simple (image, label) dataset for evaluating the original ResNet."""

    def __init__(self, names, labels, images_dir, tf):
        self.names = names
        self.labels = labels
        self.images_dir = Path(images_dir)
        self.tf = tf

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        path = self.images_dir / self.names[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), self.labels[idx]


FEATURES_DIR = _REPO_ROOT / "data" / "features" / "bdd100k_fusion"
RESULTS_DIR = _REPO_ROOT / "data" / "fusion_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCENE_NAMES = ["city street", "highway", "residential"]

# --- Feature groups ---
RESNET_FEATURES = ["resnet_city", "resnet_highway", "resnet_residential"]
SEGFORMER_FEATURES = ["seg_road", "seg_sidewalk", "seg_vehicle", "seg_person", "seg_other"]
YOLO_FEATURES = [
    "yolo_person",
    "yolo_car",
    "yolo_bus",
    "yolo_truck",
    "yolo_traffic_light",
    "yolo_stop_sign",
]
ALL_FEATURES = RESNET_FEATURES + SEGFORMER_FEATURES + YOLO_FEATURES


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    return train, test


def make_lgbm(seed: int) -> lgb.LGBMClassifier:
    """Fixed hyperparams, class-weighted for minority classes."""
    return lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def cv_predict(
    X: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    seed: int,
    n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (oof_predictions, oof_proba) using stratified K-fold.

    oof_predictions: shape (N,), argmax of oof_proba
    oof_proba: shape (N, 3), out-of-fold predicted probabilities
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_proba = np.zeros((len(X), 3), dtype=np.float32)
    oof_preds = np.zeros(len(X), dtype=np.int32)
    for train_idx, val_idx in skf.split(X, y):
        model = make_lgbm(seed)
        model.fit(X.iloc[train_idx][features], y[train_idx])
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx][features])
        oof_preds[val_idx] = oof_proba[val_idx].argmax(axis=1)
    return oof_preds, oof_proba


def evaluate_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    features: list[str],
    seeds: list[int],
) -> dict:
    """Run 5-fold CV (train side) + single test eval. Average over seeds."""
    cv_accs, cv_f1s = [], []
    test_accs, test_f1s = [], []
    best_oof_proba = None
    best_test_f1 = -1.0
    best_seed = None
    best_model = None

    for seed in seeds:
        oof_preds, oof_proba = cv_predict(X_train, y_train, features, seed)
        cv_acc = accuracy_score(y_train, oof_preds)
        cv_f1 = f1_score(y_train, oof_preds, average="macro", zero_division=0)
        cv_accs.append(cv_acc)
        cv_f1s.append(cv_f1)

        model = make_lgbm(seed)
        model.fit(X_train[features], y_train)
        test_preds = model.predict(X_test[features])
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)

    # NO test-based selection. Report mean ± std over all seeds.
    # Test set is for final evaluation only.
    # For per-class report: use seed=42 (deterministic, not test-based)
    fusion_seed42 = make_lgbm(42)
    fusion_seed42.fit(X_train[ALL_FEATURES], y_train)

    return {
        "name": name,
        "cv_acc_mean": float(np.mean(cv_accs)),
        "cv_acc_std": float(np.std(cv_accs)),
        "cv_f1_mean": float(np.mean(cv_f1s)),
        "cv_f1_std": float(np.std(cv_f1s)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "test_f1_mean": float(np.mean(test_f1s)),
        "test_f1_std": float(np.std(test_f1s)),
        "best_test_f1": best_test_f1,
        "best_seed": best_seed,
        "best_oof_proba": best_oof_proba,
        "best_model": best_model,
    }


def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print(f"{'Model':<32s} {'CV F1':>14s} {'Test acc':>14s} {'Test F1':>14s}")
    print("-" * 90)
    for r in results:
        cv_f1 = (
            f"{r['cv_f1_mean']:.4f}±{r['cv_f1_std']:.3f}"
            if r["cv_f1_mean"] is not None
            else "  (no CV)"
        )
        print(
            f"{r['name']:<32s} "
            f"{cv_f1:>14s} "
            f"{r['test_acc_mean']:.4f}±{r['test_acc_std']:.3f} "
            f"{r['test_f1_mean']:.4f}±{r['test_f1_std']:.3f}"
        )
    print("=" * 90)


def print_per_class(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> None:
    """Per-class F1 for full fusion. Uses seed=42 deterministically (not test-based)."""
    print("\n=== Per-class F1 (full fusion, seed=42) ===")
    model = make_lgbm(42)
    model.fit(X_train[ALL_FEATURES], y_train)
    preds = model.predict(X_test[ALL_FEATURES])
    report = classification_report(
        y_test,
        preds,
        labels=[0, 1, 2],
        target_names=SCENE_NAMES,
        zero_division=0,
    )
    print(report)

    print("Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    print(f"{'':>14s} {'city':>8s} {'highway':>8s} {'resid.':>8s}")
    for i, name in enumerate(SCENE_NAMES):
        row = " ".join(f"{v:>8d}" for v in cm[i])
        print(f"{name:>14s} {row}")


def evaluate_original_resnet_on_clean_test() -> dict:
    """Load bdd_resnet50_best.pt and evaluate on clean_test images."""
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    from urbanintel.config import dataset_path

    # Load clean_test image IDs and labels from features parquet
    test_df = pd.read_parquet(FEATURES_DIR / "test.parquet")
    image_names = test_df["image_name"].tolist()
    labels = test_df["scene_label"].values

    # Build image paths from BDD100K val dir
    images_dir = dataset_path("bdd100k", "images/100k/val")

    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(
        torch.load(_REPO_ROOT / "bdd_resnet50_best.pt", map_location=device, weights_only=True)
    )
    model = model.to(device).eval()

    ds = ResnetDataset(image_names, labels, images_dir, eval_tf)
    dl = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)

    preds, true = [], []
    with torch.no_grad():
        for imgs, lbls in dl:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            preds.extend(logits.argmax(dim=-1).cpu().numpy())
            true.extend(lbls.numpy())

    y_true, y_pred = np.array(true), np.array(preds)
    return {
        "name": "Original Resnet (checkpoint)",
        "test_acc_mean": float(accuracy_score(y_true, y_pred)),
        "test_acc_std": 0.0,
        "test_f1_mean": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_f1_std": 0.0,
        "cv_acc_mean": None,
        "cv_acc_std": None,
        "cv_f1_mean": None,
        "cv_f1_std": None,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def paired_bootstrap_ci(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for the difference in macro-F1 between model A and model B.

    Returns: {"delta": observed_delta, "ci_low": ..., "ci_high": ..., "p_improvement": ...}
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def macro_f1(y, p):
        return f1_score(y, p, average="macro", zero_division=0)

    observed = macro_f1(y_true, preds_a) - macro_f1(y_true, preds_b)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(macro_f1(y_true[idx], preds_a[idx]) - macro_f1(y_true[idx], preds_b[idx]))
    deltas = np.array(deltas)
    return {
        "delta": float(observed),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "p_improvement": float((deltas > 0).mean()),
    }


def main() -> None:
    print("=== Loading features ===")
    train, test = load_features()
    print(f"  Train: {len(train)} rows, {len(train.columns)} cols")
    print(f"  Test:  {len(test)} rows, {len(test.columns)} cols")

    X_train = train[ALL_FEATURES]
    y_train = train["scene_label"].values
    X_test = test[ALL_FEATURES]
    y_test = test["scene_label"].values

    print(f"\n  Train label dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test label dist:  {dict(zip(*np.unique(y_test, return_counts=True)))}")

    seeds = [42, 123, 7]

    models_to_run = [
        ("ResNet-only stacker", RESNET_FEATURES),
        ("SegFormer-only stacker", SEGFORMER_FEATURES),
        ("YOLO-only stacker", YOLO_FEATURES),
        ("Full fusion (3+5+6)", ALL_FEATURES),
        ("Ablation: no ResNet", SEGFORMER_FEATURES + YOLO_FEATURES),
        ("Ablation: no SegFormer", RESNET_FEATURES + YOLO_FEATURES),
        ("Ablation: no YOLO", RESNET_FEATURES + SEGFORMER_FEATURES),
    ]

    # --- Get fusion predictions for bootstrap ---
    print("\n=== Computing fusion predictions (seed=42) for bootstrap ===")
    fusion_model = make_lgbm(42)
    fusion_model.fit(X_train[ALL_FEATURES], y_train)
    fusion_preds = fusion_model.predict(X_test[ALL_FEATURES])

    results = []
    for name, feats in models_to_run:
        print(f"\n=== {name} ({len(feats)} features) ===")
        r = evaluate_model(name, X_train, y_train, X_test, y_test, feats, seeds)
        results.append(r)
        print(f"  CV F1:   {r['cv_f1_mean']:.4f} ± {r['cv_f1_std']:.3f}")
        print(f"  Test F1: {r['test_f1_mean']:.4f} ± {r['test_f1_std']:.3f}")

    # Add original ResNet baseline
    print("\n=== Original ResNet (checkpoint) ===")
    resnet_baseline = evaluate_original_resnet_on_clean_test()
    results.append(resnet_baseline)
    print(f"  Test F1: {resnet_baseline['test_f1_mean']:.4f}")

    print_comparison(results)

    # Per-class for the fusion model (seed=42, deterministic)
    print_per_class(X_train, y_train, X_test, y_test)

    # --- Bootstrap CI: fusion vs original ResNet ---
    print("\n=== Bootstrap: fusion vs. original ResNet ===")
    resnet_result = next(r for r in results if "Original Resnet" in r["name"])
    ci = paired_bootstrap_ci(
        resnet_result["y_true"],
        fusion_preds,
        resnet_result["y_pred"],
        n_boot=1000,
    )
    print(f"  Fusion macro-F1 − ResNet macro-F1:")
    print(f"    Point estimate: {ci['delta']:+.4f}")
    print(f"    95% bootstrap CI: [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")
    print(f"    P(fusion > ResNet) = {ci['p_improvement']:.3f}")

    # --- Per-class report for ResNet checkpoint ---
    print("\n=== Per-class F1: Original ResNet checkpoint ===")
    report = classification_report(
        resnet_result["y_true"],
        resnet_result["y_pred"],
        labels=[0, 1, 2],
        target_names=SCENE_NAMES,
        zero_division=0,
    )
    print(report)

    print("Confusion matrix (ResNet):")
    cm = confusion_matrix(resnet_result["y_true"], resnet_result["y_pred"], labels=[0, 1, 2])
    print(f"{'':>14s} {'city':>8s} {'highway':>8s} {'resid.':>8s}")
    for i, name in enumerate(SCENE_NAMES):
        row = " ".join(f"{v:>8d}" for v in cm[i])
        print(f"{name:>14s} {row}")

    # Save results
    save_results = []
    for r in results:
        save_results.append(
            {
                k: v
                for k, v in r.items()
                if k not in ("best_oof_proba", "best_model", "y_true", "y_pred")
            }
        )
    save_results.append({"bootstrap_ci_fusion_vs_resnet": ci})
    (RESULTS_DIR / "session4_results.json").write_text(
        json.dumps(save_results, indent=2), encoding="utf-8"
    )
    print(f"\n  Saved: {RESULTS_DIR / 'session4_results.json'}")


if __name__ == "__main__":
    main()
