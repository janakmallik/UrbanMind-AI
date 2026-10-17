"""Baselines for BDD100K scene classification.

Two baselines:
  1. Majority class (predicts 'city street' always)
  2. Frozen ResNet-50 features + logistic regression

Establishes the floor that fine-tuned models must beat.
Logs to MLflow under urbanintel-week5.
"""

from __future__ import annotations

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
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from urbanintel.config import dataset_path
from urbanintel.data.bdd100k import prepare_bdd100k

EXPERIMENT_NAME = "urbanintel-week5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
NUM_WORKERS = 4

LABEL_DIR = dataset_path("bdd100k", "labels")
IMAGES_ROOT = dataset_path("bdd100k", "images", "100k")


class ImageDataset(Dataset):
    """Simple image dataset returning the transformed image and label."""

    def __init__(self, df, transform):
        self.paths = df["image_path"].tolist()
        self.labels = df["label_id"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def extract_features(model, loader, device):
    """Extract features from a frozen backbone. Returns (features, labels)."""
    model.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = model(imgs)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(features_list), np.concatenate(labels_list)


def evaluate(y_true, y_pred, label_names):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0,
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))),
    }


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Device: {DEVICE}")
    print("Loading BDD100K split ...")
    split = prepare_bdd100k(
        train_json=LABEL_DIR / "bdd100k_labels_images_train.json",
        val_json=LABEL_DIR / "bdd100k_labels_images_val.json",
        train_images_dir=IMAGES_ROOT / "train",
        val_images_dir=IMAGES_ROOT / "val",
    )
    label_names = split.label_names
    print(f"Classes: {label_names}")

    dataset_info = {
        "train_rows": len(split.train_df),
        "val_rows": len(split.val_df),
        "test_rows": len(split.test_df),
        "label_names": label_names,
    }

    # ---- Baseline 1: majority class ----
    print()
    print("=== Majority class ===")
    with mlflow.start_run(run_name="bdd-majority"):
        mlflow.log_params({"baseline": "majority_class"})
        mlflow.log_dict(dataset_info, "dataset.json")
        most_common = split.train_df["label_id"].value_counts().idxmax()
        y_pred = np.full(len(split.test_df), most_common)
        m = evaluate(split.test_df["label_id"].values, y_pred, label_names)
        mlflow.log_metrics({"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
        mlflow.log_text(m["report"], "classification_report.txt")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  F1 macro:  {m['f1_macro']:.4f}")

    # ---- Baseline 2: frozen ResNet-50 features + logistic regression ----
    print()
    print("=== Frozen ResNet-50 + Logistic Regression ===")
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = ImageDataset(split.train_df, transform)
    test_ds = ImageDataset(split.test_df, transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Load pretrained ResNet-50, remove the classifier head
    print("Loading pretrained ResNet-50 ...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # remove final FC
    feature_extractor = feature_extractor.to(DEVICE)

    print(f"Extracting train features ({len(train_ds)} images) ...")
    t0 = time.time()
    X_train, y_train = extract_features(feature_extractor, train_loader, DEVICE)
    print(f"  Done in {time.time() - t0:.1f}s. Shape: {X_train.shape}")

    print(f"Extracting test features ({len(test_ds)} images) ...")
    t0 = time.time()
    X_test, y_test = extract_features(feature_extractor, test_loader, DEVICE)
    print(f"  Done in {time.time() - t0:.1f}s. Shape: {X_test.shape}")

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    with mlflow.start_run(run_name="bdd-frozen-resnet50"):
        mlflow.log_params(
            {
                "baseline": "frozen_resnet50_logreg",
                "backbone": "resnet50",
                "pretrained": "IMAGENET1K_V2",
                "feature_dim": X_train_flat.shape[1],
                "classifier": "LogisticRegression",
                "max_iter": 1000,
            }
        )
        mlflow.log_dict(dataset_info, "dataset.json")

        print("Training logistic regression on features ...")
        clf = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        clf.fit(X_train_flat, y_train)
        y_pred = clf.predict(X_test_flat)
        m = evaluate(y_test, y_pred, label_names)
        mlflow.log_metrics({"accuracy": m["accuracy"], "f1_macro": m["f1_macro"]})
        mlflow.log_text(m["report"], "classification_report.txt")
        print(f"  Accuracy:  {m['accuracy']:.4f}")
        print(f"  F1 macro:  {m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
