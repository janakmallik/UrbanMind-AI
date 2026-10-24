"""Verify split reconstruction by reproducing ResNet's published test metrics.

Ground truth (from MLflow classification_report.txt):
    test_accuracy = 0.78
    test_macro_f1 = 0.70 (0.6984)
    test_size = 4932
    per-class support: city=3056, highway=1250, residential=626

If this script does not reproduce these within ±0.01, the split
reconstruction is wrong or the preprocessing is wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from urbanintel.config import dataset_path
from urbanintel.data.bdd100k import prepare_bdd100k

SEED = 42
LABEL_DIR = dataset_path("bdd100k", "labels")
IMAGES_ROOT = dataset_path("bdd100k", "images", "100k")
CHECKPOINT = _REPO_ROOT / "bdd_resnet50_best.pt"

# Ground truth from classification_report.txt
EXPECTED = {
    "test_acc": 0.78,
    "test_f1": 0.70,
    "test_size": 4932,
    "support": {"city street": 3056, "highway": 1250, "residential": 626},
}


class SceneDataset(Dataset):
    """Exact match to train_bdd100k_resnet50.SceneDataset."""

    def __init__(self, df, transform):
        self.paths = df["image_path"].tolist()
        self.labels = df["label_id"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def build_model(num_classes: int) -> nn.Module:
    """Exact match to train_bdd100k_resnet50.build_model."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n=== Step 1 — Reconstruct split (exact match to training) ===")
    split = prepare_bdd100k(
        train_json=LABEL_DIR / "bdd100k_labels_images_train.json",
        val_json=LABEL_DIR / "bdd100k_labels_images_val.json",
        train_images_dir=IMAGES_ROOT / "train",
        val_images_dir=IMAGES_ROOT / "val",
    )
    print(f"Test set size: {len(split.test_df)}")
    assert len(split.test_df) == EXPECTED["test_size"], (
        f"Test set size mismatch: {len(split.test_df)} != {EXPECTED['test_size']}"
    )

    # Verify class support
    support = split.test_df["scene"].value_counts().to_dict()
    print(f"Test class support: {support}")
    for cls, expected_count in EXPECTED["support"].items():
        actual = support.get(cls, 0)
        match = "✓" if actual == expected_count else "✗"
        print(f"  {match} {cls}: expected={expected_count}, actual={actual}")
        assert actual == expected_count, f"Support mismatch for {cls}"

    print("\n=== Step 2 — Load checkpoint (exact architecture) ===")
    assert CHECKPOINT.exists(), f"Checkpoint not found: {CHECKPOINT}"
    model = build_model(num_classes=len(split.label_names)).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded: {CHECKPOINT}")

    print("\n=== Step 3 — Run inference (exact eval_tf) ===")
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_ds = SceneDataset(split.test_df, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=8, num_workers=4, pin_memory=True)

    preds, labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            labels.append(lbls.numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)

    test_acc = float(accuracy_score(y_true, y_pred))
    test_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    print("\n=== Step 4 — Compare to expected ===")
    print(f"  Reproduced accuracy: {test_acc:.4f}  (expected: {EXPECTED['test_acc']:.2f})")
    print(f"  Reproduced macro-F1: {test_f1:.4f}  (expected: {EXPECTED['test_f1']:.2f})")

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(split.label_names))),
        target_names=split.label_names,
        zero_division=0,
    )
    print("\n=== Per-class report ===")
    print(report)

    print("=== Verdict ===")
    acc_ok = abs(test_acc - EXPECTED["test_acc"]) < 0.01
    f1_ok = abs(test_f1 - EXPECTED["test_f1"]) < 0.01

    if acc_ok and f1_ok:
        print("✅ SPLIT RECONSTRUCTION VERIFIED")
        print("   Both accuracy and macro-F1 match within ±0.01.")
        print("   Fusion evaluation can proceed with this split.")
        sys.exit(0)
    else:
        print("❌ SPLIT RECONSTRUCTION FAILED")
        if not acc_ok:
            print(
                f"   Accuracy: {test_acc:.4f} vs expected {EXPECTED['test_acc']:.2f} (off by {abs(test_acc - EXPECTED['test_acc']):.4f})"
            )
        if not f1_ok:
            print(
                f"   Macro-F1: {test_f1:.4f} vs expected {EXPECTED['test_f1']:.2f} (off by {abs(test_f1 - EXPECTED['test_f1']):.4f})"
            )
        print(
            "   Possible causes: preprocessing changed, checkpoint corrupted, split logic changed"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
