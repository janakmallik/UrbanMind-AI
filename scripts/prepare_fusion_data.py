"""Session 2 — Prepare clean fusion data.

Reconstructs ResNet's split (verified in Session 1), reconstructs
SegFormer's training split (6,000 images), computes contamination
intersections, saves clean/excluded image ID lists.

Writes:
    data/splits/resnet_val_clean.json
    data/splits/resnet_val_excluded.json
    data/splits/resnet_test_clean.json
    data/splits/resnet_test_excluded.json
    data/splits/fusion_metadata.json  (counts, class balance)

Verifies:
    - All IDs resolve to exactly one image and one label
    - No duplicate IDs
    - Clean test accuracy stays near 0.78
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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from urbanintel.config import dataset_path
from urbanintel.data.bdd100k import prepare_bdd100k
from urbanintel.data.bdd100k_segmentation import prepare_segmentation

SEED = 42
LABEL_DIR = dataset_path("bdd100k", "labels")
IMAGES_ROOT = dataset_path("bdd100k", "images", "100k")
SEG_ROOT = dataset_path("bdd100k", "seg")
CHECKPOINT = _REPO_ROOT / "bdd_resnet50_best.pt"
OUT_DIR = _REPO_ROOT / "data" / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class SceneDataset(Dataset):
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
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    preds, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds.append(logits.argmax(dim=-1).cpu().numpy())
        labels.append(lbls.numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    return y_true, y_pred


def get_segformer_training_ids() -> set[str]:
    """Return the 6,000 image names SegFormer actually trained on."""
    seg_split = prepare_segmentation(SEG_ROOT)
    # seg_split.train_df has "image_path" column — extract just the filename
    return {Path(p).name for p in seg_split.train_df["image_path"].tolist()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Reconstruct ResNet split ---
    print("\n=== Reconstructing ResNet split ===")
    split = prepare_bdd100k(
        train_json=LABEL_DIR / "bdd100k_labels_images_train.json",
        val_json=LABEL_DIR / "bdd100k_labels_images_val.json",
        train_images_dir=IMAGES_ROOT / "train",
        val_images_dir=IMAGES_ROOT / "val",
    )
    assert len(split.test_df) == 4932, f"Test size {len(split.test_df)} != 4932"
    assert len(split.val_df) == 4932, f"Val size {len(split.val_df)} != 4932"

    # --- Get SegFormer's actual 6,000 training IDs ---
    print("\n=== Reconstructing SegFormer training split ===")
    seg_train_ids = get_segformer_training_ids()
    print(f"SegFormer trained on: {len(seg_train_ids)} images")
    assert len(seg_train_ids) == 6000, f"Expected 6000 training images, got {len(seg_train_ids)}"

    # --- Compute contamination ---
    print("\n=== Computing contamination ===")
    resnet_val_ids = set(split.val_df["image_name"].tolist())
    resnet_test_ids = set(split.test_df["image_name"].tolist())

    val_excluded = resnet_val_ids & seg_train_ids
    val_clean = resnet_val_ids - seg_train_ids
    test_excluded = resnet_test_ids & seg_train_ids
    test_clean = resnet_test_ids - seg_train_ids

    print(
        f"  Val:  {len(val_clean)} clean, {len(val_excluded)} excluded "
        f"({100 * len(val_excluded) / len(resnet_val_ids):.2f}% contamination)"
    )
    print(
        f"  Test: {len(test_clean)} clean, {len(test_excluded)} excluded "
        f"({100 * len(test_excluded) / len(resnet_test_ids):.2f}% contamination)"
    )

    # --- Verify all clean IDs resolve to exactly one image and one label ---
    print("\n=== Verifying clean IDs resolve correctly ===")
    for split_name, df, clean_ids in [
        ("val", split.val_df, val_clean),
        ("test", split.test_df, test_clean),
    ]:
        names = df["image_name"].tolist()
        assert len(names) == len(set(names)), f"Duplicate image_names in {split_name}"
        paths = df["image_path"].tolist()
        assert len(paths) == len(set(paths)), f"Duplicate image_paths in {split_name}"
        # Every clean ID should have exactly one row
        for cid in clean_ids:
            count = (df["image_name"] == cid).sum()
            assert count == 1, f"Clean ID {cid} has {count} rows in {split_name} (expected 1)"
            # Path exists
            path = df[df["image_name"] == cid]["image_path"].iloc[0]
            assert Path(path).exists(), f"Path missing for {cid}: {path}"
    print("  ✓ All clean IDs resolve to exactly one image + one label + one existing path")

    # --- Save ID lists ---
    print("\n=== Saving ID lists ===")
    for name, ids in [
        ("resnet_val_clean", val_clean),
        ("resnet_val_excluded", val_excluded),
        ("resnet_test_clean", test_clean),
        ("resnet_test_excluded", test_excluded),
    ]:
        out = OUT_DIR / f"{name}.json"
        out.write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")
        print(f"  {name}: {len(ids)} IDs → {out.name}")

    # --- Class balance on clean sets ---
    print("\n=== Class balance ===")
    metadata = {}
    for split_name, df, clean_ids in [
        ("val", split.val_df, val_clean),
        ("test", split.test_df, test_clean),
    ]:
        subset = df[df["image_name"].isin(clean_ids)]
        counts = subset["scene"].value_counts().to_dict()
        print(f"  {split_name} clean: {dict(counts)}")
        metadata[f"{split_name}_clean"] = {
            "n": len(subset),
            "class_counts": counts,
        }

    # --- Verify ResNet on clean test ---
    print("\n=== ResNet on clean test subset ===")
    model = build_model(num_classes=len(split.label_names)).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    clean_test_df = split.test_df[split.test_df["image_name"].isin(test_clean)]
    test_ds = SceneDataset(clean_test_df, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=8, num_workers=4, pin_memory=True)

    y_true, y_pred = evaluate(model, test_loader, device)
    clean_acc = float(accuracy_score(y_true, y_pred))
    clean_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    print(f"  Clean test accuracy:  {clean_acc:.4f}")
    print(f"  Clean test macro-F1:  {clean_f1:.4f}")
    print(f"  (Full test was:       0.7800 / 0.6984)")
    print(f"  Delta from full:      {clean_acc - 0.78:+.4f} / {clean_f1 - 0.6984:+.4f}")

    metadata["clean_test_metrics"] = {
        "accuracy": clean_acc,
        "macro_f1": clean_f1,
        "n": len(clean_test_df),
    }
    metadata["full_test_metrics"] = {
        "accuracy": 0.78,
        "macro_f1": 0.6984,
        "n": 4932,
    }

    # --- Save metadata ---
    (OUT_DIR / "fusion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\n  Metadata → fusion_metadata.json")

    # --- Verdict ---
    print("\n=== Verdict ===")
    if abs(clean_acc - 0.78) > 0.02:
        print(f"⚠️  Clean test accuracy drift: {clean_acc:.4f} vs 0.78 (> 0.02)")
        print("    The 224 excluded images may not be a random slice of the pool.")
        print("    Document as a finding. Investigate class distribution of excluded images.")
    else:
        print(f"✅ Clean test accuracy stable ({clean_acc:.4f} vs 0.78)")

    print("\n✅ Session 2 complete. Ready for Session 3 (feature extraction).")


if __name__ == "__main__":
    main()
