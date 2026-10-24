"""Session 3 — Extract features from ResNet, SegFormer, YOLO on clean splits.

For each image in clean_val (4,727) and clean_test (4,745):
    - ResNet-50    → 3 class probabilities (softmax over logits)
    - SegFormer-B0 → 5 pixel-share fractions (from predicted mask)
    - YOLOv8n      → 6 detection counts (person, car, truck, bus,
                     traffic light, stop sign)

Output: parquet files with 14 features + scene label + image ID.
        data/features/bdd100k_fusion/{train,test}.parquet

NO model evaluation in this script. Feature extraction only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import SegformerForSemanticSegmentation
from ultralytics import YOLO

from urbanintel.config import dataset_path
from urbanintel.data.bdd100k import prepare_bdd100k

# Paths
LABEL_DIR = dataset_path("bdd100k", "labels")
IMAGES_ROOT = dataset_path("bdd100k", "images", "100k")
SEG_ROOT = dataset_path("bdd100k", "seg")
SPLITS_DIR = _REPO_ROOT / "data" / "splits"
OUT_DIR = _REPO_ROOT / "data" / "features" / "bdd100k_fusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESNET_CKPT = _REPO_ROOT / "bdd_resnet50_best.pt"
SEGFORMER_CKPT = _REPO_ROOT / "bdd_segformer_b0_best.pt"
YOLO_CKPT = _REPO_ROOT / "yolov8n.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIABLE_CLASSES = ["city street", "highway", "residential"]
LABEL_TO_ID = {name: i for i, name in enumerate(VIABLE_CLASSES)}
NUM_SCENE_CLASSES = 3

# SegFormer: 5 segmentation classes (road, sidewalk, vehicle, person, other)
SEG_NUM_CLASSES = 5

# YOLO COCO class IDs → our 6 classes
# Order matches run_yolov8_detection.py's COCO_TO_NAME dict:
# {0: person, 2: car, 5: bus, 7: truck}
# Extended with traffic light (9) and stop sign (11).
YOLO_CLASS_IDS = {
    0: "yolo_person",
    2: "yolo_car",
    5: "yolo_bus",
    7: "yolo_truck",
    9: "yolo_traffic_light",
    11: "yolo_stop_sign",
}


# SegFormer expects raw RGB tensors in [0, 1] (its processor handles normalization)
# But the fine-tuning script used SegDataset — we need to match its preprocessing.
# From the segmentation loader, images are loaded as-is (PIL→tensor) at resolution=224.
class SegformerPreprocess:
    """Exact match to SegDataset.__getitem__ in train_bdd100k_segformer.py.
    Training used resolution=512, PIL.BILINEAR resize, then numpy
    normalization (NOT torchvision transforms).
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, img):
        img = img.resize((512, 512), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_np = (img_np - self.MEAN) / self.STD
        return torch.from_numpy(img_np).permute(2, 0, 1).float()


# --- Dataset ---


class InferenceDataset(Dataset):
    """Loads an image, returns (image, image_name, label_id)."""

    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        return self.transform(img), row["image_name"], int(row["label_id"])


# --- Model builders ---


def build_resnet() -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, NUM_SCENE_CLASSES)
    model.load_state_dict(torch.load(RESNET_CKPT, map_location="cpu", weights_only=True))
    return model.to(DEVICE).eval()


def build_segformer() -> SegformerForSemanticSegmentation:
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=SEG_NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(SEGFORMER_CKPT, map_location="cpu", weights_only=True))
    return model.to(DEVICE).eval()


# --- Feature extraction helpers ---


@torch.no_grad()
def extract_resnet(model, loader) -> tuple[np.ndarray, list[str], list[int]]:
    """Returns (probs [N,3], image_names, labels)."""
    all_probs, all_names, all_labels = [], [], []
    for imgs, names, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_names.extend(names)
        all_labels.extend(labels.tolist())
    return np.concatenate(all_probs), all_names, all_labels


@torch.no_grad()
def extract_segformer(model, loader) -> tuple[np.ndarray, list[str], list[int]]:
    """Returns (pixel_share [N,5], image_names, labels)."""
    all_shares, all_names, all_labels = [], [], []
    for imgs, names, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        outputs = model(pixel_values=imgs)
        # Upsample logits to 224x224 (input size)
        logits = torch.nn.functional.interpolate(
            outputs.logits, size=imgs.shape[-2:], mode="bilinear", align_corners=False
        )
        preds = logits.argmax(dim=1)  # [B, H, W]
        # Pixel shares per class
        B = preds.shape[0]
        shares = np.zeros((B, SEG_NUM_CLASSES), dtype=np.float32)
        for c in range(SEG_NUM_CLASSES):
            shares[:, c] = (preds == c).float().mean(dim=[1, 2]).cpu().numpy()
        all_shares.append(shares)
        all_names.extend(names)
        all_labels.extend(labels.tolist())
    return np.concatenate(all_shares), all_names, all_labels


def extract_yolo(model, df) -> tuple[np.ndarray, list[str]]:
    """Returns (counts [N,6], image_names).

    YOLO runs per-image (its API is per-image, not batched).
    """
    counts = np.zeros((len(df), len(YOLO_CLASS_IDS)), dtype=np.int32)
    names = []
    for i, row in df.iterrows():
        results = model(str(row["image_path"]), verbose=False)[0]
        img_counts = {cls_id: 0 for cls_id in YOLO_CLASS_IDS}
        for box in results.boxes:
            cls_id = int(box.cls.item())
            if cls_id in img_counts:
                img_counts[cls_id] += 1
        for j, cls_id in enumerate(YOLO_CLASS_IDS):
            counts[i, j] = img_counts[cls_id]
        names.append(row["image_name"])
        if (i + 1) % 500 == 0:
            print(f"    YOLO: {i + 1}/{len(df)}")
    return counts, names


# --- Main ---


def main() -> None:
    print(f"Device: {DEVICE}")

    # --- Assert class order matches training ---
    # From prepare_bdd100k: VIABLE_CLASSES = ["city street", "highway", "residential"]
    # and LABEL_TO_ID maps those to 0, 1, 2.
    # Model output index 0 = city street, 1 = highway, 2 = residential.
    assert VIABLE_CLASSES == ["city street", "highway", "residential"], (
        f"Class order changed: {VIABLE_CLASSES}"
    )
    print(f"✓ Class order confirmed: {VIABLE_CLASSES}")
    clean_val = set(json.loads((SPLITS_DIR / "resnet_val_clean.json").read_text()))
    clean_test = set(json.loads((SPLITS_DIR / "resnet_test_clean.json").read_text()))
    print(f"  Clean val: {len(clean_val)}")
    print(f"  Clean test: {len(clean_test)}")

    # --- ID normalization check ---
    print("\n=== ID normalization check ===")
    split = prepare_bdd100k(
        train_json=LABEL_DIR / "bdd100k_labels_images_train.json",
        val_json=LABEL_DIR / "bdd100k_labels_images_val.json",
        train_images_dir=IMAGES_ROOT / "train",
        val_images_dir=IMAGES_ROOT / "val",
    )
    assert list(split.label_names) == VIABLE_CLASSES, (
        f"Unexpected ResNet label order: {split.label_names}"
    )
    sample_val_id = next(iter(clean_val))
    sample_test_id = next(iter(clean_test))
    print(f"  Sample clean_val ID: {sample_val_id}")
    print(f"  Sample clean_test ID: {sample_test_id}")

    # Verify ALL clean IDs resolve to exactly one row
    for split_name, df, id_set in [
        ("val", split.val_df, clean_val),
        ("test", split.test_df, clean_test),
    ]:
        for cid in id_set:
            count = (df["image_name"] == cid).sum()
            assert count == 1, f"{split_name} ID {cid} has {count} rows"
        print(f"  ✓ {split_name}: all {len(id_set)} clean IDs resolve to exactly one row")

    # --- Build model-specific inference DataFrames ---
    val_df = split.val_df[split.val_df["image_name"].isin(clean_val)].reset_index(drop=True)
    test_df = split.test_df[split.test_df["image_name"].isin(clean_test)].reset_index(drop=True)
    print(f"\n  Val rows (after filter): {len(val_df)}")
    print(f"  Test rows (after filter): {len(test_df)}")

    # --- Load models ---
    print("\n=== Loading models ===")
    print("  ResNet-50...")
    resnet = build_resnet()
    print("  SegFormer-B0...")
    segformer = build_segformer()  # ← definition is here
    # From bdd100k_segmentation.py: OUR_CLASSES = ["road", "sidewalk", "vehicle", "person", "other"]
    # SegFormer output index 0 = road, 1 = sidewalk, 2 = vehicle, 3 = person, 4 = other.
    from urbanintel.data.bdd100k_segmentation import OUR_CLASSES

    assert OUR_CLASSES == ["road", "sidewalk", "vehicle", "person", "other"], (
        f"SegFormer class order changed: {OUR_CLASSES}"
    )
    print(f"  ✓ SegFormer class order: {OUR_CLASSES}")
    print("  YOLOv8n...")
    yolo = YOLO(str(YOLO_CKPT))

    # --- Preprocessing (match training scripts) ---
    resnet_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    segformer_tf = SegformerPreprocess()

    # --- Extract features per split ---
    for split_name, df in [("train", val_df), ("test", test_df)]:
        print(f"\n=== Extracting features on {split_name} ({len(df)} images) ===")

        # ResNet
        print("  ResNet probs...")
        resnet_loader = DataLoader(
            InferenceDataset(df, resnet_tf),
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )
        resnet_probs, resnet_names, resnet_labels = extract_resnet(resnet, resnet_loader)
        print(f"    shape: {resnet_probs.shape}, first row: {resnet_probs[0]}")

        # SegFormer
        print("  SegFormer pixel shares...")
        seg_loader = DataLoader(
            InferenceDataset(df, segformer_tf),
            batch_size=4,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )
        seg_shares, seg_names, seg_labels = extract_segformer(segformer, seg_loader)
        print(f"    shape: {seg_shares.shape}, first row: {seg_shares[0]}")

        # YOLO (per-image, no batching)
        print("  YOLO counts (per-image, may take a few min)...")
        yolo_counts, yolo_names = extract_yolo(yolo, df)
        print(f"    shape: {yolo_counts.shape}, first row: {yolo_counts[0]}")

        # --- Verify names align across all three ---
        assert resnet_names == seg_names == yolo_names, "Name order mismatch"
        assert resnet_labels == seg_labels, "Label order mismatch"

        # --- Build output DataFrame ---
        out = pd.DataFrame(
            {
                "image_name": resnet_names,
                "scene_label": resnet_labels,
                "resnet_city": resnet_probs[:, 0],
                "resnet_highway": resnet_probs[:, 1],
                "resnet_residential": resnet_probs[:, 2],
                "seg_road": seg_shares[:, 0],
                "seg_sidewalk": seg_shares[:, 1],
                "seg_vehicle": seg_shares[:, 2],
                "seg_person": seg_shares[:, 3],
                "seg_other": seg_shares[:, 4],
                "yolo_person": yolo_counts[:, 0],
                "yolo_car": yolo_counts[:, 1],
                "yolo_bus": yolo_counts[:, 2],
                "yolo_truck": yolo_counts[:, 3],
                "yolo_traffic_light": yolo_counts[:, 4],
                "yolo_stop_sign": yolo_counts[:, 5],
            }
        )

        # --- Verify no NaNs or infs ---
        assert not out.isnull().any().any(), "NaN found in features"
        assert not np.isinf(out.select_dtypes(include=[np.number]).values).any(), "Inf found"

        # --- Save ---
        out_path = OUT_DIR / f"{split_name}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"  Saved: {out_path} ({len(out)} rows, {len(out.columns)} cols)")

        # --- Print feature stats ---
        print(f"\n  Feature stats ({split_name}):")
        feature_cols = [c for c in out.columns if c not in ("image_name", "scene_label")]
        for col in feature_cols:
            s = out[col]
            print(f"    {col:22s}  min={s.min():8.4f}  mean={s.mean():8.4f}  max={s.max():8.4f}")

        print(f"\n  Label distribution ({split_name}):")
        print(out["scene_label"].value_counts().sort_index().to_string())

    # --- Save provenance ---
    provenance = {
        "session": "week9_session3",
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "checkpoints": {
            "resnet": str(RESNET_CKPT),
            "segformer": str(SEGFORMER_CKPT),
            "yolo": str(YOLO_CKPT),
        },
        "preprocessing": {
            "resnet": "Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)",
            "segformer": "Resize(512x512, BILINEAR) -> numpy/255 -> Normalize(ImageNet)",
            "yolo": "ultralytics default",
        },
        "class_mappings": {
            "resnet_output_order": ["city street", "highway", "residential"],
            "segformer_output_order": ["road", "sidewalk", "vehicle", "person", "other"],
            "yolo_class_ids": {
                "0": "person",
                "2": "car",
                "5": "bus",
                "7": "truck",
                "9": "traffic_light",
                "11": "stop_sign",
            },
        },
        "feature_columns": list(out.columns) if "out" in dir() else None,
        "splits": {
            "train": "data/splits/resnet_val_clean.json",
            "test": "data/splits/resnet_test_clean.json",
        },
        "counts": {
            "train_rows": len(val_df),
            "test_rows": len(test_df),
        },
    }
    (OUT_DIR / "provenance.json").write_text(
        __import__("json").dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"  Provenance → provenance.json")

    print("\n✅ Session 3 complete. Features saved to:", OUT_DIR)
    print("   Next: Session 4 — fusion head training + evaluation")


if __name__ == "__main__":
    main()
