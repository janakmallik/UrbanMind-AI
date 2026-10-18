"""Run pretrained YOLOv8n object detection on BDD100K val images.

Qualitative visual demo. NOT a quantitative detection benchmark.

- YOLOv8n trained on COCO (80 classes, general photos)
- BDD100K is urban driving footage (domain shift)
- Reports raw detection counts per class, no precision/recall
- Draws GT boxes from BDD100K labels side-by-side with YOLO predictions
- Future work: IoU-based precision/recall/F1 (requires more rigorous
  evaluation framework, planned for a later phase)

Aligned classes: person, car, truck, bus
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

from urbanintel.config import dataset_path

LABEL_DIR = dataset_path("bdd100k", "labels")
IMAGES_ROOT = dataset_path("bdd100k", "images", "100k")
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "detection_examples"

# COCO class IDs in YOLOv8: 0=person, 2=car, 5=bus, 7=truck
COCO_TO_NAME = {0: "person", 2: "car", 5: "bus", 7: "truck"}
KEEP_COCO_IDS = set(COCO_TO_NAME.keys())

# BDD100K category -> our class name
BDD_TO_NAME = {
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
}
ALIGNED_CLASSES = ["person", "car", "truck", "bus"]

# Box colors for visualization (distinct per class)
CLASS_COLORS = {
    "person": "#FF4444",
    "car": "#44FF44",
    "truck": "#4444FF",
    "bus": "#FFAA00",
}

N_SAMPLE = 200
N_VIS = 12
SEED = 42


def collect_gt_boxes(record: dict) -> dict[str, list[tuple[float, float, float, float]]]:
    """Return {class_name: [(x1,y1,x2,y2), ...]} for aligned classes."""
    out: dict[str, list[tuple[float, float, float, float]]] = {c: [] for c in ALIGNED_CLASSES}
    for label in record.get("labels", []):
        cat = label.get("category")
        bdd_name = BDD_TO_NAME.get(cat)
        if bdd_name is None:
            continue
        box = label.get("box2d")
        if box is None:
            continue
        out[bdd_name].append((box["x1"], box["y1"], box["x2"], box["y2"]))
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YOLOv8n (pretrained on COCO) ...")
    model = YOLO("yolov8n.pt")

    print("Loading BDD100K val labels ...")
    val_json = LABEL_DIR / "bdd100k_labels_images_val.json"
    with open(val_json, "r") as f:
        val_records = json.load(f)

    # Map image name -> full record for GT lookup
    records_by_name = {r["name"]: r for r in val_records if "name" in r}

    # Use image files that exist on disk AND have a val label record
    val_image_paths = []
    for jpg in (IMAGES_ROOT / "val").glob("*.jpg"):
        if jpg.name in records_by_name:
            val_image_paths.append(jpg)

    print(f"Found {len(val_image_paths)} labeled val images on disk")

    rng = random.Random(SEED)
    sample_paths = rng.sample(val_image_paths, min(N_SAMPLE, len(val_image_paths)))
    print(f"Running YOLOv8 on {len(sample_paths)} sampled images ...")

    total_yolo = Counter()
    total_gt = Counter()

    vis_examples = []

    for i, img_path in enumerate(sample_paths):
        record = records_by_name[img_path.name]
        gt_boxes = collect_gt_boxes(record)

        # Run YOLO
        results = model(str(img_path), verbose=False)[0]
        yolo_boxes: dict[str, list[tuple[float, float, float, float]]] = {
            c: [] for c in ALIGNED_CLASSES
        }
        for box in results.boxes:
            cls_id = int(box.cls.item())
            name = COCO_TO_NAME.get(cls_id)
            if name is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            yolo_boxes[name].append((x1, y1, x2, y2))

        for name in ALIGNED_CLASSES:
            total_yolo[name] += len(yolo_boxes[name])
            total_gt[name] += len(gt_boxes[name])

        if len(vis_examples) < N_VIS:
            # Prefer images with at least one GT or YOLO box in our classes
            if any(len(gt_boxes[c]) > 0 for c in ALIGNED_CLASSES):
                vis_examples.append((img_path, yolo_boxes, gt_boxes))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample_paths)} done")

    # ---- Aggregate report (raw counts only, no precision/recall claims) ----
    print(f"\n=== Raw detection counts across {len(sample_paths)} sampled images ===")
    print(f"{'Class':<10} {'YOLO detections':>16} {'GT instances':>14}")
    for name in ALIGNED_CLASSES:
        print(f"{name:<10} {total_yolo[name]:>16} {total_gt[name]:>14}")
    print()
    print("NOTE: These are raw counts, not precision/recall.")
    print("YOLO was trained on COCO (80 classes); BDD100K is urban driving")
    print("footage. Detection counts do not measure match quality. IoU-based")
    print("evaluation is planned as a future extension.")

    # ---- Visualizations ----
    print(f"\nSaving {len(vis_examples)} visualizations to {OUTPUT_DIR} ...")
    for idx, (img_path, yolo_boxes, gt_boxes) in enumerate(vis_examples):
        img = np.array(Image.open(img_path))

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Left: YOLO detections
        axes[0].imshow(img)
        for name, boxes in yolo_boxes.items():
            for x1, y1, x2, y2 in boxes:
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=CLASS_COLORS[name],
                    facecolor="none",
                )
                axes[0].add_patch(rect)
                axes[0].text(
                    x1,
                    y1 - 5,
                    name,
                    fontsize=8,
                    color="white",
                    bbox=dict(facecolor=CLASS_COLORS[name], alpha=0.7, pad=1),
                )
        yolo_summary = ", ".join(
            f"{c}={len(yolo_boxes[c])}" for c in ALIGNED_CLASSES if yolo_boxes[c]
        )
        axes[0].set_title(f"YOLOv8n detections: {yolo_summary or 'none'}")
        axes[0].axis("off")

        # Right: BDD100K ground truth
        axes[1].imshow(img)
        for name, boxes in gt_boxes.items():
            for x1, y1, x2, y2 in boxes:
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor=CLASS_COLORS[name],
                    facecolor="none",
                    linestyle="--",
                )
                axes[1].add_patch(rect)
                axes[1].text(
                    x1,
                    y1 - 5,
                    name,
                    fontsize=8,
                    color="white",
                    bbox=dict(facecolor=CLASS_COLORS[name], alpha=0.7, pad=1),
                )
        gt_summary = ", ".join(f"{c}={len(gt_boxes[c])}" for c in ALIGNED_CLASSES if gt_boxes[c])
        axes[1].set_title(f"BDD100K GT: {gt_summary or 'none'}")
        axes[1].axis("off")

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"{idx:02d}_{img_path.stem}.png"
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(vis_examples)} visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
