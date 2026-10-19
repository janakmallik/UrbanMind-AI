"""IoU-based detection evaluation for YOLOv8n on BDD100K.

Level 2 evaluation: per-class precision/recall/F1 at IoU >= 0.5.

- Confidence threshold: 0.25 (YOLO default)
- IoU threshold: 0.5
- Per-class matching: greedy, highest-IoU first
- Each GT box can match at most one prediction

Classes: person, car, truck, bus
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from ultralytics import YOLO

from urbanintel.config import dataset_path

LABEL_DIR = dataset_path("bdd100k", "labels")
IMAGES_ROOT = dataset_path("bdd100k", "images", "100k")

COCO_TO_NAME = {0: "person", 2: "car", 5: "bus", 7: "truck"}
KEEP_COCO_IDS = set(COCO_TO_NAME.keys())

BDD_TO_NAME = {
    "person": "person",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
}
ALIGNED_CLASSES = ["person", "car", "truck", "bus"]

N_SAMPLE = 200
SEED = 42
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """IoU between two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def match_boxes(
    preds: list[tuple[float, float, float, float]],
    gts: list[tuple[float, float, float, float]],
    iou_thresh: float = IOU_THRESHOLD,
) -> tuple[int, int, int]:
    """Greedy matching between predicted and GT boxes at IoU threshold.

    Returns (true_positives, false_positives, false_negatives).
    """
    matched_gt = [False] * len(gts)
    tp = 0
    for pred in preds:
        best_iou = 0.0
        best_gt_idx = -1
        for j, gt in enumerate(gts):
            if matched_gt[j]:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt[best_gt_idx] = True
            tp += 1

    fp = len(preds) - tp
    fn = sum(1 for m in matched_gt if not m)
    return tp, fp, fn


def main() -> None:
    print("Loading YOLOv8n ...")
    model = YOLO("yolov8n.pt")

    print("Loading BDD100K val labels ...")
    val_json = LABEL_DIR / "bdd100k_labels_images_val.json"
    with open(val_json, "r") as f:
        val_records = json.load(f)
    records_by_name = {r["name"]: r for r in val_records if "name" in r}

    # Valid images: exist on disk AND have a label record
    val_images = [jpg for jpg in (IMAGES_ROOT / "val").glob("*.jpg") if jpg.name in records_by_name]
    print(f"Found {len(val_images)} labeled val images")

    rng = random.Random(SEED)
    sample = rng.sample(val_images, min(N_SAMPLE, len(val_images)))
    print(f"Evaluating on {len(sample)} sampled images\n")

    # Accumulators
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in ALIGNED_CLASSES}

    for i, img_path in enumerate(sample):
        record = records_by_name[img_path.name]

        # GT boxes by class
        gt_by_class = {c: [] for c in ALIGNED_CLASSES}
        for label in record.get("labels", []):
            cat = label.get("category")
            name = BDD_TO_NAME.get(cat)
            if name is None:
                continue
            box = label.get("box2d")
            if box is None:
                continue
            gt_by_class[name].append((box["x1"], box["y1"], box["x2"], box["y2"]))

        # YOLO predictions
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)[0]
        pred_by_class = {c: [] for c in ALIGNED_CLASSES}
        for box in results.boxes:
            cls_id = int(box.cls.item())
            name = COCO_TO_NAME.get(cls_id)
            if name is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_by_class[name].append((x1, y1, x2, y2))

        # Match per class
        for c in ALIGNED_CLASSES:
            tp, fp, fn = match_boxes(pred_by_class[c], gt_by_class[c])
            stats[c]["tp"] += tp
            stats[c]["fp"] += fp
            stats[c]["fn"] += fn

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(sample)} done")

    # ---- Report ----
    print()
    print(f"=== Detection evaluation (IoU >= {IOU_THRESHOLD}, conf >= {CONF_THRESHOLD}) ===")
    print(f"{'Class':<10} {'TP':>6} {'FP':>6} {'FN':>6} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 62)
    for c in ALIGNED_CLASSES:
        s = stats[c]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{c:<10} {tp:>6} {fp:>6} {fn:>6} {precision:>10.3f} {recall:>8.3f} {f1:>8.3f}")

    # Macro-average (only over classes with GT)
    classes_with_gt = [c for c in ALIGNED_CLASSES if (stats[c]["tp"] + stats[c]["fn"]) > 0]
    if classes_with_gt:
        precisions = []
        recalls = []
        f1s = []
        for c in classes_with_gt:
            s = stats[c]
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        print("-" * 62)
        print(
            f"{'macro avg':<10} {'':>6} {'':>6} {'':>6} "
            f"{np.mean(precisions):>10.3f} {np.mean(recalls):>8.3f} {np.mean(f1s):>8.3f}"
        )

    print()
    print("Notes:")
    print(f"- Evaluated on {len(sample)} randomly sampled BDD100K val images (seed={SEED})")
    print(f"- IoU threshold: {IOU_THRESHOLD}, confidence threshold: {CONF_THRESHOLD}")
    print("- YOLOv8n is COCO-pretrained; BDD100K is urban driving footage.")
    print("- These metrics characterize YOLOv8n on THIS evaluation subset; they do")
    print("  not by themselves establish that domain shift caused any errors.")
    print("- mAP@0.5:0.95 and confidence sweeps are out of scope (future work).")


if __name__ == "__main__":
    main()
