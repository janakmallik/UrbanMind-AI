"""Verify ResNet reproduces its published result on reconstructed test set.

Expected:
    accuracy ≈ 0.7800
    macro_f1 ≈ 0.6984

If this doesn't match, the split reconstruction is wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from urbanintel.config import dataset_path

SPLITS_DIR = _REPO_ROOT / "data" / "splits"
IMAGES_DIR = dataset_path("bdd100k", "images/100k")
LABELS_DIR = dataset_path("bdd100k", "labels/100k")
CHECKPOINT = _REPO_ROOT / "bdd_resnet50_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIABLE_CLASSES = ["city street", "highway", "residential"]
LABEL_TO_ID = {name: i for i, name in enumerate(VIABLE_CLASSES)}


class InferenceDataset(Dataset):
    def __init__(self, image_ids: list[str], image_dir: Path, labels: dict[str, str]):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.labels = labels
        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        name = self.image_ids[idx]
        img = Image.open(self.image_dir / name).convert("RGB")
        label = LABEL_TO_ID[self.labels[name]]
        return self.tf(img), label


def load_labels(val_json: Path) -> dict[str, str]:
    data = json.loads(val_json.read_text())
    out = {}
    for r in data:
        scene = r.get("attributes", {}).get("scene")
        if scene in VIABLE_CLASSES:
            out[r["name"]] = scene
    return out


def build_model() -> nn.Module:
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, len(VIABLE_CLASSES))
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    m.load_state_dict(state)
    return m.to(DEVICE).eval()


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, float]:
    preds, labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        p = model(x).argmax(-1).cpu().numpy()
        preds.extend(p.tolist())
        labels.extend(y.tolist())
    return (
        accuracy_score(labels, preds),
        f1_score(labels, preds, average="macro", zero_division=0),
    )


def main() -> None:
    print("Loading labels...")
    val_labels = load_labels(LABELS_DIR / "bdd100k_labels_images_val.json")
    print(f"  {len(val_labels)} labeled val images in 3 classes")

    for split_name in ["clean_test", "contaminated_test", "full_test"]:
        if split_name == "full_test":
            full = json.loads((SPLITS_DIR / "clean_test_ids.json").read_text()) + json.loads(
                (SPLITS_DIR / "contaminated_test_ids.json").read_text()
            )
            ids = full
        else:
            ids = json.loads((SPLITS_DIR / f"{split_name}_ids.json").read_text())

        # Filter to images with labels
        ids = [i for i in ids if i in val_labels]
        print(f"\n=== {split_name}: {len(ids)} images ===")

        ds = InferenceDataset(ids, IMAGES_DIR / "val", val_labels)
        dl = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)

        model = build_model()
        acc, f1 = evaluate(model, dl)
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Macro-F1:  {f1:.4f}")

    print("\n=== Published (from Week 5 model card) ===")
    print("  Accuracy:  0.7800")
    print("  Macro-F1:  0.6984")
    print("\nIf clean_test matches published within ±0.005, split is verified.")


if __name__ == "__main__":
    main()
