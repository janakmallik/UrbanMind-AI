"""Fine-tune DistilBERT for NYC 311 descriptor classification.

This is a HuggingFace learning exercise. The task is nearly deterministic
(descriptor → complaint type mapping), so we expect ~100% accuracy.
The goal is to learn the fine-tuning workflow, not to demonstrate NLP capability.

Logs to MLflow under urbanintel-week4.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DB_PATH = (_REPO_ROOT / "mlflow.db").as_posix()
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_DB_PATH}"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from urbanintel.config import dataset_path
from urbanintel.data.loaders import load_nyc311
from urbanintel.data.nyc311_text import prepare_text_data
from urbanintel.evaluation.classification import (
    masked_accuracy,
    masked_f1_macro,
    per_class_report,
)

EXPERIMENT_NAME = "urbanintel-week4"
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 16  # descriptors are 1-4 words
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DescriptorDataset(Dataset):
    """Tokenized text dataset for classification."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def evaluate(model, loader, device, num_classes: int):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch["labels"].numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return labels, preds


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Device: {DEVICE}")
    print(f"Loading NYC 311 ...")
    df = load_nyc311(dataset_path("nyc311", "nyc311_sample.csv"), validate=False)
    split = prepare_text_data(df)

    label_names = split.label_names
    num_classes = len(label_names)
    print(f"Classes: {label_names}")
    print(
        f"Train: {len(split.train_texts):,} | Val: {len(split.val_texts):,} | Test: {len(split.test_texts):,}"
    )

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = DescriptorDataset(split.train_texts, split.train_labels, tokenizer, MAX_LEN)
    val_ds = DescriptorDataset(split.val_texts, split.val_labels, tokenizer, MAX_LEN)
    test_ds = DescriptorDataset(split.test_texts, split.test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    with mlflow.start_run(run_name="distilbert-descriptor"):
        mlflow.log_params(
            {
                "model": MODEL_NAME,
                "max_len": MAX_LEN,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "seed": SEED,
                "device": DEVICE,
                "n_params": n_params,
                "num_classes": num_classes,
                "task_note": "deterministic descriptor→label lookup; not semantic classification",
            }
        )
        mlflow.log_dict(
            {
                "train_rows": len(split.train_texts),
                "val_rows": len(split.val_texts),
                "test_rows": len(split.test_texts),
                "label_names": label_names,
                "test_row_overlap_fraction": split.overlap_report["test_row_overlap_fraction"],
            },
            "dataset.json",
        )

        best_val_f1 = -1.0
        best_state = None
        best_epoch = 0

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss, n_batches = 0.0, 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            train_loss = total_loss / n_batches
            y_true, y_pred = evaluate(model, val_loader, DEVICE, num_classes)
            val_acc = masked_accuracy(y_true, y_pred)
            val_f1 = masked_f1_macro(y_true, y_pred, num_classes)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_accuracy": val_acc, "val_f1_macro": val_f1},
                step=epoch,
            )
            print(
                f"Epoch {epoch}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"Best epoch: {best_epoch} (val_f1={best_val_f1:.4f})")
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_f1_macro", best_val_f1)

        model.load_state_dict(best_state)
        y_true, y_pred = evaluate(model, test_loader, DEVICE, num_classes)
        test_acc = masked_accuracy(y_true, y_pred)
        test_f1 = masked_f1_macro(y_true, y_pred, num_classes)
        report = per_class_report(y_true, y_pred, label_names)

        mlflow.log_metrics({"test_accuracy": test_acc, "test_f1_macro": test_f1})
        mlflow.log_text(report, "classification_report.txt")

        print()
        print(f"Test accuracy:   {test_acc:.4f}")
        print(f"Test F1 macro:   {test_f1:.4f}")
        print()
        print("=== Comparison with baselines ===")
        print(f"{'Model':<28} {'Accuracy':>10} {'F1 Macro':>10}")
        print(f"{'Majority class':<28} {0.3387:>10.4f} {0.1012:>10.4f}")
        print(f"{'Descriptor lookup':<28} {0.9997:>10.4f} {0.9994:>10.4f}")
        print(f"{'TF-IDF + LogReg':<28} {1.0000:>10.4f} {1.0000:>10.4f}")
        print(f"{'DistilBERT':<28} {test_acc:>10.4f} {test_f1:>10.4f}")


if __name__ == "__main__":
    main()
