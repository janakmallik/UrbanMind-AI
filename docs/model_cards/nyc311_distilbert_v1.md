# Model Card: NYC 311 Descriptor Classifier (DistilBERT v1)

## What this model does

Classifies NYC 311 complaint type from the `descriptor` field.

## What this model does NOT do

- It does **not** classify complaints from free text written by residents.
  It classifies them from a short, structured field assigned by 311 operators.
- It does **not** generalize to descriptors not seen during training.
  Every test descriptor appeared in training (99.97% test-row overlap).
- It does **not** demonstrate that transformers are necessary for this task.
  A simple lookup table achieves 99.97% accuracy; TF-IDF + LogReg achieves 100%.

## Task limitation

The `descriptor` field is assigned **at the same time as** `complaint_type`
by the intake operator. It functions as a short code that nearly determines
the label. This experiment is a valid DistilBERT workflow demonstration, but
it is **not** a semantic classification problem.

## Architecture

- Base model: `distilbert-base-uncased` (66M parameters)
- Task head: linear classifier (5 classes)
- Max sequence length: 16 tokens
- Trained 3 epochs, AdamW (lr=2e-5), batch size 32

## Training

- Dataset: NYC 311 sample, top-5 complaint types
- Split: stratified 70/15/15, random_state=42
- Train: 37,194 | Val: 7,970 | Test: 7,971
- Checkpoint selection: best validation macro-F1
- Best epoch: 1 (val F1 = 1.0000)

## Results (test set)

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority class | 0.3387 | 0.1012 |
| Descriptor lookup | 0.9997 | 0.9994 |
| TF-IDF + LogReg | 1.0000 | 1.0000 |
| **DistilBERT** | **1.0000** | **1.0000** |

## Interpretation

DistilBERT matches the ceiling established by the deterministic lookup
baseline. It does not exceed TF-IDF + LogReg. This is the expected result
for a trivial task.

**Value of this experiment:**
- Confirms the HuggingFace fine-tuning workflow works end-to-end
- Establishes a template for future text classification tasks
- Produces a documented model card that honestly states the limitation

**Value it does NOT demonstrate:**
- Semantic language understanding
- Generalization to unseen text
- Advantage of transformers over classical methods

## Limitations

- Task is deterministic; model is essentially a memorization exercise
- Does not generalize beyond training descriptor vocabulary
- Not suitable for deployment as a complaint classifier

## Reproduce

```powershell
uv run python scripts\run_distilbert_nyc311.py