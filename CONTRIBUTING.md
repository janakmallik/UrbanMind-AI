# Contributing to UrbanMind-AI

This is a personal learning and portfolio project. Issues and
discussion are welcome, but pull requests are not being accepted at
this time.

If you're using this repo as a template for your own project, here's
how the pieces fit together.

---

## How to add a new modality

Every modality in this repo follows the same 8-step pattern. To add a
ninth (e.g., point cloud, remote sensing, tabular time series):

1. **Dataset loader** — `src/urbanintel/data/{name}.py`
   - Load from the canonical source (or cache to `data/`)
   - Validate schema with pandas or pydantic
   - Return splits (chronological for time, stratified for classification)
   - Add a fixture to `tests/unit/test_{name}.py` that runs without the
     real dataset (synthetic data)

2. **Preprocessing** (if needed) — `scripts/precompute_{name}.py`
   - Cache expensive transforms (mel-spectrograms, clip extraction,
     segmentation masks) to a memmap or parquet
   - Write a `ready.txt` marker when done
   - Skip on subsequent runs

3. **Trainer** — `scripts/train_{name}.py`
   - Same shape as existing trainers:
     - `argparse` with `--epochs`, `--batch-size`, `--lr`, `--num-workers`
     - MLflow experiment `urbanintel-weekN` (increment N)
     - Best-epoch checkpoint selection on val metric
     - Early stopping (patience 3, min_delta 0.005)
     - Log per-epoch train loss + val metric
     - Save best checkpoint to `checkpoints/{name}_best.pt`

4. **Evaluation** — inline or in `src/urbanintel/evaluation/`
   - Report the canonical metric for the task (accuracy/F1 for
     classification, MAE/RMSE for forecasting, mIoU for segmentation)
   - Include per-class breakdowns when class balance matters
   - Include a baseline (majority class, random, or persistence)

5. **Model card** — `docs/model_cards/{name}_v1.md`
   - Follow the structure of existing cards: "What this model does",
     "What this does NOT do", "Architecture", "Training", "Results",
     "Known limitations", "Reproduce", "Citation"
   - Be honest about failure modes — this is the project's identity

6. **Dataset card** — `docs/dataset_cards/{name}.md`
   - What the dataset is, taxonomy, class distribution, splits,
     preprocessing, known limitations, citation

7. **Experiments entry** — `docs/experiments.md`
   - `## YYYY-MM-DD — {Title} (Week N)`
   - Hypothesis → setup → result → takeaway
   - Include the full metrics table
   - Document failure modes

8. **Git tag** — `vX.Y.Z-weekN-{slug}`

---

## How to run tests

```powershell
uv run pytest -q