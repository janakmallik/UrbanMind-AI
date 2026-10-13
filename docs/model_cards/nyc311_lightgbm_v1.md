# Model Card: NYC 311 Complaint Classifier (LightGBM v1)

## Model Overview

| Field | Value |
|---|---|
| Model name | `nyc311-lightgbm-v1` |
| Version | 1.0 |
| Task | Multi-class classification of NYC 311 complaint type |
| Classes | Top 5: HEAT/HOT WATER, Illegal Parking, Noise - Residential, Blocked Driveway, Traffic Signal Condition |
| Author | Janak Mallik |
| Date | 2026-10-13 |
| Git tag | `v0.2.0-week1` |
| MLflow experiment | `urbanintel-week1` |
| MLflow run name | `optuna-lightgbm` |

## Intended Use

Classify the **type** of an incoming NYC 311 service request using only
information available **before** the complaint is categorized by an operator.

**Intended users:** Researchers, analysts, and engineers exploring urban
complaint patterns. This is a **research prototype**, not a system that should
drive any real decision about residents or services.

## Out-of-Scope Use

- Do **not** use to route or triage real 311 calls
- Do **not** deploy to production without further validation on recent data
- Do **not** use as a substitute for the classification performed by 311 intake
- Do **not** infer anything about individuals or neighborhoods from predictions

## Model Details

### Architecture
LightGBM gradient-boosted trees with the following tuned hyperparameters
(discovered via 15-trial Optuna search):

| Parameter | Value |
|---|---|
| n_estimators | 700 |
| learning_rate | 0.0677 |
| num_leaves | 135 |
| min_child_samples | 11 |
| subsample | 0.9381 |
| colsample_bytree | 0.8396 |
| reg_alpha | 0.1523 |
| reg_lambda | 0.1826 |

### Features (all available at prediction time)

| Feature | Type | Source | Notes |
|---|---|---|---|
| borough | categorical | `Borough` | One-hot encoded |
| zip_binned | categorical | `Incident Zip` | Top-50 ZIPs + "OTHER" |
| hour | numeric | `Created Date` | 0–23 |
| dow | numeric | `Created Date` | 0–6 |
| month | numeric | `Created Date` | 1–12 |
| is_weekend | binary | `Created Date` | dow >= 5 |
| hour_sin, hour_cos | numeric | Derived | Cyclical encoding |
| dow_sin, dow_cos | numeric | Derived | Cyclical encoding |
| month_sin, month_cos | numeric | Derived | Cyclical encoding |
| latitude | numeric | `Latitude` | Median-imputed |
| longitude | numeric | `Longitude` | Median-imputed |

### Explicitly Excluded Features

The following were tested and **removed** as features because they are
assigned **at the same time as the target** (see dataset card):

- `descriptor` (Problem Detail) — sub-label refinement
- `agency` — assigned during classification (HPD → HEAT, DOT → Traffic Signal, NYPD → most others)
- `location_type` — assigned during classification

Each of these showed ~100% concentration against `complaint_type` in crosstabs,
indicating they encode the label rather than provide contextual signal.

## Training Data

| Field | Value |
|---|---|
| Source | NYC Open Data — 311 Service Requests (2010–present) |
| Sample | 100,000 rows, bounded download |
| Filtered | Top 5 complaint types only |
| Final size | 53,135 rows |
| Train | 37,194 (70%) |
| Validation | 7,970 (15%) |
| Test | 7,971 (15%) |
| Split method | Stratified random, random_state=42 |
| Dataset SHA256 | *5d139a21edfb4d3bf9dc042a298a3976f2ffedec51aef9c4433e5120e888e900* |

Class distribution (top 5):

| Class | Count | Fraction |
|---|---|---|
| HEAT/HOT WATER | 17,998 | 33.87% |
| Illegal Parking | 16,680 | 31.39% |
| Noise - Residential | 10,606 | 19.96% |
| Blocked Driveway | 5,092 | 9.58% |
| Traffic Signal Condition | 2,759 | 5.19% |

## Evaluation

### Metrics

| Metric | Test Value |
|---|---|
| Accuracy | 0.6871 |
| Macro-F1 | 0.6188 |

### Comparison with Baselines (test set)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority class | 0.3387 | 0.1012 |
| Logistic Regression | 0.5018 | 0.3074 |
| Random Forest | 0.5701 | 0.4839 |
| Hist Gradient Boosting | 0.6378 | 0.5487 |
| MLP | 0.5175 | 0.3376 |
| XGBoost | 0.6145 | 0.4957 |
| LightGBM (untuned) | 0.6551 | 0.5749 |
| **LightGBM (Optuna-tuned)** | **0.6871** | **0.6188** |

**Improvement over untuned:** +3.2 points accuracy, +4.4 points macro-F1.

### Validation vs. Test

| Set | Macro-F1 |
|---|---|
| Validation | 0.6136 |
| Test | 0.6188 |

Val and test are within 0.5 points — no evidence of overfitting to validation.

### Artifacts (logged in MLflow)

- Confusion matrix (PNG) per run
- Classification report (precision/recall/F1 per class) per run
- Dataset identity JSON: CSV SHA256, row counts, class list, feature list, git SHA

## Limitations

1. **Random split, not temporal.** A temporal split would better reflect production.
2. **Small sample.** 100k rows out of >30M in the full dataset. Distribution shifts may not be captured.
3. **Top 5 classes only.** The other 153 classes are excluded, so the model cannot classify them.
4. **`zip_binned` collapses 216 ZIPs to 51 categories.** Fine-grained geography is lost.
5. **Latitude/longitude imputed with median.** ~1.4% of rows have missing coordinates.
6. **No temporal features.** Year, season-over-year trends not modeled.
7. **NYC-specific.** Does not transfer to other cities without retraining.

## Ethical Considerations

- **Do not use for policing or enforcement.** Complaint-type patterns reflect
  reporting behavior, not underlying incidence.
- **ZIP code correlates with race and income in NYC.** Even with ZIP excluded
  from final features, borough and lat/lon carry similar signals. Model output
  should not drive resource allocation without fairness audits.
- **Do not use for individual-level decisions.** This model classifies complaint
  types, not people.

## Reproducibility

```bash
uv sync --extra dev
uv run python scripts/compare_models.py
uv run python scripts/tune_best.py
```
All runs tracked in `mlflow.db` (SQLite). To view:

```bash
uv run mlflow server --backend-store-uri "sqlite:///C:/Users/Janak Mallik/UrbanMind-AI/mlflow.db" --host 127.0.0.1 --port 5000 --workers 1
```

## Citation

If referencing this model, cite:

​```text
Mallik, J. (2026). UrbanMind-AI: NYC 311 Complaint Classifier (v1.0).
GitHub: https://github.com/janakmallik/UrbanMind-AI
Tag: v0.2.0-week1
​```

## Changelog

| **Version** | **Date**   | **Notes**                                     |
| :---------- | :--------- | :-------------------------------------------- |
| 1.0         | 2026-10-13 | Initial release. LightGBM with Optuna tuning. |
