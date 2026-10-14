
# Dataset Card: METR-LA Traffic Speed

## 1. Dataset Identity

| Field | Value |
|---|---|
| Dataset | METR-LA |
| Domain | Urban transportation and highway traffic forecasting |
| Original paper | Li et al., *Diffusion Convolutional Recurrent Neural Network* |
| Reference repository | https://github.com/liyaguang/DCRNN |
| Primary data file | `metr-la.h5` |
| Supporting files | `adj_mx.pkl`, `graph_sensor_locations.csv` |
| Local data directory | `<URBANINTEL_DATA_ROOT>/metr-la/` |
| Intended use | Research, experimentation, and traffic forecasting |
| Verification script | `scripts/verify_metr_la.py` |
| Download script | `scripts/download_metr_la.ps1` |

**License note:** The data was obtained from a mirror and supporting files from the DCRNN repository. The applicable dataset license and redistribution terms have not yet been independently verified. Check the original sources before redistributing the files.

## 2. Purpose in UrbanMind-AI

METR-LA provides historical traffic-speed measurements from highway loop detectors in the Los Angeles area.

Within UrbanMind-AI, this dataset will support the time-series and spatiotemporal learning component of the project.

The initial task is to forecast traffic speeds at multiple future horizons. Later experiments may compare classical machine learning, deep time-series models, and graph-based neural networks.

The planned development sequence is:

1. Inspect and document the data.
2. Investigate zero-valued readings and potential outages.
3. Define leakage-safe chronological splits and evaluation masks.
4. Establish simple forecasting baselines.
5. Train a classical ML model, such as LightGBM.
6. Compare selected deep learning models.
7. Explore graph-based forecasting using the provided adjacency matrix.

Published benchmark results may be used for context only after confirming that the data processing, splits, evaluation masks, and metrics are comparable.

## 3. Files and Dimensions

| File | Description | Verified dimensions |
|---|---|---|
| `metr-la.h5` | Traffic-speed observations | 34,272 timestamps × 207 sensors |
| `adj_mx.pkl` | Sensor IDs, sensor-to-index mapping, adjacency matrix | 207 × 207 matrix |
| `graph_sensor_locations.csv` | Sensor identifiers and coordinates | 207 rows × 4 columns |

All three files were downloaded successfully and passed the structural verification script.

The files are stored outside the Git repository under:

`C:\Users\Janak Mallik\ml-data\metr-la\`

Raw datasets should remain outside Git unless the project's data policy explicitly changes.

## 4. Data Format

### 4.1 `metr-la.h5`

The HDF5 file contains a pandas DataFrame under the key `df`.

Verified internal datasets:

- `axis0`: 207 sensor IDs.
- `axis1`: 34,272 timestamps represented as epoch nanoseconds.
- `block0_values`: a 34,272 × 207 matrix of `float64` values.

The values are conventionally interpreted as traffic speeds in miles per hour (mph). This unit should be confirmed against the original dataset documentation before being treated as authoritative.

Example loading code:

```python
import pandas as pd

df = pd.read_hdf(
    r"C:\Users\Janak Mallik\ml-data\metr-la\metr-la.h5",
    key="df",
)

print("Shape:", df.shape)
print("First timestamp:", df.index.min())
print("Last timestamp:", df.index.max())
print("Number of sensors:", df.shape[1])
```

Expected shape:

```text
(34272, 207)
```

### 4.2 `adj_mx.pkl`

The file contains a three-element structure:

1. `sensor_ids`: sensor identifiers in graph order.
2. `id_to_index_map`: mapping from sensor ID to matrix index.
3. `adjacency_matrix`: a 207 × 207 numerical matrix.

Verified properties:

- Matrix shape: `(207, 207)`.
- Data type after conversion: `float64`.
- All matrix values are finite.
- Nonzero entries: 1,722.
- Sensor identifiers align with the HDF5 sensor order.
- Mapping indices cover `0` through `206`.
- The mapping agrees with the adjacency matrix's row/column order.

The observed nonzero fraction is approximately 4.02% of all matrix positions.

The numerical range and sparsity alone do not establish the exact meaning of the edge weights. Their construction and interpretation should be confirmed from the original DCRNN implementation.

**Security note:** Pickle files can execute code when loaded. Load this file only from a trusted source.

### 4.3 `graph_sensor_locations.csv`

Verified columns:

| Column | Description |
|---|---|
| `index` | Dataset-provided index |
| `sensor_id` | Sensor identifier |
| `latitude` | Latitude |
| `longitude` | Longitude |

Verified properties:

- 207 rows.
- Sensor IDs align with HDF5 and adjacency ordering.
- Coordinates are finite and within valid latitude/longitude ranges.

Coordinate reference details and the precise meaning of the `index` field should be confirmed from the source documentation if required by downstream geospatial processing.

## 5. Verified Dataset Statistics

| Field | Verified value |
|---|---:|
| Sensors | 207 |
| Timesteps | 34,272 |
| Expected interval | 5 minutes |
| Time range | 2012-03-01 00:00:00 to 2012-06-27 23:55:00 |
| HDF5 matrix shape | 34,272 × 207 |
| HDF5 value type | `float64` |
| NaN or infinite traffic values | None observed |
| Minimum observed value | 0.0 |
| Maximum observed value | 70.0 |
| Mean observed value | Approximately 53.719 |
| Zero-valued cells | 8.11% |
| Timestamps where all sensors are zero | 2,148 |
| Adjacency matrix shape | 207 × 207 |
| Adjacency nonzero entries | 1,722 |

These statistics describe the downloaded files and the checks performed. They do not, by themselves, establish the semantic validity of every measurement.

## 6. Data Quality: Zero Values and Possible Outages

### 6.1 Observed issue

The HDF5 matrix contains no NaN or infinite values, but approximately 8.11% of its cells equal zero.

Additionally, 2,148 timestamps have zero values across all 207 sensors.

These patterns are suspicious and warrant investigation. However, structural verification alone cannot prove that every zero represents a missing measurement or sensor outage.

Possible explanations include:

- Missing or invalid readings encoded as zero.
- Sensor or data-collection outages.
- Other dataset-specific preprocessing or encoding conventions.

The correct interpretation must be investigated using the original dataset documentation and source implementation.

### 6.2 Long all-zero intervals

An earlier inspection identified a longest consecutive all-sensor-zero interval of 381 timesteps, approximately 31 hours and 45 minutes, beginning at 2012-03-26 08:25.

This is an observed pattern that warrants further investigation. It should not automatically be treated as a confirmed outage until the dataset's zero-value convention has been established.

### 6.3 Required investigation

Before training a forecasting model:
**Status: investigation pending.** Items below must be completed before the first forecasting baseline is trained.

1. Inspect the original DCRNN preprocessing and data-loading code.
2. Determine how zero-valued readings are interpreted in the reference implementation.
3. Compute zero fractions per sensor and over time.
4. Identify consecutive all-zero intervals and their durations.
5. Compare candidate validity masks without modifying the raw data.
6. Record the chosen interpretation and its evidence in the project documentation.

### 6.4 Modeling implications

Do not automatically replace zeros with NaN or impute them during data loading.

Keep the raw observations unchanged. If the investigation supports treating some zeros as invalid, define an explicit validity mask and document its rules.

For fair evaluation:

- Apply the same documented validity policy across comparable experiments.
- Report the number of valid targets used for each metric.
- Avoid allowing invalid targets to improve reported metrics artificially.
- Fit imputers, scalers, and other learned preprocessing only on training data.
- Preserve the chronological order of observations.

## 7. Chronological Splitting and Leakage Prevention

Traffic forecasting must respect time order. Randomly splitting individual timestamps can expose future conditions during training and create overly optimistic evaluation results.

The reference DCRNN setup uses chronological train, validation, and test partitions commonly described as 70%, 10%, and 20%.

For this dataset, the exact integer boundaries and treatment of forecasting windows must be defined explicitly in UrbanMind-AI rather than assumed from rounded percentages.

A proposed starting point is:

| Partition | Approximate proportion | Purpose |
|---|---:|---|
| Train | 70% | Fit model parameters and training-only preprocessing |
| Validation | 10% | Model selection and hyperparameter tuning |
| Test | 20% | Final evaluation after choices are fixed |

For a dataset of 34,272 timestamps, an implementation must document its exact boundaries and resulting sample counts. The final forecasting samples must not cross partition boundaries in a way that leaks target information.


| Partition | Count | Index range | Time range |
|---|---:|---|---|
| Train | 23,990 | 0 .. 23,989 | 2012-03-01 00:00 → 2012-05-24 19:55 |
| Validation | 3,427 | 23,990 .. 27,416 | 2012-05-24 20:00 → 2012-06-12 23:55 |
| Test | 6,856 | 27,417 .. 34,271 | 2012-06-13 00:00 → 2012-06-27 23:55 |


The test set must not be used for model selection, preprocessing fitting, or threshold tuning.

## 8. Forecasting Task Definition

The initial task is multivariate traffic-speed forecasting.

Given a historical input window, predict traffic speeds at future timestamps for all 207 sensors.

A typical starting configuration is:

| Parameter | Initial setting |
|---|---|
| Input window | 12 timesteps |
| Historical duration | 60 minutes |
| Sampling interval | 5 minutes |
| 15-minute horizon | 3 steps ahead |
| 30-minute horizon | 6 steps ahead |
| 60-minute horizon | 12 steps ahead |

The prediction horizons must be implemented consistently and evaluated against the corresponding future timestamps.

The initial experiments should use MAE as a primary metric, with additional metrics such as RMSE reported where useful. The treatment of invalid targets must be explicit.

## 9. Baseline and Model Comparison Plan

Build baselines before introducing complex architectures.

**Success threshold for Week 2:** beat persistence by ≥15% relative MAE at
the 60-minute horizon. Published GNN results (MAE ≈ 3.5 mph) are the
long-term reference, not Week 2 targets.

### Stage A: Simple baselines

- Persistence: predict future speed using the latest observed speed.
- Historical or seasonal baseline: evaluate only after defining its training-only fitting procedure and seasonal assumptions.

### Stage B: Classical machine learning

- LightGBM using carefully constructed lag features.
- Optional calendar features, provided their availability at prediction time is respected.

### Stage C: Deep learning

- LSTM or another recurrent model.
- Temporal convolutional network (TCN).
- Graph-based models, potentially including DCRNN or Graph WaveNet, after verifying graph semantics and implementation requirements.

### Evaluation requirements

Every experiment should document:

- Exact data split boundaries.
- Input window and prediction horizons.
- Zero-value and invalid-target policy.
- Preprocessing fitted on training data only.
- MAE and other selected metrics.
- Number of valid targets.
- Random seed, model configuration, and software environment.

Published model results are reference points, not guaranteed targets. Direct comparisons are meaningful only when the benchmark protocol and evaluation conventions match.

### Success threshold for Week 2

**Beat persistence by ≥15% relative MAE at the 60-minute horizon.**

Published GNN results (MAE ≈ 3.5 mph at 60 min) are the long-term reference
target, not the Week 2 goal. Week 2 focuses on establishing a reliable
classical-ML forecasting pipeline. Deep models and graph models come later.

If the Week 2 model does not beat persistence by this margin, it indicates
either (a) a data or split problem, or (b) insufficient feature engineering.
Investigate before proceeding to Week 3.

## 10. Known Limitations

- Coverage is limited to the sensors represented in METR-LA.
- The data covers a historical period in 2012.
- Zero-valued readings require semantic investigation.
- Sensor failures and outages may affect evaluation.
- The graph's edge-weight semantics need source confirmation.
- Traffic forecasting performance on this dataset does not establish performance on other cities or transportation networks.
- Reproducing published results may require matching the original split, preprocessing, model configuration, and evaluation code.

## 11. Reproducibility

Run these commands from the UrbanMind-AI project root in PowerShell.

### Download the dataset

```powershell
.\scripts\download_metr_la.ps1
```

### Verify structural consistency

```powershell
uv run python scripts/verify_metr_la.py
```

Expected final line:

```text
All checks passed.
```

The verification script checks file presence, matrix dimensions, finite values, timestamp cadence, sensor ID consistency, adjacency structure, location coordinates, and sensor-to-index mapping.

Passing verification does not establish that every zero is a valid measurement, that all source metadata is correct, or that the dataset's license permits redistribution.

The project declares `h5py` as a dependency so the verifier can run in the project environment.

### View experiments

```powershell
uv run mlflow server --backend-store-uri "sqlite:///C:/Users/Janak Mallik/UrbanMind-AI/mlflow.db" --host 127.0.0.1 --port 5000 --workers 1
```

## 12. Citation

Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018).

*Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.*

International Conference on Learning Representations (ICLR).

Paper: https://arxiv.org/abs/1707.01926

Reference implementation: https://github.com/liyaguang/DCRNN

## 13. Changelog

| Date | Notes |
|---|---|
| 2026-10-14 | Initial dataset card based on downloaded files and successful structural verification. |
