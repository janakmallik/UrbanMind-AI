# Experiments Journal

Short entries for every experiment. Structure: hypothesis → setup → result → takeaway.

---

## 2026-10-14 — METR-LA naive forecasting baselines

**Hypothesis:** Persistence will be strong at 15 min and decay with horizon.
The sensor-hour historical mean will be more stable across horizons because
it captures the rush-hour rhythm that persistence cannot.

**Setup:**
- Dataset: METR-LA, 207 sensors, 5-min sampling
- Split: chronological 70/10/20
- Windows: 12 input steps (60 min), horizons 3 / 6 / 12 (15 / 30 / 60 min)
- Train windows: 23,967 | Validation windows: 3,404 | Test windows: 6,832
- Metrics: masked MAE / RMSE / MAPE on valid targets only
  (zero-valued targets treated as suspected missing and excluded)
- Random seed: N/A (deterministic baselines)

**Result:**

| Baseline | MAE h=3 | MAE h=6 | MAE h=12 | RMSE h=3 | RMSE h=6 | RMSE h=12 | MAPE h=3 | MAPE h=6 | MAPE h=12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 4.0207 | 5.1001 | 6.8031 | 8.7028 | 11.1450 | 14.2231 | 9.41% | 12.23% | 16.74% |
| Sensor-hour mean | 7.4968 | 7.4959 | 7.4943 | 9.9949 | 9.9944 | 9.9932 | 19.34% | 19.34% | 19.33% |

- *MAE is measured in mph; lower is better.*
- *RMSE is measured in mph; lower is better. Larger errors are penalized more heavily.*
- *MAPE is measured in %; lower is better. It can be unstable when actual speeds are near zero.*

**Takeaway:**

Persistence strongly outperforms the sensor-hour historical mean at every
horizon — including the longest one. This is the opposite of the initial
hypothesis (that the historical mean would catch up at longer horizons).

Two implications:

1. **Persistence is a genuinely strong baseline for METR-LA.** This matches
   the original DCRNN paper, which reported persistence MAE ≈ 4.5 mph at
   15 min. Our h=3 MAE of 4.02 mph is in the same range, consistent with a
   slightly different split and mask policy.

2. **Sensor-hour means ignore more information than they add.** A single
   mean per (sensor, hour) discards the current-day context — the current
   speed is a much better predictor than the historical average. This
   suggests the mean-based baseline is not competitive with persistence and
   should not be the reference for ML models. **Persistence is the floor.**

**Reference (published):**

- Li et al. 2018 (DCRNN) — persistence MAE @15 min: ≈ 4.5 mph
- Li et al. 2018 — DCRNN MAE @15 min: 2.77 mph (target after ML/DL work)

**Success threshold for Week 2:**
Beat persistence at h=12 by ≥15% relative MAE. Persistence h=12 MAE is
6.8031 → target ≤ 5.783 mph at 60-min horizon for the Week 2 ML model.


---

## 2026-10-15 — METR-LA LightGBM forecaster (per-horizon)

**Hypothesis:** A tree-based model with 12 lag features per sensor will beat
persistence by capturing short-term temporal patterns.

**Setup:**
- Same windows, split, and mask as baselines
- Features (20): lags 1–12, lag_h, sensor_mean, sensor_std, hour, dow,
  is_weekend, sensor_idx, horizon
- One LightGBM per horizon (n_estimators=500, lr=0.05, num_leaves=63)
- Training subsampled to ~2M rows
- Persistence MAE computed on the same test targets (in-run, not hardcoded)

**Result:**

| Model | MAE h=3 | MAE h=6 | MAE h=12 |
|---|---:|---:|---:|
| Persistence | 4.0207 | 5.1001 | 6.8031 |
| Sensor-hour mean | 7.4968 | 7.4959 | 7.4943 |
| **LightGBM** | **3.1095** | **3.6992** | **4.3356** |

| Horizon | Improvement vs persistence |
|---|---:|
| h=3 | +22.7% |
| h=6 | +27.5% |
| h=12 | **+36.3%** |

**Takeaway:** LightGBM with lag features substantially outperforms persistence
at every horizon and easily clears the Week 2 success threshold (≥15% at h=12).
This confirms that short-term temporal context (the last hour of readings) is
highly informative for traffic forecasting.

The relative improvement *grows* with horizon — persistence degrades much
faster than the learned model. This suggests the model is learning patterns
that persist beyond the immediate reading.

**Reference:** Published DCRNN MAE at h=12 is 3.60 mph. LightGBM at 4.34 is
within 20% of that without any graph structure or deep learning, which is a
strong tabular baseline. The next stage (LSTM/TCN) will test whether sequence
models improve further.

---

