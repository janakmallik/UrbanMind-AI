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