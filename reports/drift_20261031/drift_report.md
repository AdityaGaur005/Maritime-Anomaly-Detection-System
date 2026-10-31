# Maritime AIS Anomaly Detection -- Drift Analysis Report

**Generated:** 2026-10-31T15:47:10+00:00 UTC  
**Reference:** 2017-2019 normal windows from incident-vessel population -- 760,204 windows from 115 vessels  
**Current (combined):** clean-2020 vessels -- 1,253,579 windows from 892 vessels  
**Current (fixed pop):** 2020 normal windows from same incident-vessel population -- 241,215 windows  
**Features analysed:** 55

> **Population note:** The reference is the steady-state behaviour of incident-prone vessels (tows, tankers, cargo ships) -- not all normal maritime traffic. Vessel types absent from this population (e.g. fishing, pleasure craft) will appear to drift even if their behaviour did not change. See methodology section.

> **In-sample bias note:** Reference model_scores are in-sample predictions (the model was trained on this data), so they are optimistically low. The combined-drift KS should be read as an **upper bound** on operational drift. The pure-temporal-drift comparison scores both sides in-sample and is unbiased.

---

## 1. Prediction Drift -- Combined (Year + Population Change)


*This is the operationally relevant number: how much did the model's score distribution shift on the actual 2020 traffic mix?*

| Metric | Reference (2017-2019) | Current (2020) | Shift |
|---|---|---|---|
| Mean score | 0.4299 | 0.4043 | -0.0256 |
| Median | 0.3938 | 0.3393 | -0.0545 |
| P90 | 0.7726 | 0.7705 | **-0.0021** (-0.3%) |
| P95 | 0.8351 | 0.8541 | +0.0190 |

**KS statistic:** 0.0682 &nbsp;|&nbsp; **PSI:** 0.0253 &nbsp;|&nbsp; **Severity:** stable

> *PSI < 0.10 = stable, 0.10-0.25 = moderate, >= 0.25 = significant.*


---

## 1b. Prediction Drift -- Pure Temporal (Fixed Population)


*Same incident-vessel population, 2020 vs 2017-2019. Isolates year change from population change. Both sides in-sample, so unbiased.*

| Metric | Reference (2017-2019) | Same-pop 2020 | Shift |
|---|---|---|---|
| Mean score | 0.4299 | 0.3700 | -0.0598 |
| P90 | 0.7726 | 0.7214 | **-0.0511** (-6.6%) |

**KS:** 0.1298 &nbsp;|&nbsp; **PSI:** 0.0901 &nbsp;|&nbsp; **Severity:** moderate

> Compare the KS here with the combined drift KS above. The gap quantifies the **population composition effect** (how much of the overall drift is explained by the different vessel mix rather than by the year changing).

---

## 2. Operational Threshold Impact

| Threshold | Purpose | Reference FPR | Current FPR | Inflation |
|---|---|---|---|---|
| 0.85 | Flag anomaly | 4.15% | 5.21% | **1.26x** |
| 0.77 | Baseline update gate | 89.75% pass | 89.97% pass | -- |

> *FPR inflation = how many more false alerts per day in 2020 vs training period.*

---

## 3. Year Stability Check -- AUC (Within Training Population)


> **Interpretation note:** Both the 2017-2019 and 2020 subsets used here are **in-training-set data** -- the XGBoost was trained on all 4 years of incident vessels. This delta is a year-over-year *stability* check, not a generalization estimate. For generalization, use the CV-pooled AUC = **0.853 +/- 0.029** from `kfold_cv_hybrid.py`.
| Period | Windows | Positives | AUC-ROC | Avg Precision |
|---|---|---|---|---|
| Reference 2017-2019 | 768,363 | 8,159 | 0.9053 | 0.2811 |
| Current 2020 | 243,465 | 2,250 | 0.9383 | 0.4437 |

**AUC delta = +0.0330** (improved)

---

## 4. Top 20 Drifted Features (by KS statistic)

| # | Feature | KS | PSI | Severity | Ref mean | Cur mean | Shift |
|---|---|---|---|---|---|---|---|
| 1 | `vessel_type_code` | 0.320 | 4.156 | significant | 352.3849 | 43.7200 | -308.6650 |
| 2 | `width_m` | 0.227 | 0.760 | significant | 10.8223 | 10.4667 | -0.3556 |
| 3 | `length_m` | 0.212 | 0.977 | significant | 48.9411 | 51.7501 | +2.8090 |
| 4 | `dist_from_vessel_centroid_km_mean` | 0.194 | 0.226 | moderate | 46.6544 | 31.1058 | -15.5486 |
| 5 | `dist_from_vessel_centroid_km_max` | 0.188 | 0.203 | moderate | 48.3936 | 32.4787 | -15.9149 |
| 6 | `lon_max` | 0.159 | 0.232 | moderate | -157.6088 | -157.7384 | -0.1296 |
| 7 | `lon_mean` | 0.150 | 0.206 | moderate | -157.6236 | -157.7497 | -0.1261 |
| 8 | `lon_min` | 0.150 | 0.184 | moderate | -157.6384 | -157.7610 | -0.1226 |
| 9 | `lat_min` | 0.147 | 0.189 | moderate | 21.1729 | 21.2064 | +0.0335 |
| 10 | `speed_zscore_vs_vessel` | 0.140 | 0.284 | significant | -0.0001 | -0.0006 | -0.0005 |
| 11 | `lat_mean` | 0.137 | 0.147 | moderate | 21.1830 | 21.2145 | +0.0315 |
| 12 | `delta_time_mean` | 0.132 | 0.097 | moderate | 189.4923 | 265.1341 | +75.6417 |
| 13 | `lat_max` | 0.128 | 0.172 | moderate | 21.1929 | 21.2224 | +0.0295 |
| 14 | `speed_over_ground_knots_max` | 0.115 | 0.090 | moderate | 4.1170 | 3.2360 | -0.8810 |
| 15 | `computed_speed_knots_max` | 0.115 | 0.082 | moderate | 3.9494 | 3.0237 | -0.9258 |
| 16 | `speed_over_ground_knots_mean` | 0.115 | 0.081 | moderate | 3.1248 | 2.3627 | -0.7621 |
| 17 | `acceleration_knots_per_sec_min` | 0.111 | 0.057 | moderate | -0.0111 | -0.0101 | +0.0010 |
| 18 | `acceleration_knots_per_sec_max` | 0.110 | 0.056 | moderate | 0.0108 | 0.0098 | -0.0010 |
| 19 | `acceleration_knots_per_sec_std` | 0.108 | 0.060 | moderate | 0.0041 | 0.0036 | -0.0005 |
| 20 | `trans_err_acceleration` | 0.108 | 0.071 | moderate | 0.0208 | 0.0277 | +0.0068 |

---

## 5. Population Composition Drift

- Unique vessels in reference: **115**
- Unique vessels in current: **892**
- Vessels in 2020 never seen in 2017-2019: **819**

**Top vessel-type shifts** (by absolute change):

| Vessel type code | Ref % | Cur % | Delta |
|---|---|---|---|
| 30 | 19.2% | 40.6% | +21.5% |
| 1025 | 20.8% | 0.0% | -20.8% |
| 31 | 31.4% | 19.1% | -12.3% |
| 90 | 2.8% | 12.1% | +9.3% |
| 1001 | 6.9% | 0.0% | -6.9% |
| 37 | 0.4% | 5.7% | +5.3% |
| 36 | 0.2% | 5.1% | +4.9% |
| 70 | 1.2% | 4.2% | +3.0% |

---

## 6. Suggested Trigger Thresholds for Phase 8 Retraining

These thresholds are calibrated from this analysis -- not arbitrary.

| Trigger | Threshold | Rationale |
|---|---|---|
| Primary: KS statistic | > **0.05** | ~67% of observed 2017->2020 shift (KS=0.068) |
| Secondary: P90 relative shift | > **15%** | Observed shift was -0.3%; 15% fires before full drift is reached |
| Guard: FPR inflation | > **2x** | Observed inflation was 1.26x |

> **Usage:** compute these metrics on a rolling 7-day window of live scores against the stored reference distribution. Fire retraining when any primary trigger is exceeded for two consecutive windows.
