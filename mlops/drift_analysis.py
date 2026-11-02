"""
Drift analysis: reference (2017-2019 normal windows) vs current (clean 2020).

Five analyses:
  1. Feature-level covariate drift  — KS statistic + PSI per feature
  2. Prediction drift               — model_score distribution shift
  3. Population composition drift   — vessel type, geography, new vessels
  4. Operational threshold impact   — FPR inflation at 0.77 and 0.85
  5. Discriminative degradation     — AUC on incident windows: 2017-19 vs 2020

Why this framing
----------------
The deep models (Transformer-VAE, LSTM-AE) and the calibration thresholds were
built on 2017-2019 clean behaviour. 2020 is genuinely held-out from the system's
operational perspective. Measuring how much the score distribution shifted between
these two periods quantifies the empirical case for the self-learning loop.

Reference set choice and known limitations
------------------------------------------
We use normal windows (label==0, years 2017-2019) from
hybrid_features_incident_vessels_with_scores.parquet rather than a separate
"clean 2017-2019" table.  This is the exact distribution the XGBoost was
calibrated against at training time — the most defensible reference for drift.

If either parquet lacks a 'model_score' column, this script scores it on-the-fly
using xgboost_hybrid_final.json.  Both reference and current distributions are
therefore scored by the same model, keeping the comparison fair.

Four caveats to state explicitly in the thesis methodology section:

1. COMBINED DRIFT — The primary comparison (reference vs clean-2020) mixes two
   things: (a) year change 2019->2020 and (b) population change from incident-prone
   vessels to clean vessels.  Both are operationally real, but a reviewer may ask
   to separate them.  The script therefore also runs a "pure temporal drift"
   comparison: 2020 normal windows from the *same* incident-vessel population vs
   the 2017-2019 reference.  This isolates year change on a fixed population.

2. IN-SAMPLE vs OUT-OF-SAMPLE BIAS — The reference set (2017-2019 incident-vessel
   normal windows) was part of XGBoost's training data; its model_score is therefore
   an in-sample prediction, optimistically low.  The clean-2020 set is out-of-sample,
   scored honestly.  The combined-drift KS is thus slightly inflated by the
   in-sample/out-of-sample gap, and should be read as an upper bound on operational
   drift.  The pure-temporal-drift comparison scores both sides in-sample, so it is
   unbiased and is the more honest year-over-year signal.

3. AUC IS A YEAR STABILITY CHECK, NOT GENERALIZATION — analyse_auc_degradation
   compares 2017-19 vs 2020 incident windows, but the model was trained on all
   4 years of incident vessels.  Both periods are in-training-set data.  Do NOT
   cite the AUC delta as generalization evidence.  Use the CV-pooled AUC (0.853
   from kfold_cv_hybrid.py) for generalization claims; use this delta only to
   show year-over-year stability within the known population.

4. BIASED REFERENCE POPULATION — The reference is not "all normal maritime
   behaviour" — it is the steady-state behaviour of vessels that occasionally
   become incidents (tows, tankers, cargo ships).  Vessel types absent from that
   population (fishing, pleasure craft) will appear to drift even if their
   behaviour is constant.  State this framing explicitly so reviewers understand
   what the reference represents.

Column assumptions
------------------
  Meta  : MMSI, label (0/1), incident_type, year ('2017'..'2020'),
           isoforest_score, model_score (may be absent — script will score)
  Features (55): see FEATURE_NAMES below — exactly matches xgboost_hybrid_final.json

Usage
-----
    # Default paths (run from repo root)
    python -m mlops.drift_analysis

    # Custom paths
    python -m mlops.drift_analysis \\
        --incident-parquet hybrid/hybrid_features_incident_vessels_with_scores.parquet \\
        --clean-parquet    hybrid/hybrid_features_clean_2020_with_scores.parquet \\
        --output           reports/drift_2020/

Outputs (written to --output directory)
---------------------------------------
    drift.json              machine-readable; feeds Phase 8 retraining trigger
    drift_report.md         thesis-ready markdown
    figures/score_cdf.png   CDF of model_score: reference vs current
    figures/feature_ks.png  Bar chart of top-20 drifted features
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

BASE_DIR = Path(__file__).resolve().parent.parent
HYBRID_DIR = BASE_DIR / "hybrid"

# ── default data paths ────────────────────────────────────────────────────────
DEFAULT_INCIDENT_PARQUET = HYBRID_DIR / "hybrid_features_incident_vessels_with_scores.parquet"
DEFAULT_CLEAN_PARQUET = HYBRID_DIR / "hybrid_features_clean_2020_with_scores.parquet"

# ── production model path (for on-the-fly scoring) ────────────────────────────
XGB_MODEL_PATH = HYBRID_DIR / "xgboost_hybrid_final.json"

# ── thresholds (must match mlops/main.py) ─────────────────────────────────────
ANOMALY_THRESHOLD = 0.85   # flags a window as anomalous
UPDATE_THRESHOLD = 0.77    # gate for baseline self-update

REFERENCE_YEARS = {"2017", "2018", "2019"}

# ── 55 features used by the trained XGBoost model ────────────────────────────
FEATURE_NAMES = [
    "baseline_is_fallback", "vessel_type_code", "length_m", "width_m",
    "lat_mean", "lat_std", "lat_min", "lat_max",
    "lon_mean", "lon_std", "lon_min", "lon_max",
    "speed_over_ground_knots_mean", "speed_over_ground_knots_std",
    "speed_over_ground_knots_min", "speed_over_ground_knots_max",
    "course_over_ground_deg_mean", "course_over_ground_deg_std",
    "course_over_ground_deg_min", "course_over_ground_deg_max",
    "computed_speed_knots_mean", "computed_speed_knots_std",
    "computed_speed_knots_min", "computed_speed_knots_max",
    "acceleration_knots_per_sec_mean", "acceleration_knots_per_sec_std",
    "acceleration_knots_per_sec_min", "acceleration_knots_per_sec_max",
    "heading_change_deg_mean", "heading_change_deg_std",
    "heading_change_deg_min", "heading_change_deg_max",
    "delta_time_mean", "delta_time_max", "delta_time_std",
    "speed_zscore_vs_vessel", "heading_zscore_vs_vessel",
    "dist_from_vessel_centroid_km_mean", "dist_from_vessel_centroid_km_max",
    "trans_err_lat", "trans_err_lon", "trans_err_speed", "trans_err_course",
    "trans_err_computed_speed", "trans_err_acceleration",
    "trans_err_heading_change", "trans_err_agg",
    "lstm_err_lat", "lstm_err_lon", "lstm_err_speed", "lstm_err_course",
    "lstm_err_computed_speed", "lstm_err_acceleration",
    "lstm_err_heading_change", "lstm_err_agg",
]

META_COLS = {
    "MMSI", "label", "incident_type", "year",
    "isoforest_score", "model_score",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_col(df: pd.DataFrame, col: str, parquet: str) -> None:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in {parquet}.\n"
            f"Available: {sorted(df.columns.tolist())}"
        )


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index.
    <0.10 -> stable, 0.10-0.25 -> moderate change, >=0.25 -> significant change.
    """
    if len(expected) < 10 or len(actual) < 10:
        return float("nan")
    edges = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_cnt, _ = np.histogram(expected, edges)
    a_cnt, _ = np.histogram(actual, edges)
    e_pct = np.maximum(e_cnt / len(expected), 1e-6)
    a_pct = np.maximum(a_cnt / len(actual), 1e-6)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _severity(ks: float, psi_val: float) -> str:
    if psi_val >= 0.25 or ks >= 0.20:
        return "significant"
    if psi_val >= 0.10 or ks >= 0.10:
        return "moderate"
    return "stable"


def _cdf_at(data: np.ndarray, xs: np.ndarray) -> np.ndarray:
    s = np.sort(data)
    return np.searchsorted(s, xs, side="right") / len(s)


# ── on-the-fly XGBoost scoring ────────────────────────────────────────────────

def _score_with_xgb(df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Score a DataFrame with the production XGBoost model."""
    import xgboost as xgb
    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"XGBoost model not found at {XGB_MODEL_PATH}. "
            "Run hybrid/hybrid_final.py first."
        )
    booster = xgb.Booster()
    booster.load_model(str(XGB_MODEL_PATH))

    model_features = list(booster.feature_names) if booster.feature_names else []
    if model_features and model_features != feature_cols:
        mismatch = set(model_features) ^ set(feature_cols)
        raise ValueError(
            f"Feature list mismatch between parquet and model.\n"
            f"  model expects {len(model_features)} features\n"
            f"  parquet provides {len(feature_cols)} features\n"
            f"  symmetric difference (first 10): {sorted(mismatch)[:10]}"
        )

    dmat = xgb.DMatrix(df[feature_cols], feature_names=feature_cols)
    return booster.predict(dmat)


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(incident_path: Path, clean_path: Path):
    print(f"Loading incident-vessel parquet: {incident_path} ...")
    inc = pd.read_parquet(incident_path)
    for col in ("MMSI", "label", "year"):
        _require_col(inc, col, str(incident_path))
    inc["year"] = inc["year"].astype(str)
    inc["label"] = inc["label"].astype(int)
    print(f"  {len(inc):,} windows -- {inc['MMSI'].nunique():,} vessels -- "
          f"{int(inc['label'].sum())} positives")

    print(f"Loading clean-2020 parquet: {clean_path} ...")
    cur = pd.read_parquet(clean_path)
    _require_col(cur, "MMSI", str(clean_path))
    print(f"  {len(cur):,} windows -- {cur['MMSI'].nunique():,} vessels")

    available_in_inc = set(inc.columns) - META_COLS
    available_in_cur = set(cur.columns) - META_COLS
    feature_cols = [f for f in FEATURE_NAMES
                    if f in available_in_inc and f in available_in_cur]
    missing = set(FEATURE_NAMES) - set(feature_cols)
    if missing:
        print(f"  !!  {len(missing)} feature(s) absent from one parquet -- "
              f"skipped in covariate drift: "
              f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    if len(feature_cols) != len(FEATURE_NAMES):
        raise ValueError(
            f"Only {len(feature_cols)}/{len(FEATURE_NAMES)} model features are "
            f"available across both parquets. Cannot score with "
            f"{XGB_MODEL_PATH.name}."
        )

    if "model_score" not in inc.columns:
        print(f"\n  incident parquet has no model_score -- scoring "
              f"{len(inc):,} windows with {XGB_MODEL_PATH.name} ...")
        inc["model_score"] = _score_with_xgb(inc, feature_cols)
        print(f"  incident scored: mean={inc['model_score'].mean():.4f}  "
              f"P90={np.percentile(inc['model_score'], 90):.4f}  "
              f"P95={np.percentile(inc['model_score'], 95):.4f}  "
              f"max={inc['model_score'].max():.4f}")
    else:
        print("  incident parquet already has model_score -- using existing")

    if "model_score" not in cur.columns:
        print(f"  clean parquet has no model_score -- scoring "
              f"{len(cur):,} windows with {XGB_MODEL_PATH.name} ...")
        cur["model_score"] = _score_with_xgb(cur, feature_cols)
        print(f"  clean scored:    mean={cur['model_score'].mean():.4f}  "
              f"P90={np.percentile(cur['model_score'], 90):.4f}  "
              f"P95={np.percentile(cur['model_score'], 95):.4f}  "
              f"max={cur['model_score'].max():.4f}")
    else:
        print("  clean parquet already has model_score -- using existing")

    return inc, cur, feature_cols


def build_reference_current(inc: pd.DataFrame, cur: pd.DataFrame):
    ref = inc[(inc["label"] == 0) & (inc["year"].isin(REFERENCE_YEARS))].copy()
    if len(ref) == 0:
        raise ValueError(
            "Reference set is empty. Check that the incident parquet has "
            "label==0 rows with year in {2017, 2018, 2019}."
        )

    # Same population as reference but year 2020 — used for pure temporal drift.
    # This isolates the year-change signal by holding the vessel population fixed.
    same_pop_2020 = inc[(inc["label"] == 0) & (inc["year"] == "2020")].copy()

    print(f"\nReference        : {len(ref):,} normal windows from "
          f"{ref['MMSI'].nunique():,} vessels "
          f"({', '.join(sorted(ref['year'].unique()))})")
    print(f"Same-pop 2020    : {len(same_pop_2020):,} normal windows from "
          f"{same_pop_2020['MMSI'].nunique():,} vessels (pure temporal drift)")
    print(f"Current (clean)  : {len(cur):,} clean-2020 windows from "
          f"{cur['MMSI'].nunique():,} vessels")
    return ref, same_pop_2020, cur


# ── analysis 1 — feature covariate drift ─────────────────────────────────────

def analyse_feature_drift(ref: pd.DataFrame, cur: pd.DataFrame,
                           feature_cols: list) -> list:
    rows = []
    for f in feature_cols:
        a = ref[f].dropna().values
        b = cur[f].dropna().values
        if len(a) < 50 or len(b) < 50:
            continue
        ks, _ = ks_2samp(a, b)
        psi_val = psi(a, b)
        rows.append({
            "feature": f,
            "ks_stat": round(float(ks), 6),
            "psi": round(float(psi_val), 6),
            "severity": _severity(ks, psi_val),
            "ref_mean": round(float(a.mean()), 6),
            "cur_mean": round(float(b.mean()), 6),
            "ref_std": round(float(a.std()), 6),
            "cur_std": round(float(b.std()), 6),
            "mean_shift": round(float(b.mean() - a.mean()), 6),
        })
    return sorted(rows, key=lambda r: -r["ks_stat"])


# ── analysis 2 — prediction drift ────────────────────────────────────────────

def analyse_prediction_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    a = ref["model_score"].dropna().values
    b = cur["model_score"].dropna().values
    ks, ks_p = ks_2samp(a, b)
    psi_val = psi(a, b)

    def stats(x):
        return {
            "mean": round(float(x.mean()), 6),
            "median": round(float(np.median(x)), 6),
            "p90": round(float(np.percentile(x, 90)), 6),
            "p95": round(float(np.percentile(x, 95)), 6),
            "p99": round(float(np.percentile(x, 99)), 6),
        }

    ref_p90 = np.percentile(a, 90)
    cur_p90 = np.percentile(b, 90)

    return {
        "ks_stat": round(float(ks), 6),
        "ks_pvalue": float(ks_p),
        "psi": round(float(psi_val), 6),
        "severity": _severity(ks, psi_val),
        "reference": stats(a),
        "current": stats(b),
        "p90_absolute_shift": round(float(cur_p90 - ref_p90), 6),
        "p90_relative_shift": round(float((cur_p90 - ref_p90) / max(ref_p90, 1e-9)), 6),
    }


# ── analysis 3 — population composition drift ─────────────────────────────────

def analyse_population_drift(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    result = {
        "n_unique_vessels_reference": int(ref["MMSI"].nunique()),
        "n_unique_vessels_current": int(cur["MMSI"].nunique()),
        "n_new_vessels_in_current": int(
            len(set(cur["MMSI"].unique()) - set(ref["MMSI"].unique()))
        ),
    }

    if "vessel_type_code" in ref.columns and "vessel_type_code" in cur.columns:
        ref_vt = ref["vessel_type_code"].value_counts(normalize=True)
        cur_vt = cur["vessel_type_code"].value_counts(normalize=True)
        all_types = sorted(set(ref_vt.index) | set(cur_vt.index))
        type_table = [
            {
                "vessel_type_code": int(t),
                "ref_pct": round(float(ref_vt.get(t, 0.0)) * 100, 2),
                "cur_pct": round(float(cur_vt.get(t, 0.0)) * 100, 2),
                "delta_pct": round(float(cur_vt.get(t, 0.0) - ref_vt.get(t, 0.0)) * 100, 2),
            }
            for t in all_types
        ]
        result["vessel_type_shift"] = sorted(
            type_table, key=lambda r: -abs(r["delta_pct"])
        )
    else:
        result["vessel_type_shift"] = []

    for label, df in [("reference", ref), ("current", cur)]:
        if "lat_mean" in df.columns and "lon_mean" in df.columns:
            result[f"{label}_geo_bbox"] = {
                "lat_min": round(float(df["lat_mean"].min()), 4),
                "lat_max": round(float(df["lat_mean"].max()), 4),
                "lon_min": round(float(df["lon_mean"].min()), 4),
                "lon_max": round(float(df["lon_mean"].max()), 4),
            }

    return result


# ── analysis 4 — operational threshold impact ─────────────────────────────────

def analyse_threshold_impact(ref: pd.DataFrame, cur: pd.DataFrame) -> dict:
    a = ref["model_score"].dropna().values
    b = cur["model_score"].dropna().values

    def fpr(scores, thr):
        return float((scores >= thr).mean())

    ref_fpr_anomaly = fpr(a, ANOMALY_THRESHOLD)
    cur_fpr_anomaly = fpr(b, ANOMALY_THRESHOLD)

    ref_update_pass = float((a < UPDATE_THRESHOLD).mean())
    cur_update_pass = float((b < UPDATE_THRESHOLD).mean())

    return {
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "update_threshold": UPDATE_THRESHOLD,
        "false_positive_rate": {
            "reference": round(ref_fpr_anomaly, 6),
            "current": round(cur_fpr_anomaly, 6),
            "inflation_factor": round(cur_fpr_anomaly / max(ref_fpr_anomaly, 1e-9), 4),
        },
        "baseline_update_pass_rate": {
            "reference": round(ref_update_pass, 6),
            "current": round(cur_update_pass, 6),
        },
    }


# ── analysis 5 — AUC year stability check ─────────────────────────────────────

def analyse_auc_degradation(inc: pd.DataFrame) -> dict:
    """Year-over-year AUC stability check on the incident-vessel population.

    IMPORTANT -- this is NOT a generalization test.  The final XGBoost was
    trained on all 4 years of incident vessels, so both the 2017-19 and 2020
    subsets are in-training-set data.  Interpret the delta as:

        "Did 2020 incident patterns become harder or easier to separate from
         2020 normal patterns, compared to 2017-2019?"

    For generalization claims, use the CV-pooled AUC from kfold_cv_hybrid.py
    (0.853 +/- 0.029) -- that is the honest held-out estimate.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    def auc_for(df: pd.DataFrame, label: str) -> dict:
        sub = df.dropna(subset=["model_score", "label"])
        if sub["label"].nunique() < 2:
            print(f"  !!  '{label}' has only one class -- skipping AUC")
            return None
        auc = float(roc_auc_score(sub["label"], sub["model_score"]))
        ap = float(average_precision_score(sub["label"], sub["model_score"]))
        return {
            "n_windows": int(len(sub)),
            "n_positives": int(sub["label"].sum()),
            "auc_roc": round(auc, 6),
            "average_precision": round(ap, 6),
        }

    ref_df = inc[inc["year"].isin(REFERENCE_YEARS)]
    cur_df = inc[inc["year"] == "2020"]

    ref_auc = auc_for(ref_df, "reference 2017-2019")
    cur_auc = auc_for(cur_df, "current 2020")

    delta = None
    if ref_auc and cur_auc:
        delta = round(cur_auc["auc_roc"] - ref_auc["auc_roc"], 6)

    per_type = []
    if "incident_type" in inc.columns and cur_auc:
        # FIXED: Group only the positives, then combine with all negatives
        # to calculate a valid AUC (needs both classes).
        neg_df = cur_df[cur_df["label"] == 0]
        pos_df = cur_df[cur_df["label"] == 1]
        
        for itype in pos_df["incident_type"].dropna().unique():
            pos_subset = pos_df[pos_df["incident_type"] == itype]
            if len(pos_subset) < 20:
                continue
            combined = pd.concat([pos_subset, neg_df])
            try:
                auc = roc_auc_score(combined["label"], combined["model_score"])
                per_type.append({
                    "incident_type": str(itype),
                    "n_windows": int(len(pos_subset)),
                    "auc_roc": round(float(auc), 6),
                })
            except Exception:
                pass
        per_type.sort(key=lambda r: r["auc_roc"])

    return {
        "reference_2017_2019": ref_auc,
        "current_2020": cur_auc,
        "auc_delta": delta,
        "per_incident_type_2020": per_type,
    }


# ── output: JSON ──────────────────────────────────────────────────────────────

def write_json(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"  ✓ {path}")


# ── output: markdown report ───────────────────────────────────────────────────

def write_markdown(path: Path, r: dict) -> None:
    p = r["prediction"]
    pt = r.get("pure_temporal_drift")
    t = r["thresholds"]
    a = r["auc"]
    pop = r["population"]

    lines = [
        "# Maritime AIS Anomaly Detection -- Drift Analysis Report",
        f"\n**Generated:** {r['generated_at']} UTC  ",
        f"**Reference:** 2017-2019 normal windows from incident-vessel population -- "
        f"{r['data']['n_reference']:,} windows from {r['data']['n_reference_vessels']:,} vessels  ",
        f"**Current (combined):** clean-2020 vessels -- {r['data']['n_current']:,} windows from "
        f"{r['data']['n_current_vessels']:,} vessels  ",
        f"**Current (fixed pop):** 2020 normal windows from same incident-vessel population -- "
        f"{r['data']['n_same_pop_2020']:,} windows  ",
        f"**Features analysed:** {r['data']['n_features']}",
        "\n> **Population note:** The reference is the steady-state behaviour of "
        "incident-prone vessels (tows, tankers, cargo ships) -- not all normal maritime "
        "traffic. Vessel types absent from this population (e.g. fishing, pleasure craft) "
        "will appear to drift even if their behaviour did not change. See methodology section.",
        "\n> **In-sample bias note:** Reference model_scores are in-sample predictions "
        "(the model was trained on this data), so they are optimistically low. "
        "The combined-drift KS should be read as an **upper bound** on operational drift. "
        "The pure-temporal-drift comparison scores both sides in-sample and is unbiased.",
        "\n---\n",

        "## 1. Prediction Drift -- Combined (Year + Population Change)\n",
        "\n*This is the operationally relevant number: how much did the model's score "
        "distribution shift on the actual 2020 traffic mix?*\n",
        "| Metric | Reference (2017-2019) | Current (2020) | Shift |",
        "|---|---|---|---|",
        f"| Mean score | {p['reference']['mean']:.4f} | {p['current']['mean']:.4f} | "
        f"{p['current']['mean'] - p['reference']['mean']:+.4f} |",
        f"| Median | {p['reference']['median']:.4f} | {p['current']['median']:.4f} | "
        f"{p['current']['median'] - p['reference']['median']:+.4f} |",
        f"| P90 | {p['reference']['p90']:.4f} | {p['current']['p90']:.4f} | "
        f"**{p['p90_absolute_shift']:+.4f}** ({p['p90_relative_shift']:+.1%}) |",
        f"| P95 | {p['reference']['p95']:.4f} | {p['current']['p95']:.4f} | "
        f"{p['current']['p95'] - p['reference']['p95']:+.4f} |",
        f"\n**KS statistic:** {p['ks_stat']:.4f} &nbsp;|&nbsp; "
        f"**PSI:** {p['psi']:.4f} &nbsp;|&nbsp; "
        f"**Severity:** {p['severity']}",
        "\n> *PSI < 0.10 = stable, 0.10-0.25 = moderate, >= 0.25 = significant.*\n",
    ]

    lines += [
        "\n---\n",
        "## 1b. Prediction Drift -- Pure Temporal (Fixed Population)\n",
        "\n*Same incident-vessel population, 2020 vs 2017-2019. "
        "Isolates year change from population change. Both sides in-sample, so unbiased.*\n",
    ]
    if pt:
        lines += [
            "| Metric | Reference (2017-2019) | Same-pop 2020 | Shift |",
            "|---|---|---|---|",
            f"| Mean score | {pt['reference']['mean']:.4f} | {pt['current']['mean']:.4f} | "
            f"{pt['current']['mean'] - pt['reference']['mean']:+.4f} |",
            f"| P90 | {pt['reference']['p90']:.4f} | {pt['current']['p90']:.4f} | "
            f"**{pt['p90_absolute_shift']:+.4f}** ({pt['p90_relative_shift']:+.1%}) |",
            f"\n**KS:** {pt['ks_stat']:.4f} &nbsp;|&nbsp; "
            f"**PSI:** {pt['psi']:.4f} &nbsp;|&nbsp; **Severity:** {pt['severity']}",
            "\n> Compare the KS here with the combined drift KS above. "
            "The combined drift KS is lower because the 2020 population shift "
            "toward lower-scoring vessel types (fishing) masks the temporal drift.",
        ]
    else:
        lines.append("\n*Not computed -- insufficient same-population 2020 windows.*\n")

    lines += [
        "\n---\n",
        "## 2. Operational Threshold Impact\n",
        "| Threshold | Purpose | Reference FPR | Current FPR | Inflation |",
        "|---|---|---|---|---|",
        f"| {t['anomaly_threshold']} | Flag anomaly | "
        f"{t['false_positive_rate']['reference']:.2%} | "
        f"{t['false_positive_rate']['current']:.2%} | "
        f"**{t['false_positive_rate']['inflation_factor']:.2f}x** |",
        f"| {t['update_threshold']} | Baseline update gate | "
        f"{t['baseline_update_pass_rate']['reference']:.2%} pass | "
        f"{t['baseline_update_pass_rate']['current']:.2%} pass | -- |",
        "\n> *FPR inflation = how many more false alerts per day in 2020 vs training period.*\n",

        "---\n",
        "## 3. Year Stability Check -- AUC (Within Training Population)\n",
        "\n> **Interpretation note:** Both the 2017-2019 and 2020 subsets used here "
        "are **in-training-set data** -- the XGBoost was trained on all 4 years of incident "
        "vessels. This delta is a year-over-year *stability* check, not a generalization "
        "estimate. For generalization, use the CV-pooled AUC = **0.853 +/- 0.029** from "
        "`kfold_cv_hybrid.py`.",
    ]

    if a["reference_2017_2019"] and a["current_2020"]:
        ref_a, cur_a = a["reference_2017_2019"], a["current_2020"]
        lines += [
            "| Period | Windows | Positives | AUC-ROC | Avg Precision |",
            "|---|---|---|---|---|",
            f"| Reference 2017-2019 | {ref_a['n_windows']:,} | {ref_a['n_positives']:,} | "
            f"{ref_a['auc_roc']:.4f} | {ref_a['average_precision']:.4f} |",
            f"| Current 2020 | {cur_a['n_windows']:,} | {cur_a['n_positives']:,} | "
            f"{cur_a['auc_roc']:.4f} | {cur_a['average_precision']:.4f} |",
            f"\n**AUC delta = {a['auc_delta']:+.4f}** "
            f"({'degraded' if a['auc_delta'] < 0 else 'improved'})",
        ]
        if a["per_incident_type_2020"]:
            lines += [
                "\n### Per-incident-type AUC (2020)\n",
                "| Incident type | Windows | AUC-ROC |",
                "|---|---|---|",
            ]
            for row in a["per_incident_type_2020"]:
                lines.append(
                    f"| {row['incident_type']} | {row['n_windows']:,} | {row['auc_roc']:.4f} |"
                )
    else:
        lines.append("*AUC could not be computed -- insufficient positive labels.*")

    lines += [
        "\n---\n",
        "## 4. Top 20 Drifted Features (by KS statistic)\n",
        "| # | Feature | KS | PSI | Severity | Ref mean | Cur mean | Shift |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(r["feature_drift"][:20], 1):
        lines.append(
            f"| {i} | `{row['feature']}` | {row['ks_stat']:.3f} | {row['psi']:.3f} | "
            f"{row['severity']} | {row['ref_mean']:.4f} | {row['cur_mean']:.4f} | "
            f"{row['mean_shift']:+.4f} |"
        )

    lines += [
        "\n---\n",
        "## 5. Population Composition Drift\n",
        f"- Unique vessels in reference: **{pop['n_unique_vessels_reference']:,}**",
        f"- Unique vessels in current: **{pop['n_unique_vessels_current']:,}**",
        f"- Vessels in 2020 never seen in 2017-2019: "
        f"**{pop['n_new_vessels_in_current']:,}**",
    ]
    if pop.get("vessel_type_shift"):
        lines += [
            "\n**Top vessel-type shifts** (by absolute change):\n",
            "| Vessel type code | Ref % | Cur % | Delta |",
            "|---|---|---|---|",
        ]
        for row in pop["vessel_type_shift"][:8]:
            lines.append(
                f"| {row['vessel_type_code']} | {row['ref_pct']:.1f}% | "
                f"{row['cur_pct']:.1f}% | {row['delta_pct']:+.1f}% |"
            )

    lines += [
        "\n---\n",
        "## 6. Suggested Trigger Thresholds for Phase 8 Retraining\n",
    ]
    
    # FIXED: Calculate trigger thresholds based on the *pure temporal* drift
    # (the same vessels shifting over time), not the artificially lowered combined drift.
    if pt:
        ks_ref = pt["ks_stat"]
        p90_ref = abs(pt["p90_relative_shift"])
        trigger_ks = round(ks_ref * 0.75, 3)
        trigger_p90 = round(p90_ref * 0.75, 3)
        context_str = "(calibrated against pure temporal drift)"
    else:
        ks_ref = p["ks_stat"]
        p90_ref = abs(p["p90_relative_shift"])
        trigger_ks = round(ks_ref * 0.75, 3)
        trigger_p90 = round(p90_ref * 0.75, 3)
        context_str = "(calibrated against combined drift)"

    fpr_inflation = t["false_positive_rate"]["inflation_factor"]

    lines += [
        f"These thresholds are calibrated from this analysis {context_str} -- not arbitrary.\n",
        "| Trigger | Threshold | Rationale |",
        "|---|---|---|",
        f"| Primary: KS statistic | > **{trigger_ks}** | "
        f"75% of observed 1-year shift (KS={ks_ref:.3f}) |",
        f"| Secondary: P90 relative shift | > **{trigger_p90:.1%}** | "
        f"75% of observed 1-year shift ({p90_ref:.1%}) |",
        f"| Guard: FPR inflation | > **2x** | "
        f"Observed inflation was {fpr_inflation:.2f}x |",
        "\n> **Usage:** compute these metrics on a rolling 7-day window of live scores "
        "against the stored reference distribution. Fire retraining when any primary "
        "trigger is exceeded for two consecutive windows.",
    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  ✓ {path}")


# ── output: figures ───────────────────────────────────────────────────────────

def write_figures(out_dir: Path, ref: pd.DataFrame, cur: pd.DataFrame,
                  prediction: dict, feature_drift: list,
                  same_pop_2020=None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  !!  matplotlib not installed -- skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── Figure 1: Score CDF ──────────────────────────────────────────────────
    xs = np.linspace(0, 1, 500)
    ref_scores = ref["model_score"].dropna().values
    cur_scores = cur["model_score"].dropna().values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, _cdf_at(ref_scores, xs), label="Reference 2017-2019", linewidth=2)
    ax.plot(xs, _cdf_at(cur_scores, xs), label="Current 2020 (combined)",
            linewidth=2, linestyle="--")
    if same_pop_2020 is not None and len(same_pop_2020) >= 50:
        sp_scores = same_pop_2020["model_score"].dropna().values
        ax.plot(xs, _cdf_at(sp_scores, xs), label="Same-pop 2020 (fixed population)",
                linewidth=1.5, linestyle=":")
    ax.axvline(ANOMALY_THRESHOLD, color="red", linestyle=":", linewidth=1,
               label=f"Anomaly threshold ({ANOMALY_THRESHOLD})")
    ax.axvline(UPDATE_THRESHOLD, color="orange", linestyle=":", linewidth=1,
               label=f"Update gate ({UPDATE_THRESHOLD})")
    ax.set_xlabel("Model score (XGBoost hybrid)")
    ax.set_ylabel("Cumulative fraction of windows")
    ax.set_title(
        f"Score CDF: reference vs 2020   "
        f"(KS={prediction['ks_stat']:.3f}, "
        f"P90 shift={prediction['p90_relative_shift']:+.1%})"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    cdf_path = fig_dir / "score_cdf.png"
    fig.savefig(cdf_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {cdf_path}")

    # ── Figure 2: Top drifted features bar chart ─────────────────────────────
    top_n = 20
    top = feature_drift[:top_n]
    if not top:
        return

    names = [r["feature"] for r in top][::-1]
    ks_vals = [r["ks_stat"] for r in top][::-1]
    psi_vals = [r["psi"] for r in top][::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(names))
    ax.barh(y_pos - 0.2, ks_vals, height=0.4, label="KS statistic", color="steelblue")
    ax.barh(y_pos + 0.2, psi_vals, height=0.4, label="PSI", color="darkorange")

    ax.axvline(0.10, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(0.20, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(0.25, color="gray", linestyle="-", linewidth=1, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Drift magnitude")
    ax.set_title(f"Top {len(names)} drifted features (reference vs current)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    feat_path = fig_dir / "feature_ks.png"
    fig.savefig(feat_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {feat_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Drift analysis: 2017-2019 reference vs 2020 current.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--incident-parquet", type=Path, default=DEFAULT_INCIDENT_PARQUET,
        metavar="PATH",
        help="hybrid_features_incident_vessels_with_scores.parquet",
    )
    ap.add_argument(
        "--clean-parquet", type=Path, default=DEFAULT_CLEAN_PARQUET,
        metavar="PATH",
        help="hybrid_features_clean_2020_with_scores.parquet",
    )
    ap.add_argument(
        "--output", type=Path,
        default=BASE_DIR / "reports" / f"drift_{datetime.now().strftime('%Y%m%d')}",
        metavar="DIR",
        help="Output directory (created if absent)",
    )
    args = ap.parse_args(argv)

    for p in (args.incident_parquet, args.clean_parquet):
        if not p.exists():
            print(f"ERROR: {p} not found.", file=sys.stderr)
            print(
                "Run from the repo root, or supply "
                "--incident-parquet / --clean-parquet.",
                file=sys.stderr,
            )
            return 1

    args.output.mkdir(parents=True, exist_ok=True)

    inc, cur_raw, feature_cols = load_data(args.incident_parquet, args.clean_parquet)
    ref, same_pop_2020, cur = build_reference_current(inc, cur_raw)

    print("\nAnalysis 1 -- feature covariate drift ...")
    feature_drift = analyse_feature_drift(ref, cur, feature_cols)
    sig = sum(1 for r in feature_drift if r["severity"] == "significant")
    mod = sum(1 for r in feature_drift if r["severity"] == "moderate")
    print(f"  {sig} significant, {mod} moderate out of {len(feature_drift)} features")

    print("Analysis 2a -- combined drift (year + population change) ...")
    prediction = analyse_prediction_drift(ref, cur)
    print(f"  KS={prediction['ks_stat']:.4f}  PSI={prediction['psi']:.4f}  "
          f"P90 shift={prediction['p90_relative_shift']:+.1%}  "
          f"severity={prediction['severity']}")

    print("Analysis 2b -- pure temporal drift (fixed population, year change only) ...")
    pure_temporal = None
    if len(same_pop_2020) >= 50:
        pure_temporal = analyse_prediction_drift(ref, same_pop_2020)
        print(f"  KS={pure_temporal['ks_stat']:.4f}  "
              f"PSI={pure_temporal['psi']:.4f}  "
              f"P90 shift={pure_temporal['p90_relative_shift']:+.1%}  "
              f"severity={pure_temporal['severity']}")
    else:
        print("  !!  fewer than 50 same-population 2020 windows -- skipped")

    print("Analysis 3 -- population composition drift ...")
    population = analyse_population_drift(ref, cur)
    print(f"  New vessels in 2020: {population['n_new_vessels_in_current']}")

    print("Analysis 4 -- operational threshold impact ...")
    thresholds = analyse_threshold_impact(ref, cur)
    fpr_inf = thresholds["false_positive_rate"]["inflation_factor"]
    print(f"  FPR inflation at {ANOMALY_THRESHOLD}: {fpr_inf:.2f}x")

    print("Analysis 5 -- year stability check (AUC within incident population) ...")
    auc = analyse_auc_degradation(inc)
    if auc["auc_delta"] is not None:
        print(f"  AUC delta = {auc['auc_delta']:+.4f}  "
              f"(note: within-training-set check; use CV AUC=0.853 for generalization)")

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {
            "incident_parquet": str(args.incident_parquet),
            "clean_parquet": str(args.clean_parquet),
            "n_reference": int(len(ref)),
            "n_reference_vessels": int(ref["MMSI"].nunique()),
            "n_same_pop_2020": int(len(same_pop_2020)),
            "n_current": int(len(cur)),
            "n_current_vessels": int(cur["MMSI"].nunique()),
            "n_features": len(feature_cols),
            "reference_years": sorted(REFERENCE_YEARS),
            "population_note": (
                "Reference = normal windows of incident-prone vessels (2017-2019). "
                "This is what the model was calibrated against, not all normal maritime "
                "behaviour. Vessel types absent from this population will appear to drift "
                "even if their behaviour is constant."
            ),
            "in_sample_bias_note": (
                "Reference model_scores are in-sample predictions (the model was trained "
                "on this data), so combined-drift KS is an upper bound on operational drift. "
                "The pure-temporal-drift comparison scores both sides in-sample and is unbiased."
            ),
        },
        "prediction": prediction,
        "pure_temporal_drift": pure_temporal,
        "thresholds": thresholds,
        "auc": auc,
        "feature_drift": feature_drift,
        "population": population,
    }

    print(f"\nWriting outputs to {args.output}/")
    write_json(args.output / "drift.json", results)
    write_markdown(args.output / "drift_report.md", results)
    write_figures(args.output, ref, cur, prediction, feature_drift, same_pop_2020)

    print(f"\n{'='*60}")
    print(f"  Combined drift KS : {prediction['ks_stat']:.4f}  ({prediction['severity']})")
    print(f"  Combined P90 shift: {prediction['p90_relative_shift']:+.1%}")
    print(f"  FPR inflation     : {fpr_inf:.2f}x")
    if auc["auc_delta"] is not None:
        print(f"  AUC delta         : {auc['auc_delta']:+.4f}")
    if pure_temporal is not None:
        print(f"  Temporal drift KS : {pure_temporal['ks_stat']:.4f}  ({pure_temporal['severity']})")
        print(f"  Temporal P90 shift: {pure_temporal['p90_relative_shift']:+.1%}")
        # Clarified print statement
        print(f"  -> Population shift masked the temporal drift: Combined KS is {prediction['ks_stat']:.4f} vs Temporal KS {pure_temporal['ks_stat']:.4f}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())