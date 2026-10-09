"""
Final production fit of the XGBoost Hybrid model on ALL 145 incident vessels
(not a train/val/test split) — this is the artifact to actually deploy.

The k-fold CV run already told us roughly how many boosting rounds the model
needs before overfitting starts (each fold's early-stopping best_iteration).
Rather than early-stopping again here (which would need yet another held-out
slice, defeating the point of using 100% of the data), this fits a FIXED
number of rounds based on that — set N_ESTIMATORS_FINAL below to roughly the
median best_iteration your kfold_cv_hybrid.py run landed on for each fold.
Check your kfold run's console output / model.best_iteration per fold if you
still have it; if not, this defaults to a conservative estimate.

isoforest_score is excluded — tested, no measurable benefit (see session log
Section 7) — kept out for serving simplicity.

Place in : maritime/hybrid/
Run from : maritime/hybrid/
Output   : maritime/hybrid/xgboost_hybrid_final.json
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
HYBRID_DIR = BASE_DIR / "hybrid"

SEED = 42
NON_FEATURE_COLS = ['MMSI', 'label', 'incident_type', 'year', 'isoforest_score']
CATEGORICAL_COLS = ['vessel_type_code']

# Derived from the actual kfold_cv_hybrid.py run: per-fold best_iteration was
# [19, 41, 67, 69, 104] -> median 67. Note the wide spread (19 to 104, >5x) --
# that instability itself is worth mentioning in the thesis as a consequence
# of small per-fold validation groups (~17 vessels), not a bug. Median is the
# defensible central choice given that noise.
N_ESTIMATORS_FINAL = 67


def prep_X(df, feature_cols):
    X = df[feature_cols].copy()
    for c in CATEGORICAL_COLS:
        if c in X.columns:
            # Convert categorical values to strings first.
            # This avoids XGBoost's error when pandas category
            # labels have a floating-point dtype.
            X[c] = (
                X[c]
                .astype("string")
                .fillna("__MISSING__")
                .astype("category")
            )

    return X


def main():
    print("Loading full feature table...")
    df = pd.read_parquet(HYBRID_DIR / "hybrid_features_incident_vessels_with_scores.parquet")
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"{len(feature_cols)} features, {df['MMSI'].nunique()} vessels, {len(df)} windows")

    X = prep_X(df, feature_cols)
    y = df['label'].values

    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    print(f"scale_pos_weight = {scale_pos_weight:.1f}")
    print(f"Training for a fixed {N_ESTIMATORS_FINAL} rounds (no early stopping — using 100% of data)")

    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS_FINAL,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        tree_method='hist',
        enable_categorical=True,
        random_state=SEED,
    )

    print("\nTraining on all 145 vessels...")
    model.fit(X, y, verbose=50)

    out_path = HYBRID_DIR / "xgboost_hybrid_final.json"
    model.save_model(out_path)
    print(f"\nSaved final production model to {out_path}")
    print("This is the artifact to load in the MLOps serving layer — trained on")
    print("100% of available incident vessels, not the 101/145 single-split model.")

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n=== Top 15 features (final model) ===")
    print(importances.head(15))


if __name__ == "__main__":
    main()