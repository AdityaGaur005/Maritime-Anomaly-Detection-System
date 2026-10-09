"""
Adds Isolation Forest's anomaly score as one more feature column to the
existing hybrid feature tables — Option B from the ensemble discussion:
let XGBoost learn how to weight Isolation Forest's signal itself, rather
than building a separate blending/meta-model stage.

Refits Isolation Forest with the same winning hyperparameters and seed as
isoforest_per_type_diagnostic.py (n_estimators=300, max_samples=256,
contamination=0.0103, 200K subsample, seed=42) so the score is directly
consistent with the standalone Isolation Forest numbers already reported.

Isolation Forest was originally fit/evaluated on NORMALIZED data
(X_train.npy), so the raw hybrid_sequences_*.npy arrays are normalized
here with the same norm_mean/norm_std before flattening and scoring —
skipping this step would silently mismatch the scale the model learned.

Overwrites hybrid_features_<mode>_with_scores.parquet in place, adding a
single 'isoforest_score' column. No changes needed to kfold_cv_hybrid.py —
it already picks up every column as a feature automatically.

Place in : maritime/hybrid/
Run from : maritime/hybrid/
Usage    : python add_isoforest_score.py incident_vessels
           python add_isoforest_score.py clean_2020
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest

BASE_DIR = Path(r"C:\Users\Aman Hooda\Desktop\gaur")
HYBRID_DIR = BASE_DIR / "hybrid"
MODEL_DIR = BASE_DIR / "model"

N_ESTIMATORS = 300
MAX_SAMPLES = 256
CONTAMINATION = 0.0103
SUBSAMPLE_SIZE = 200_000
SEED = 42


def fit_isoforest():
    print("Fitting Isolation Forest (same hyperparameters as isoforest_per_type_diagnostic.py)...")
    X_train = np.load(BASE_DIR / "X_train.npy")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(len(X_train_flat), size=SUBSAMPLE_SIZE, replace=False)
    X_train_sub = X_train_flat[sub_idx]

    iso = IsolationForest(
        n_estimators=N_ESTIMATORS, max_samples=MAX_SAMPLES,
        contamination=CONTAMINATION, random_state=SEED, n_jobs=-1,
    )
    iso.fit(X_train_sub)
    print(f"Fit on {SUBSAMPLE_SIZE:,} rows, n_estimators={N_ESTIMATORS}, max_samples={MAX_SAMPLES}")
    return iso


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('incident_vessels', 'clean_2020'):
        print("Usage: python add_isoforest_score.py [incident_vessels|clean_2020]")
        sys.exit(1)
    mode = sys.argv[1]

    iso = fit_isoforest()

    print(f"\nLoading {mode} raw sequences + feature table...")
    X_raw = np.load(HYBRID_DIR / f"hybrid_sequences_{mode}.npy")
    df_path = HYBRID_DIR / f"hybrid_features_{mode}_with_scores.parquet"
    df = pd.read_parquet(df_path)
    assert len(X_raw) == len(df), f"Row count mismatch: {len(X_raw)} sequences vs {len(df)} feature rows"

    # FIX: load the actual numpy arrays instead of Path objects
    mean = np.load(BASE_DIR / "norm_mean.npy")
    std = np.load(BASE_DIR / "norm_std.npy")

    X_norm = (X_raw - mean) / (std + 1e-8)
    X_flat = X_norm.reshape(X_norm.shape[0], -1)

    print("Scoring...")
    iso_score = -iso.score_samples(X_flat)  # higher = more anomalous, same convention as elsewhere

    df['isoforest_score'] = iso_score
    df.to_parquet(df_path, index=False)
    print(f"Saved {df_path} — {df.shape[1]} columns total (added isoforest_score)")
    print(f"isoforest_score stats: mean={iso_score.mean():.4f}, std={iso_score.std():.4f}, "
          f"min={iso_score.min():.4f}, max={iso_score.max():.4f}")


if __name__ == "__main__":
    main()
