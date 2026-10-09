"""
Isolation Forest per-incident-type diagnostic — completes the 4-way
comparison (Transformer-VAE, LSTM-AE, XGBoost hybrid, Isolation Forest).

Reruns the exact grid search from tune_isoforest.py (same seed, same
200K subsample, same hyperparameter grid) rather than depending on old
terminal output, so this is reproducible on its own. Hyperparameters are
selected by AUC on the anomaly-vessel test population (same criterion your
original script used) — that's mild leakage since there's no separate
validation set, a pre-existing property of the original script, not
something introduced here.

For the final per-incident-type report (the number to compare against your
other three models), scores against the FULL clean-2020 normal pool as the
negative set — matching the standard used for Transformer/LSTM/XGBoost.

Place in : maritime/model/   (needs X_train.npy, X_anomaly_test_norm.npy,
                               y_anomaly_test.npy already there)
Run from : maritime/model/
"""

import numpy as np
from pathlib import Path
from itertools import product
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent
PROCESSING_DIR = BASE_DIR.parent / "processing"

N_ESTIMATORS_LIST = [200, 300, 500]
MAX_SAMPLES_LIST = [256, 1024, "auto"]
CONTAMINATION = 0.0103
SUBSAMPLE_SIZE = 200_000
SEED = 42


def per_type_report(pos_types, pos_scores, neg_scores, label=""):
    print(f"\n=== Per-Incident-Type Performance {label} ===")
    unique_types = sorted(set(pos_types))
    results = []
    for t in unique_types:
        type_scores = pos_scores[pos_types == t]
        n_pos = len(type_scores)
        if n_pos < 5:
            print(f"{t:45s} n={n_pos:5d}  (too few positives, skipped)")
            continue
        y_true = np.concatenate([np.zeros(len(neg_scores)), np.ones(n_pos)])
        y_score = np.concatenate([neg_scores, type_scores])
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        results.append((t, n_pos, auc, ap))
        print(f"{t:45s} n={n_pos:5d}  AUC={auc:.3f}  AP={ap:.4f}")

    y_true_all = np.concatenate([np.zeros(len(neg_scores)), np.ones(len(pos_scores))])
    y_score_all = np.concatenate([neg_scores, pos_scores])
    overall_auc = roc_auc_score(y_true_all, y_score_all)
    overall_ap = average_precision_score(y_true_all, y_score_all)
    print(f"\n{'OVERALL':45s} n={len(pos_scores):5d}  AUC={overall_auc:.3f}  AP={overall_ap:.4f}")

    if results:
        best = max(results, key=lambda r: r[2])
        worst = min(results, key=lambda r: r[2])
        print(f"Best:  {best[0]} (AUC={best[2]:.3f}, n={best[1]})")
        print(f"Worst: {worst[0]} (AUC={worst[2]:.3f}, n={worst[1]})")
    return overall_auc, overall_ap


def main():
    print("Loading data...")
    X_train = np.load(BASE_DIR / "X_train.npy")
    X_anomaly_test = np.load(BASE_DIR / "X_anomaly_test_norm.npy")
    y_anomaly_test = np.load(BASE_DIR / "y_anomaly_test.npy")
    X_normal_test = np.load(PROCESSING_DIR / "X_normal_test_norm.npy")
    incident_types = np.load(PROCESSING_DIR / "incident_type_anomaly_test.npy", allow_pickle=True)

    assert len(X_anomaly_test) == len(y_anomaly_test) == len(incident_types), \
        "Row count mismatch between X_anomaly_test, y_anomaly_test, incident_type_anomaly_test"

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_anomaly_flat = X_anomaly_test.reshape(X_anomaly_test.shape[0], -1)
    X_normal_flat = X_normal_test.reshape(X_normal_test.shape[0], -1)

    rng = np.random.default_rng(SEED)
    sub_idx = rng.choice(len(X_train_flat), size=SUBSAMPLE_SIZE, replace=False)
    X_train_sub = X_train_flat[sub_idx]

    print("\nGrid search (reproducing tune_isoforest.py exactly)...")
    best_model, best_auc, best_scores = None, -1, None
    for n_est, max_samp in product(N_ESTIMATORS_LIST, MAX_SAMPLES_LIST):
        iso = IsolationForest(
            n_estimators=n_est, max_samples=max_samp,
            contamination=CONTAMINATION, random_state=SEED, n_jobs=-1,
        )
        iso.fit(X_train_sub)
        scores = -iso.score_samples(X_anomaly_flat)  # higher = more anomalous
        auc = roc_auc_score(y_anomaly_test, scores)
        ap = average_precision_score(y_anomaly_test, scores)
        print(f"n_estimators={n_est}, max_samples={max_samp}  AUC={auc:.4f}  AP={ap:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_model = iso
            best_scores = scores

    print(f"\nBest config: n_estimators={best_model.n_estimators}, "
          f"max_samples={best_model.max_samples}  (AUC={best_auc:.4f})")

    # --- Eval 1: same population the grid search selected on (for continuity with your original run) ---
    ap1 = average_precision_score(y_anomaly_test, best_scores)
    print("\n=== Eval 1: within anomaly-vessel test population (matches original tune_isoforest.py metric) ===")
    print(f"AUC={best_auc:.3f}  AP={ap1:.4f}")

    pos_mask = y_anomaly_test == 1
    per_type_report(
        incident_types[pos_mask],
        best_scores[pos_mask],
        best_scores[~pos_mask],
        label="(vs. within-population negatives)"
    )

    # --- Eval 2: vs full clean-2020 pool, comparable to Transformer/LSTM/XGBoost hybrid numbers ---
    print("\nScoring full clean-2020 normal pool...")
    normal_scores = -best_model.score_samples(X_normal_flat)

    per_type_report(
        incident_types[pos_mask],
        best_scores[pos_mask],
        normal_scores,
        label="(vs. full clean-2020 pool — comparable to Transformer/LSTM/XGBoost hybrid)"
    )


if __name__ == "__main__":
    main()
