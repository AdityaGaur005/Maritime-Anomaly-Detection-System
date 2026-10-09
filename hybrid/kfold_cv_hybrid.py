"""
Grouped 5-fold cross-validation by MMSI for the XGBoost hybrid model.

Your last run trained on one 70/15/15 vessel split (101/21/23 vessels) and
got AUC=0.837. With only 145 total vessels, that single split is not enough
to know whether 0.837 is a stable estimate or this particular split happened
to be favorable. This runs 5 different grouped splits (each vessel appears
in the test fold exactly once, across the 5 folds combined) and reports
mean +/- std for overall AUC/AP, plus per-incident-type AUC aggregated
across all folds (so rare types get pooled sample size instead of being
stuck at n<20 in any single fold).

Uses a per-fold train/val split within the training portion for early
stopping (no separate held-out val needed beyond that).

Place in : maritime/hybrid/
Run from : maritime/hybrid/
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score


BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
HYBRID_DIR = BASE_DIR / "hybrid"

SEED = 42
N_FOLDS = 5

NON_FEATURE_COLS = [
    'MMSI',
    'label',
    'incident_type',
    'year',
    'isoforest_score'
]  # tested — no measurable AUC/AP gain (pooled 0.8564->0.8496,
  # within CV fold noise), dropped to avoid a third model in serving

CATEGORICAL_COLS = ['vessel_type_code']


def prep_X(df, feature_cols, category_levels):
    """
    Prepare feature matrix for XGBoost.

    Categorical columns are converted to strings first and then to pandas
    categorical dtype using a category vocabulary learned from the training
    portion of the current fold. This ensures train/validation/test all use
    the exact same category mapping.
    """
    X = df[feature_cols].copy()

    for c in CATEGORICAL_COLS:
        if c in X.columns:
            X[c] = X[c].fillna("__MISSING__").astype(str)

            X[c] = pd.Categorical(
                X[c],
                categories=category_levels[c]
            )

    return X


def main():
    print("Loading feature table...")

    df = pd.read_parquet(
        HYBRID_DIR / "hybrid_features_incident_vessels_with_scores.parquet"
    )

    feature_cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
    ]

    print(
        f"{len(feature_cols)} features, "
        f"{df['MMSI'].nunique()} vessels, "
        f"{len(df)} windows"
    )

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df['MMSI'].values

    fold_aucs, fold_aps = [], []

    all_pos_types = []
    all_pos_scores = []
    all_neg_scores = []

    for fold, (trainval_idx, test_idx) in enumerate(
        gkf.split(df, groups=groups)
    ):
        trainval_df = df.iloc[trainval_idx]
        test_df = df.iloc[test_idx]

        # Further split trainval into train/val (grouped) for early stopping
        trainval_mmsi = trainval_df['MMSI'].unique()

        rng = np.random.RandomState(SEED + fold)
        rng.shuffle(trainval_mmsi)

        n_val = max(
            1,
            int(len(trainval_mmsi) * 0.15)
        )

        val_mmsi = set(trainval_mmsi[:n_val])
        train_mmsi = set(trainval_mmsi[n_val:])

        train_df = trainval_df[
            trainval_df['MMSI'].isin(train_mmsi)
        ]

        val_df = trainval_df[
            trainval_df['MMSI'].isin(val_mmsi)
        ]

        # ---------------------------------------------------------------
        # FIX:
        # Build categorical vocabulary ONLY from the training vessels.
        # Then use the exact same categories for train/validation/test.
        # ---------------------------------------------------------------
        category_levels = {}

        for c in CATEGORICAL_COLS:
            if c in train_df.columns:

                levels = sorted(
                    train_df[c]
                    .fillna("__MISSING__")
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if "__MISSING__" not in levels:
                    levels.append("__MISSING__")

                category_levels[c] = levels

        X_train = prep_X(
            train_df,
            feature_cols,
            category_levels
        )

        X_val = prep_X(
            val_df,
            feature_cols,
            category_levels
        )

        X_test = prep_X(
            test_df,
            feature_cols,
            category_levels
        )

        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values

        n_test_vessels = test_df['MMSI'].nunique()
        n_test_pos = int(y_test.sum())

        print(
            f"\n--- Fold {fold+1}/{N_FOLDS}: "
            f"{n_test_vessels} test vessels, "
            f"{len(test_df)} windows, "
            f"{n_test_pos} positive ---"
        )

        if n_test_pos == 0:
            print(
                "  WARNING: zero positives in this fold's test set, "
                "skipping fold."
            )
            continue

        scale_pos_weight = (
            (y_train == 0).sum()
            / max((y_train == 1).sum(), 1)
        )

        model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            early_stopping_rounds=50,
            tree_method='hist',
            enable_categorical=True,
            random_state=SEED + fold,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        print(
            f"  best_iteration: {model.best_iteration}"
        )

        test_scores = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(
            y_test,
            test_scores
        )

        ap = average_precision_score(
            y_test,
            test_scores
        )

        fold_aucs.append(auc)
        fold_aps.append(ap)

        print(
            f"  Fold {fold+1} AUC={auc:.3f}  "
            f"AP={ap:.4f}"
        )

        # Store positive predictions by incident type
        # and all negative predictions for pooled analysis.
        pos_mask = y_test == 1

        all_pos_types.append(
            test_df.loc[
                pos_mask,
                'incident_type'
            ].values
        )

        all_pos_scores.append(
            test_scores[pos_mask]
        )

        all_neg_scores.append(
            test_scores[~pos_mask]
        )

    # -------------------------------------------------------------------
    # Cross-validation summary
    # -------------------------------------------------------------------

    fold_aucs = np.array(fold_aucs)
    fold_aps = np.array(fold_aps)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    print(
        f"Per-fold AUC: "
        f"{[f'{a:.3f}' for a in fold_aucs]}"
    )

    print(
        f"Mean AUC: {fold_aucs.mean():.3f}  "
        f"+/- {fold_aucs.std():.3f}"
    )

    print(
        f"Per-fold AP:  "
        f"{[f'{a:.4f}' for a in fold_aps]}"
    )

    print(
        f"Mean AP:  {fold_aps.mean():.4f}  "
        f"+/- {fold_aps.std():.4f}"
    )

    if fold_aucs.std() > 0.05:

        print(
            "\nSpread across folds is large (std > 0.05) — "
            "the single-split"
        )

        print(
            "result of 0.837 should be treated as one sample "
            "from a noisy"
        )

        print(
            "distribution, not a fixed number. Report the "
            "mean +/- std, not"
        )

        print(
            "a single fold's number, in your writeup."
        )

    else:

        print(
            "\nSpread across folds is small — 0.837 from the "
            "single split was"
        )

        print(
            "a reasonably representative estimate, not a "
            "lucky outlier."
        )

    # -------------------------------------------------------------------
    # Per-incident-type AUC, pooled across all folds' test predictions
    # -------------------------------------------------------------------

    pos_types_all = np.concatenate(all_pos_types)
    pos_scores_all = np.concatenate(all_pos_scores)
    neg_scores_all = np.concatenate(all_neg_scores)

    print("\n" + "=" * 60)
    print(
        "PER-INCIDENT-TYPE AUC, POOLED ACROSS ALL FOLDS"
    )
    print(
        "(each incident vessel contributes to exactly one fold's "
        "test set,"
    )
    print(
        " so this pools every incident-vessel prediction across "
        "the full"
    )
    print(
        " 145-vessel population, unlike the single-split table "
        "which only"
    )
    print(
        " had 23 test vessels)"
    )
    print("=" * 60)

    unique_types = sorted(
        set(pos_types_all)
    )

    for t in unique_types:

        type_scores = pos_scores_all[
            pos_types_all == t
        ]

        n_pos = len(type_scores)

        if n_pos < 5:

            print(
                f"{t:45s} "
                f"n={n_pos:5d}  "
                f"(too few, skipped)"
            )

            continue

        y_true = np.concatenate(
            [
                np.zeros(len(neg_scores_all)),
                np.ones(n_pos)
            ]
        )

        y_score = np.concatenate(
            [
                neg_scores_all,
                type_scores
            ]
        )

        auc = roc_auc_score(
            y_true,
            y_score
        )

        ap = average_precision_score(
            y_true,
            y_score
        )

        print(
            f"{t:45s} "
            f"n={n_pos:5d}  "
            f"AUC={auc:.3f}  "
            f"AP={ap:.4f}"
        )

    # -------------------------------------------------------------------
    # Overall pooled out-of-fold performance
    # -------------------------------------------------------------------

    y_true_all = np.concatenate(
        [
            np.zeros(len(neg_scores_all)),
            np.ones(len(pos_scores_all))
        ]
    )

    y_score_all = np.concatenate(
        [
            neg_scores_all,
            pos_scores_all
        ]
    )

    overall_auc = roc_auc_score(
        y_true_all,
        y_score_all
    )

    overall_ap = average_precision_score(
        y_true_all,
        y_score_all
    )

    print(
        f"\n{'OVERALL (pooled)':45s} "
        f"n={len(pos_scores_all):5d}  "
        f"AUC={overall_auc:.3f}  "
        f"AP={overall_ap:.4f}"
    )


if __name__ == "__main__":
    main()
