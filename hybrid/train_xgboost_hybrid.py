"""
Trains XGBoost on the hybrid feature table using a GROUPED split by MMSI.

No vessel's windows appear in more than one split.

Evaluations:
    1. Held-out incident-vessel test fold
    2. Held-out test-fold positives vs FULL clean-2020 negative pool

Required files:

    hybrid/
        hybrid_features_incident_vessels_with_scores.parquet
        hybrid_features_clean_2020_with_scores.parquet

Output:

    hybrid/
        xgboost_hybrid.json
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
HYBRID_DIR = BASE_DIR / "hybrid"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Remaining 15% is test


# Columns that must NOT be given to XGBoost
NON_FEATURE_COLS = [
    "MMSI",
    "label",
    "incident_type",
    "year",
]


# Categorical columns
CATEGORICAL_COLS = [
    "vessel_type_code",
]


# ============================================================
# GROUPED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def grouped_split(df, seed=SEED):
    """
    Split unique MMSIs into train/validation/test.

    Every window belonging to one MMSI stays in the same split.

    This prevents vessel leakage.
    """

    print("\n" + "=" * 70)
    print("GROUPED TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    mmsis = df["MMSI"].unique()

    rng = np.random.RandomState(seed)
    rng.shuffle(mmsis)

    n = len(mmsis)

    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    train_mmsi = set(
        mmsis[:n_train]
    )

    val_mmsi = set(
        mmsis[n_train:n_train + n_val]
    )

    test_mmsi = set(
        mmsis[n_train + n_val:]
    )

    train_df = df[
        df["MMSI"].isin(train_mmsi)
    ].copy()

    val_df = df[
        df["MMSI"].isin(val_mmsi)
    ].copy()

    test_df = df[
        df["MMSI"].isin(test_mmsi)
    ].copy()

    for name, split_df, mmsi_set in [
        ("train", train_df, train_mmsi),
        ("val", val_df, val_mmsi),
        ("test", test_df, test_mmsi),
    ]:

        n_pos = int(
            split_df["label"].sum()
        )

        n_neg = len(split_df) - n_pos

        print(
            f"{name:5s}: "
            f"{len(mmsi_set):4d} vessels, "
            f"{len(split_df):8d} windows, "
            f"{n_pos:6d} positive, "
            f"{n_neg:8d} negative"
        )

        if n_pos == 0:

            print(
                f"WARNING: {name} split has ZERO positives."
            )

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    train_ids = set(train_df["MMSI"].unique())
    val_ids = set(val_df["MMSI"].unique())
    test_ids = set(test_df["MMSI"].unique())

    assert train_ids.isdisjoint(val_ids), \
        "ERROR: Train/Validation vessel leakage detected."

    assert train_ids.isdisjoint(test_ids), \
        "ERROR: Train/Test vessel leakage detected."

    assert val_ids.isdisjoint(test_ids), \
        "ERROR: Validation/Test vessel leakage detected."

    print("\nVessel leakage check: PASSED")

    return train_df, val_df, test_df


# ============================================================
# PREPARE X
# ============================================================

def prep_X(df, feature_cols):
    """
    Prepare the feature matrix for XGBoost.

    IMPORTANT:

    vessel_type_code is categorical.

    Instead of using pandas categorical dtype directly,
    we convert it to integer category codes.

    This avoids the XGBoost error:

        Category index from DataFrame has floating point dtype
    """

    X = df[feature_cols].copy()

    # --------------------------------------------------------
    # Convert categorical columns to integer codes
    # --------------------------------------------------------

    for c in CATEGORICAL_COLS:

        if c in X.columns:

            X[c] = (
                X[c]
                .fillna("UNKNOWN")
                .astype(str)
                .astype("category")
                .cat.codes
                .astype(np.int32)
            )

    # --------------------------------------------------------
    # Convert remaining columns to numeric
    # --------------------------------------------------------

    for c in X.columns:

        if c not in CATEGORICAL_COLS:

            X[c] = pd.to_numeric(
                X[c],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Replace NaN / +inf / -inf
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing numerical values with column median
    for c in X.columns:

        if X[c].isna().any():

            median_value = X[c].median()

            if pd.isna(median_value):

                median_value = 0.0

            X[c] = X[c].fillna(
                median_value
            )

    return X


# ============================================================
# PER INCIDENT TYPE REPORT
# ============================================================

def per_type_report(
    y_true_pos_types,
    pos_scores,
    normal_scores,
    label=""
):

    print(
        f"\n=== Per-Incident-Type Performance {label} ==="
    )

    unique_types = sorted(
        set(y_true_pos_types)
    )

    results = []

    for incident_type in unique_types:

        type_scores = (
            pos_scores[
                y_true_pos_types == incident_type
            ]
        )

        n_pos = len(type_scores)

        if n_pos < 5:

            print(
                f"{str(incident_type):25s} "
                f"n={n_pos:5d} "
                f"(too few positives, skipped)"
            )

            continue

        y_true = np.concatenate(
            [
                np.zeros(len(normal_scores)),
                np.ones(n_pos),
            ]
        )

        y_score = np.concatenate(
            [
                normal_scores,
                type_scores,
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

        results.append(
            (
                incident_type,
                n_pos,
                auc,
                ap,
            )
        )

        print(
            f"{str(incident_type):25s} "
            f"n={n_pos:5d} "
            f"AUC={auc:.3f} "
            f"AP={ap:.4f}"
        )

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    y_true_all = np.concatenate(
        [
            np.zeros(len(normal_scores)),
            np.ones(len(pos_scores)),
        ]
    )

    y_score_all = np.concatenate(
        [
            normal_scores,
            pos_scores,
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
        f"\n{'OVERALL':25s} "
        f"n={len(pos_scores):5d} "
        f"AUC={overall_auc:.3f} "
        f"AP={overall_ap:.4f}"
    )

    # --------------------------------------------------------
    # Best / worst incident types
    # --------------------------------------------------------

    if results:

        best = max(
            results,
            key=lambda r: r[2]
        )

        worst = min(
            results,
            key=lambda r: r[2]
        )

        print(
            f"Best:  {best[0]} "
            f"(AUC={best[2]:.3f}, n={best[1]})"
        )

        print(
            f"Worst: {worst[0]} "
            f"(AUC={worst[2]:.3f}, n={worst[1]})"
        )

    return overall_auc, overall_ap


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("XGBOOST HYBRID MODEL")
    print("=" * 70)

    print("\nProject directory:")
    print(BASE_DIR)

    print("\nHybrid directory:")
    print(HYBRID_DIR)

    # ========================================================
    # CHECK REQUIRED FILES
    # ========================================================

    incident_path = (
        HYBRID_DIR
        / "hybrid_features_incident_vessels_with_scores.parquet"
    )

    clean_path = (
        HYBRID_DIR
        / "hybrid_features_clean_2020_with_scores.parquet"
    )

    print("\nChecking required files...")

    if not incident_path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{incident_path}"
        )

    print("Incident dataset: FOUND")

    if not clean_path.exists():

        raise FileNotFoundError(
            f"Missing file:\n{clean_path}"
        )

    print("Clean 2020 dataset: FOUND")

    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        "\nLoading feature tables "
        "(with deep-model reconstruction-error features)..."
    )

    df = pd.read_parquet(
        incident_path
    )

    clean_df = pd.read_parquet(
        clean_path
    )

    print(
        f"\nIncident dataset shape: "
        f"{df.shape}"
    )

    print(
        f"Clean 2020 dataset shape: "
        f"{clean_df.shape}"
    )

    print(
        f"\nIncident positive windows: "
        f"{int(df['label'].sum()):,}"
    )

    print(
        f"Clean 2020 positive windows: "
        f"{int(clean_df['label'].sum()):,}"
    )

    # ========================================================
    # CHECK DEEP MODEL FEATURES
    # ========================================================

    required_deep_features = [

        "trans_err_lat",
        "trans_err_lon",
        "trans_err_speed",
        "trans_err_course",
        "trans_err_computed_speed",
        "trans_err_acceleration",
        "trans_err_heading_change",
        "trans_err_agg",

        "lstm_err_lat",
        "lstm_err_lon",
        "lstm_err_speed",
        "lstm_err_course",
        "lstm_err_computed_speed",
        "lstm_err_acceleration",
        "lstm_err_heading_change",
        "lstm_err_agg",
    ]

    missing_deep_features = [

        c
        for c in required_deep_features
        if c not in df.columns

    ]

    if missing_deep_features:

        raise ValueError(
            "Missing deep-model features:\n"
            + "\n".join(
                missing_deep_features
            )
        )

    print(
        "\nDeep-model feature check: PASSED"
    )

    # ========================================================
    # CREATE FEATURE LIST
    # ========================================================

    feature_cols = [

        c
        for c in df.columns
        if c not in NON_FEATURE_COLS

    ]

    print(
        f"\nNumber of model features: "
        f"{len(feature_cols)}"
    )

    print("\nFeature columns:")

    for i, col in enumerate(
        feature_cols,
        start=1
    ):

        print(
            f"{i:2d}. {col}"
        )

    # We expect 55 features:
    #
    # 39 handcrafted/statistical features
    # + 16 deep-model reconstruction features
    #
    expected_features = 55

    if len(feature_cols) != expected_features:

        raise ValueError(
            f"Expected {expected_features} "
            f"features but found "
            f"{len(feature_cols)}."
        )

    print(
        f"\nFeature count check: "
        f"PASSED ({len(feature_cols)} features)"
    )

    # ========================================================
    # GROUPED SPLIT
    # ========================================================

    train_df, val_df, test_df = grouped_split(
        df
    )

    # ========================================================
    # PREPARE MATRICES
    # ========================================================

    print(
        "\nPreparing train/validation/test matrices..."
    )

    X_train = prep_X(
        train_df,
        feature_cols
    )

    y_train = train_df[
        "label"
    ].values.astype(np.int32)

    X_val = prep_X(
        val_df,
        feature_cols
    )

    y_val = val_df[
        "label"
    ].values.astype(np.int32)

    X_test = prep_X(
        test_df,
        feature_cols
    )

    y_test = test_df[
        "label"
    ].values.astype(np.int32)

    # ========================================================
    # CHECK MATRICES
    # ========================================================

    print("\nMatrix shapes:")

    print(
        f"X_train: {X_train.shape}"
    )

    print(
        f"X_val:   {X_val.shape}"
    )

    print(
        f"X_test:  {X_test.shape}"
    )

    # Check for remaining NaN / inf

    for name, X in [
        ("X_train", X_train),
        ("X_val", X_val),
        ("X_test", X_test),
    ]:

        if X.isna().any().any():

            raise ValueError(
                f"{name} still contains NaN values."
            )

        if np.isinf(
            X.to_numpy()
        ).any():

            raise ValueError(
                f"{name} contains infinite values."
            )

    print(
        "\nFeature matrix validation: PASSED"
    )

    # ========================================================
    # CLASS BALANCE
    # ========================================================

    positive_count = (
        y_train == 1
    ).sum()

    negative_count = (
        y_train == 0
    ).sum()

    scale_pos_weight = (
        negative_count
        / max(positive_count, 1)
    )

    print(
        "\nTraining class distribution:"
    )

    print(
        f"Positive: {positive_count:,}"
    )

    print(
        f"Negative: {negative_count:,}"
    )

    print(
        f"scale_pos_weight = "
        f"{scale_pos_weight:.2f}"
    )

    # ========================================================
    # CREATE XGBOOST MODEL
    # ========================================================

    print(
        "\nCreating XGBoost model..."
    )

    model = xgb.XGBClassifier(

        n_estimators=1000,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        scale_pos_weight=scale_pos_weight,

        eval_metric="aucpr",

        early_stopping_rounds=50,

        tree_method="hist",

        # IMPORTANT:
        # vessel_type_code has already been converted
        # into integer category codes.
        enable_categorical=False,

        random_state=SEED,
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING XGBOOST"
    )

    print(
        "=" * 70
    )

    model.fit(

        X_train,
        y_train,

        eval_set=[
            (
                X_val,
                y_val
            )
        ],

        verbose=50,
    )

    # ========================================================
    # EVALUATION 1
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATION 1: "
        "HELD-OUT INCIDENT-VESSEL TEST FOLD"
    )

    print(
        "=" * 70
    )

    test_scores = model.predict_proba(
        X_test
    )[:, 1]

    test_auc = roc_auc_score(
        y_test,
        test_scores
    )

    test_ap = average_precision_score(
        y_test,
        test_scores
    )

    print(
        f"\nAUC = {test_auc:.4f}"
    )

    print(
        f"AP  = {test_ap:.4f}"
    )

    print(
        f"Positive windows = "
        f"{int(y_test.sum()):,}"
    )

    print(
        f"Total windows = "
        f"{len(y_test):,}"
    )

    # ========================================================
    # PER TYPE — TEST NEGATIVES
    # ========================================================

    pos_mask = (
        y_test == 1
    )

    per_type_report(

        test_df.loc[
            pos_mask,
            "incident_type"
        ].values,

        test_scores[
            pos_mask
        ],

        test_scores[
            ~pos_mask
        ],

        label=(
            "(vs. held-out "
            "test-fold negatives)"
        ),
    )

    # ========================================================
    # EVALUATION 2
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "EVALUATION 2: "
        "TEST POSITIVES VS CLEAN 2020"
    )

    print(
        "=" * 70
    )

    print(
        "\nPreparing clean-2020 matrix..."
    )

    X_clean = prep_X(
        clean_df,
        feature_cols
    )

    print(
        f"X_clean: {X_clean.shape}"
    )

    clean_scores = model.predict_proba(
        X_clean
    )[:, 1]

    per_type_report(

        test_df.loc[
            pos_mask,
            "incident_type"
        ].values,

        test_scores[
            pos_mask
        ],

        clean_scores,

        label=(
            "(vs. full clean-2020 "
            "pool — comparable to "
            "Transformer/LSTM/"
            "Isolation Forest)"
        ),
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP 15 FEATURES"
    )

    print(
        "=" * 70
    )

    importances = pd.Series(

        model.feature_importances_,

        index=feature_cols

    ).sort_values(
        ascending=False
    )

    print(
        importances.head(15)
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        HYBRID_DIR
        / "xgboost_hybrid.json"
    )

    model.save_model(
        model_path
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSaved model to:"
    )

    print(
        model_path
    )

    print(
        "\nFinal test performance:"
    )

    print(
        f"  AUC = {test_auc:.4f}"
    )

    print(
        f"  AP  = {test_ap:.4f}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()