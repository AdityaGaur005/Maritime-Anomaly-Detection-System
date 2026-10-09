"""
Rebuilds the anomaly test set with two changes from the original create_sequence_test.py:

1. LABEL TIGHTENING: a window is now labeled positive only if at least
   MIN_OVERLAP_FRAC of its 30 points fall inside the incident interval,
   instead of the old rule (ANY single point overlapping = positive).
   This directly targets label noise from windows that only brush the edge
   of an incident period.

   Both the old ("any") and new ("strict") label arrays are saved, so you
   can re-run evaluation against both and directly compare whether AUC/AP
   improves once labels are tightened. This is the diagnostic you actually
   want before concluding label noise was the ceiling.

2. MMSI TRACKING: mmsi array saved alongside every sequence array, needed
   for grouped CV in the XGBoost hybrid script.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

from pathlib import Path

WINDOW_SIZE = 30
STRIDE = 15
MIN_OVERLAP_FRAC = 0.5

FEATURES = [
    'lat', 'lon',
    'speed_over_ground_knots',
    'course_over_ground_deg',
    'computed_speed_knots',
    'acceleration_knots_per_sec',
    'heading_change_deg'
]

BASE_DIR = Path(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime")
DATA_DIR = BASE_DIR / "HawaiiCoast_GT"
PROCESSING_DIR = BASE_DIR / "processing"

YEAR_PATHS = {
    '2017': DATA_DIR / "hawaii_2017.parquet",
    '2018': DATA_DIR / "hawaii_2018.parquet",
    '2019': DATA_DIR / "hawaii_2019.parquet",
    '2020': DATA_DIR / "hawaii_2020.parquet"
}

OUT_DIR = PROCESSING_DIR



def create_labeled_sequences(df, window_size, stride, feature_cols, min_overlap_frac):
    """
    Sliding windows over vessels that have at least one incident-labeled point.
    Returns sequences, mmsi array, label_any (old rule), label_strict (new rule),
    and overlap_frac (the raw fraction, useful for inspecting the threshold choice).
    """
    sequences, mmsi_list = [], []
    labels_any, labels_strict, overlap_fracs = [], [], []

    grouped = df.sort_values(['MMSI', 'datetime_hst']).groupby('MMSI')
    for mmsi, group in tqdm(grouped, total=df['MMSI'].nunique()):
        if not group['is_incident'].any():
            continue
        data = group[feature_cols].values
        inc = group['is_incident'].values
        T = data.shape[0]
        if T < window_size:
            continue
        for start in range(0, T - window_size + 1, stride):
            window = data[start:start + window_size]
            window_inc = inc[start:start + window_size]
            frac = window_inc.mean()  # fraction of points flagged incident in this window

            sequences.append(window)
            mmsi_list.append(mmsi)
            labels_any.append(int(window_inc.any()))
            labels_strict.append(int(frac >= min_overlap_frac))
            overlap_fracs.append(frac)

    return (np.array(sequences, dtype=np.float32),
            np.array(mmsi_list),
            np.array(labels_any, dtype=np.int64),
            np.array(labels_strict, dtype=np.int64),
            np.array(overlap_fracs, dtype=np.float32))


def create_normal_sequences(df, window_size, stride, feature_cols):
    """Same as original create_sequence.py version, with MMSI added."""
    sequences, mmsi_list = [], []
    grouped = df.sort_values(['MMSI', 'datetime_hst']).groupby('MMSI')
    for mmsi, group in tqdm(grouped, total=df['MMSI'].nunique(), desc="Processing vessels"):
        if group['is_incident'].any():
            continue
        data = group[feature_cols].values
        T = data.shape[0]
        if T < window_size:
            continue
        for start in range(0, T - window_size + 1, stride):
            window = data[start:start + window_size]
            sequences.append(window)
            mmsi_list.append(mmsi)
    return np.array(sequences, dtype=np.float32), np.array(mmsi_list)


def main():
    # --- 1. Build anomaly test set across all years, with both label rules ---
    all_seqs, all_mmsi = [], []
    all_labels_any, all_labels_strict, all_overlap_fracs = [], [], []

    for year, path in YEAR_PATHS.items():
        df = pd.read_parquet(path)
        seq, mmsi_arr, lab_any, lab_strict, frac = create_labeled_sequences(
            df, WINDOW_SIZE, STRIDE, FEATURES, MIN_OVERLAP_FRAC
        )
        print(f"{year}: {seq.shape[0]} windows | "
              f"any-overlap positives: {int(lab_any.sum())} | "
              f"strict (>={MIN_OVERLAP_FRAC*100:.0f}%) positives: {int(lab_strict.sum())}")
        all_seqs.append(seq)
        all_mmsi.append(mmsi_arr)
        all_labels_any.append(lab_any)
        all_labels_strict.append(lab_strict)
        all_overlap_fracs.append(frac)

    X_anomaly_test = np.concatenate(all_seqs, axis=0)
    mmsi_anomaly_test = np.concatenate(all_mmsi, axis=0)
    y_anomaly_test_any = np.concatenate(all_labels_any, axis=0)       # old rule, kept for comparison
    y_anomaly_test_strict = np.concatenate(all_labels_strict, axis=0)  # new rule, use this going forward
    overlap_frac_all = np.concatenate(all_overlap_fracs, axis=0)

    np.save(OUT_DIR / "X_anomaly_test.npy", X_anomaly_test)
    np.save(OUT_DIR / "mmsi_anomaly_test.npy", mmsi_anomaly_test)
    np.save(OUT_DIR / "y_anomaly_test.npy", y_anomaly_test_any)
    np.save(OUT_DIR / "y_anomaly_test_strict.npy", y_anomaly_test_strict)
    np.save(OUT_DIR / "overlap_frac_anomaly_test.npy", overlap_frac_all)

    print(f"\nTotal: {X_anomaly_test.shape[0]} windows")
    print(f"  any-overlap positives:    {int(y_anomaly_test_any.sum())} "
          f"({y_anomaly_test_any.mean()*100:.3f}%)")
    print(f"  strict (>={MIN_OVERLAP_FRAC*100:.0f}%) positives: {int(y_anomaly_test_strict.sum())} "
          f"({y_anomaly_test_strict.mean()*100:.3f}%)")

    # --- 2. Rebuild 2020 normal test set with MMSI ---
    print("\nRebuilding 2020 normal sequences with MMSI...")
    df_2020 = pd.read_parquet(YEAR_PATHS['2020'])
    X_normal_test, mmsi_normal_test = create_normal_sequences(df_2020, WINDOW_SIZE, STRIDE, FEATURES)
    np.save(OUT_DIR / "normal_sequences_2020_with_mmsi.npy", X_normal_test)
    np.save(OUT_DIR / "mmsi_normal_test.npy", mmsi_normal_test)
    print(f"Normal test: {X_normal_test.shape[0]} windows")

    # --- 3. Normalize with saved TRAIN stats (never recompute) ---
    mean = np.load(OUT_DIR / "norm_mean.npy")
    std = np.load(OUT_DIR / "norm_std.npy")


    X_normal_test_norm = (X_normal_test - mean) / (std + 1e-8)
    np.save(OUT_DIR / "X_normal_test_norm.npy", X_normal_test_norm)

    X_anomaly_test_norm = (X_anomaly_test - mean) / (std + 1e-8)
    np.save(OUT_DIR / "X_anomaly_test_norm.npy", X_anomaly_test_norm)

    print("\nDone. Saved:")
    print("  y_anomaly_test.npy          -> old 'any overlap' labels (for comparison)")
    print("  y_anomaly_test_strict.npy   -> new tightened labels (use this going forward)")
    print("  mmsi_anomaly_test.npy, mmsi_normal_test.npy -> for grouped CV")


if __name__ == "__main__":
    main()
