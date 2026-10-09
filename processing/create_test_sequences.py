import pandas as pd
import numpy as np
from tqdm import tqdm

WINDOW_SIZE = 30
STRIDE = 15
FEATURES = [
    'lat', 'lon',
    'speed_over_ground_knots',
    'course_over_ground_deg',
    'computed_speed_knots',
    'acceleration_knots_per_sec',
    'heading_change_deg'
]

YEAR_PATHS = {
    '2017': r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\hawaii_2017.parquet",
    '2018': r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\hawaii_2018.parquet",
    '2019': r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\hawaii_2019.parquet",
    '2020': r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\hawaii_2020.parquet"
}



def create_labeled_sequences(df, window_size, stride, feature_cols):
    """Sliding windows over vessels that have at least one incident-labeled
    point. Window label = 1 if any point in the window is anomalous."""
    sequences, labels = [], []
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
            label = int(inc[start:start + window_size].any())
            sequences.append(window)
            labels.append(label)
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    # --- 1. Build anomaly test set across all years ---
    all_seqs, all_labels = [], []
    for year, path in YEAR_PATHS.items():
        df = pd.read_parquet(path)
        seq, lab = create_labeled_sequences(df, WINDOW_SIZE, STRIDE, FEATURES)
        print(year, seq.shape, int(lab.sum()), "anomalous windows")
        all_seqs.append(seq)
        all_labels.append(lab)

    X_anomaly_test = np.concatenate(all_seqs, axis=0)
    y_anomaly_test = np.concatenate(all_labels, axis=0)
    np.save(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\X_anomaly_test.npy", X_anomaly_test)
    np.save(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\y_anomaly_test.npy", y_anomaly_test)
    print("X_anomaly_test:", X_anomaly_test.shape, "positives:", int(y_anomaly_test.sum()))

    # --- 2. Normalize test sets with saved TRAIN stats (never recompute) ---
    mean = np.load(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\processing\norm_mean.npy")
    std = np.load(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\processing\norm_std.npy")


    X_normal_test = np.load(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\normal_sequences_2020.npy")
    X_normal_test_norm = (X_normal_test - mean) / (std + 1e-8)
    np.save(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\X_normal_test_norm.npy", X_normal_test_norm)
    print("X_normal_test_norm:", X_normal_test_norm.shape)

    X_anomaly_test_norm = (X_anomaly_test - mean) / (std + 1e-8)
    np.save(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\X_anomaly_test_norm.npy", X_anomaly_test_norm)

    # --- 3. Train/val split from X_train_norm (95/5, shuffled) ---
    X_train_full = np.load(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\X_train_norm.npy")
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train_full))
    split = int(0.95 * len(idx))
    X_train = X_train_full[idx[:split]]
    X_val = X_train_full[idx[split:]]
    np.save(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\X_train.npy", X_train)
    np.save(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime\X_val.npy", X_val)
    print("X_train:", X_train.shape, "X_val:", X_val.shape)


if __name__ == "__main__":
    main()
