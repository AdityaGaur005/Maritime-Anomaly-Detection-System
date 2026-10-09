"""
Extracts the majority incident type for every window in the EXISTING anomaly
test set, without rebuilding the feature sequences (X_anomaly_test.npy already
has those). Verifies alignment by rebuilding the 'any-overlap' label locally
and asserting it matches y_anomaly_test.npy exactly.

If the assertion fails: X_anomaly_test.npy was built with different data or
different windowing logic than this script produces. Do NOT use the type
array in that case — you'd be silently misattributing incident types to the
wrong windows. Rebuild sequences + types together in one pass instead.

Place in : maritime/processing/
Run from : maritime/processing/
Output   : maritime/processing/incident_type_anomaly_test.npy
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter

WINDOW_SIZE = 30
STRIDE = 15

BASE_DIR = Path(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime")
DATA_DIR = BASE_DIR / "HawaiiCoast_GT"  # actual parquet location
PROCESSING_DIR = BASE_DIR / "processing"         # confirmed canonical copy (newer, has mmsi/norm files)

YEAR_PATHS = {
    '2017': DATA_DIR / "hawaii_2017.parquet",
    '2018': DATA_DIR / "hawaii_2018.parquet",
    '2019': DATA_DIR / "hawaii_2019.parquet",
    '2020': DATA_DIR / "hawaii_2020.parquet",
}


def extract_types(df):
    labels_any, types = [], []
    grouped = df.sort_values(['MMSI', 'datetime_hst']).groupby('MMSI')
    for mmsi, group in tqdm(grouped, total=df['MMSI'].nunique()):
        if not group['is_incident'].any():
            continue
        inc = group['is_incident'].values
        inc_type = group['ais_incident_type'].values
        T = len(inc)
        if T < WINDOW_SIZE:
            continue
        for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
            window_inc = inc[start:start + WINDOW_SIZE]
            label = int(window_inc.any())
            if label == 1:
                window_types = inc_type[start:start + WINDOW_SIZE]
                flagged = [t for t, f in zip(window_types, window_inc) if f and pd.notna(t)]
                wtype = Counter(flagged).most_common(1)[0][0] if flagged else "unknown_type"
            else:
                wtype = "normal_window"
            labels_any.append(label)
            types.append(wtype)
    return np.array(labels_any, dtype=np.int64), np.array(types, dtype=object)


def main():
    all_labels, all_types = [], []
    for year, path in YEAR_PATHS.items():
        df = pd.read_parquet(path)
        lab, typ = extract_types(df)
        print(f"{year}: {len(lab)} windows, {int(lab.sum())} positive")
        all_labels.append(lab)
        all_types.append(typ)

    labels_any = np.concatenate(all_labels)
    types_anomaly = np.concatenate(all_types)

    # --- Alignment check against existing y_anomaly_test.npy ---
    y_existing = np.load(PROCESSING_DIR / "y_anomaly_test.npy")

    if labels_any.shape[0] != y_existing.shape[0]:
        raise RuntimeError(
            f"Window count mismatch: rebuilt {labels_any.shape[0]} vs "
            f"saved {y_existing.shape[0]}. X_anomaly_test.npy was built from "
            f"different data or windowing logic than this script produces. "
            f"Do not trust the type array — stop and reconcile the two first."
        )
    if not np.array_equal(labels_any, y_existing):
        n_diff = int((labels_any != y_existing).sum())
        raise RuntimeError(
            f"Label mismatch on {n_diff}/{len(y_existing)} windows. "
            f"Same problem as above — these two arrays are not from the same run."
        )

    print("Alignment check passed: labels match y_anomaly_test.npy exactly.")
    np.save(PROCESSING_DIR / "incident_type_anomaly_test.npy", types_anomaly)
    print(f"Saved incident_type_anomaly_test.npy ({types_anomaly.shape[0]} entries)")


if __name__ == "__main__":
    main()
