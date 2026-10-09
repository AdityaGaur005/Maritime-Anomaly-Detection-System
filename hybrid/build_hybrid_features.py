"""
Builds a flat per-window feature table for the XGBoost hybrid model.
Two modes:

  incident_vessels : all 4 years, vessels with >=1 incident point.
                      This is the population used for the grouped
                      train/val/test split in train_xgboost_hybrid.py.

  clean_2020        : 2020 vessels with ZERO incident points (the same
                      population as your existing normal test set).
                      Used only as an extra negative pool at eval time,
                      for a number directly comparable to the Transformer
                      /LSTM/Isolation-Forest AUC.

Uses RAW (unnormalized) feature values throughout — tree models don't need
scaling, and it keeps feature importances interpretable.

Place in : maritime/hybrid/   (new folder — keep this workstream separate
                                from processing/ and gaur/)
Run from : maritime/hybrid/
Usage    : python build_hybrid_features.py incident_vessels
           python build_hybrid_features.py clean_2020
Output   : maritime/hybrid/hybrid_features_<mode>.parquet
"""

import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import Counter

WINDOW_SIZE = 30
STRIDE = 15
EARTH_RADIUS_KM = 6371.0088

KINEMATIC_FEATURES = [
    'lat', 'lon',
    'speed_over_ground_knots',
    'course_over_ground_deg',
    'computed_speed_knots',
    'acceleration_knots_per_sec',
    'heading_change_deg',
]

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "HawaiiCoast_GT"
OUT_DIR = BASE_DIR / "hybrid"
OUT_DIR.mkdir(exist_ok=True)

YEAR_PATHS = {
    '2017': DATA_DIR / "hawaii_2017.parquet",
    '2018': DATA_DIR / "hawaii_2018.parquet",
    '2019': DATA_DIR / "hawaii_2019.parquet",
    '2020': DATA_DIR / "hawaii_2020.parquet",
}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def vessel_baseline(group):
    """Mean/std speed & heading_change, and centroid lat/lon, computed from
    this vessel's own NON-incident points. Falls back to population stats
    (passed in from caller) if the vessel has no non-incident points at all."""
    normal = group[group['is_incident'] == 0]
    if len(normal) == 0:
        return None  # caller fills in population fallback
    return {
        'speed_mean': normal['speed_over_ground_knots'].mean(),
        'speed_std': normal['speed_over_ground_knots'].std() + 1e-6,
        'heading_mean': normal['heading_change_deg'].mean(),
        'heading_std': normal['heading_change_deg'].std() + 1e-6,
        'lat_centroid': normal['lat'].median(),
        'lon_centroid': normal['lon'].median(),
    }


def build_windows(df, mode, pop_fallback):
    rows = []
    seqs = []  # raw (unnormalized) 30x7 windows, same order as rows -> used for deep-model scoring later
    grouped = df.sort_values(['MMSI', 'datetime_hst']).groupby('MMSI')

    for mmsi, group in tqdm(grouped, total=df['MMSI'].nunique()):
        has_incident = group['is_incident'].any()

        if mode == 'incident_vessels' and not has_incident:
            continue
        if mode == 'clean_2020' and has_incident:
            continue

        baseline = vessel_baseline(group)
        is_fallback = baseline is None
        if is_fallback:
            baseline = pop_fallback

        data = group[KINEMATIC_FEATURES].values
        inc = group['is_incident'].values
        inc_type = group['ais_incident_type'].values if 'ais_incident_type' in group else None
        dt = group['delta_time_sec'].values
        lat_arr = group['lat'].values
        lon_arr = group['lon'].values
        vessel_type = group['vessel_type_code'].iloc[0]
        length_m = group['length_m'].iloc[0]
        width_m = group['width_m'].iloc[0]

        T = data.shape[0]
        if T < WINDOW_SIZE:
            continue

        for start in range(0, T - WINDOW_SIZE + 1, STRIDE):
            w = data[start:start + WINDOW_SIZE]
            w_inc = inc[start:start + WINDOW_SIZE]
            w_dt = dt[start:start + WINDOW_SIZE]
            w_lat = lat_arr[start:start + WINDOW_SIZE]
            w_lon = lon_arr[start:start + WINDOW_SIZE]

            label = int(w_inc.any())
            if label == 1 and inc_type is not None:
                w_type = inc_type[start:start + WINDOW_SIZE]
                flagged = [t for t, f in zip(w_type, w_inc) if f and pd.notna(t)]
                wtype = Counter(flagged).most_common(1)[0][0] if flagged else "unknown_type"
            else:
                wtype = "normal_window"

            feat = {'MMSI': mmsi, 'label': label, 'incident_type': wtype,
                    'baseline_is_fallback': int(is_fallback),
                    'vessel_type_code': vessel_type,
                    'length_m': length_m, 'width_m': width_m}

            for i, col in enumerate(KINEMATIC_FEATURES):
                v = w[:, i]
                feat[f'{col}_mean'] = v.mean()
                feat[f'{col}_std'] = v.std()
                feat[f'{col}_min'] = v.min()
                feat[f'{col}_max'] = v.max()

            feat['delta_time_mean'] = np.nanmean(w_dt)
            feat['delta_time_max'] = np.nanmax(w_dt)
            feat['delta_time_std'] = np.nanstd(w_dt)

            speed_mean_w = w[:, 2].mean()  # speed_over_ground_knots
            heading_mean_w = w[:, 6].mean()  # heading_change_deg
            feat['speed_zscore_vs_vessel'] = (speed_mean_w - baseline['speed_mean']) / baseline['speed_std']
            feat['heading_zscore_vs_vessel'] = (heading_mean_w - baseline['heading_mean']) / baseline['heading_std']

            dist_from_centroid = haversine_km(w_lat, w_lon, baseline['lat_centroid'], baseline['lon_centroid'])
            feat['dist_from_vessel_centroid_km_mean'] = dist_from_centroid.mean()
            feat['dist_from_vessel_centroid_km_max'] = dist_from_centroid.max()

            rows.append(feat)
            seqs.append(w)  # raw, unnormalized — normalize at scoring time to match training norm stats

    return pd.DataFrame(rows), np.array(seqs, dtype=np.float32)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('incident_vessels', 'clean_2020'):
        print("Usage: python build_hybrid_features.py [incident_vessels|clean_2020]")
        sys.exit(1)
    mode = sys.argv[1]

    years = YEAR_PATHS.items() if mode == 'incident_vessels' else [('2020', YEAR_PATHS['2020'])]

    all_dfs, all_seqs = [], []
    for year, path in years:
        df = pd.read_parquet(path)
        # population-level fallback baseline, used only for the rare vessel with zero non-incident points
        normal_pop = df[df['is_incident'] == 0]
        pop_fallback = {
            'speed_mean': normal_pop['speed_over_ground_knots'].mean(),
            'speed_std': normal_pop['speed_over_ground_knots'].std() + 1e-6,
            'heading_mean': normal_pop['heading_change_deg'].mean(),
            'heading_std': normal_pop['heading_change_deg'].std() + 1e-6,
            'lat_centroid': normal_pop['lat'].median(),
            'lon_centroid': normal_pop['lon'].median(),
        }
        table, seq_arr = build_windows(df, mode, pop_fallback)
        table['year'] = year
        print(f"{year}: {len(table)} windows, {int(table['label'].sum())} positive")
        all_dfs.append(table)
        all_seqs.append(seq_arr)

    full = pd.concat(all_dfs, ignore_index=True)
    full_seqs = np.concatenate(all_seqs, axis=0)
    assert len(full) == len(full_seqs), "Row count mismatch between feature table and sequence array"

    out_path = OUT_DIR / f"hybrid_features_{mode}.parquet"
    full.to_parquet(out_path, index=False)
    seq_path = OUT_DIR / f"hybrid_sequences_{mode}.npy"
    np.save(seq_path, full_seqs)
    print(f"\nSaved {out_path} — {len(full)} rows, {full.shape[1]} columns")
    print(f"Saved {seq_path} — shape {full_seqs.shape} (raw, unnormalized — for deep-model scoring)")
    print(f"Unique vessels: {full['MMSI'].nunique()}")
    if mode == 'incident_vessels':
        print(f"Total positives: {int(full['label'].sum())} ({full['label'].mean()*100:.3f}%)")
        print(f"Fallback baseline used on {int(full['baseline_is_fallback'].sum())} windows")


if __name__ == "__main__":
    main()