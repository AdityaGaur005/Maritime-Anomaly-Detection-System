import json
from pathlib import Path

import numpy as np
import pandas as pd

# Paths to your yearly raw AIS parquet files (the ones with per-point data)
BASE_DIR = Path(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime")
DATA_DIR = BASE_DIR / "HawaiiCoast_GT"  # Adjust if needed

YEAR_PATHS = {
    '2017': DATA_DIR / "hawaii_2017.parquet",
    '2018': DATA_DIR / "hawaii_2018.parquet",
    '2019': DATA_DIR / "hawaii_2019.parquet",
    '2020': DATA_DIR / "hawaii_2020.parquet",
}

OUTPUT_PATH = BASE_DIR / "pop_fallback.json"  # Will be saved in maritime/

def main():
    all_normal = []
    for year, path in YEAR_PATHS.items():
        if not path.exists():
            print(f"Warning: {path} not found, skipping {year}")
            continue
        print(f"Loading {year}...")
        df = pd.read_parquet(path)
        # Keep only normal (non-incident) points
        normal = df[df['is_incident'] == 0]
        print(f"  Normal points: {len(normal):,}")
        all_normal.append(normal)

    if not all_normal:
        raise FileNotFoundError("No normal data found. Check your YEAR_PATHS.")

    # Concatenate all normal points
    df_all = pd.concat(all_normal, ignore_index=True)
    print(f"Total normal points: {len(df_all):,}")

    # Compute statistics exactly as build_hybrid_features.py did
    pop_fallback = {
        'speed_mean': float(df_all['speed_over_ground_knots'].mean()),
        'speed_std': float(df_all['speed_over_ground_knots'].std() + 1e-6),
        'heading_mean': float(df_all['heading_change_deg'].mean()),
        'heading_std': float(df_all['heading_change_deg'].std() + 1e-6),
        'lat_centroid': float(df_all['lat'].median()),
        'lon_centroid': float(df_all['lon'].median()),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(pop_fallback, f, indent=2)

    print(f"\nSaved pop_fallback to {OUTPUT_PATH}")
    print("Values:")
    for k, v in pop_fallback.items():
        print(f"  {k}: {v:.6f}")

if __name__ == "__main__":
    main()
