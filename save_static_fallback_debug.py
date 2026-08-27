import pandas as pd
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime")
DATA_DIR = BASE_DIR / "HawaiiCoast_GT"
OUTPUT_PATH = BASE_DIR / "static_fallback.json"

YEAR_PATHS = {
    '2017': DATA_DIR / "hawaii_2017.parquet",
    '2018': DATA_DIR / "hawaii_2018.parquet",
    '2019': DATA_DIR / "hawaii_2019.parquet",
    '2020': DATA_DIR / "hawaii_2020.parquet",
}

def main():
    all_normal = []
    for year, path in YEAR_PATHS.items():
        if not path.exists():
            print(f"Warning: {path} not found, skipping {year}")
            continue
        print(f"Loading {year}...")
        df = pd.read_parquet(path)
        # Keep only non-incident points
        normal = df[df['is_incident'] == 0]
        # Ensure columns exist
        required = ['vessel_type_code', 'length_m', 'width_m']
        for col in required:
            if col not in normal.columns:
                print(f"ERROR: Column '{col}' missing in {year}")
                return
        all_normal.append(normal[required])
    
    if not all_normal:
        print("No data loaded. Check YEAR_PATHS.")
        return
    
    df_all = pd.concat(all_normal, ignore_index=True)
    print(f"Total normal points: {len(df_all):,}")
    
    # Compute fallback values with safe handling
    # Mode for vessel_type_code (most common type)
    mode_vals = df_all['vessel_type_code'].mode()
    if not mode_vals.empty:
        vessel_type_mode = int(mode_vals[0])
    else:
        vessel_type_mode = 0  # default
        print("WARNING: vessel_type_code all NaN, using 0")
    
    static_fallback = {
        'vessel_type_code': vessel_type_mode,
        'length_m': float(df_all['length_m'].median()),
        'width_m': float(df_all['width_m'].median()),
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(static_fallback, f, indent=2)
    
    print(f"Saved static_fallback to {OUTPUT_PATH}")
    print("Values:", static_fallback)

if __name__ == "__main__":
    main()