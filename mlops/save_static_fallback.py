import pandas as pd
import json
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
        normal = df[df['is_incident'] == 0]
        all_normal.append(normal[['vessel_type_code', 'length_m', 'width_m']])
    
    df_all = pd.concat(all_normal, ignore_index=True)
    
    static_fallback = {
        'vessel_type_code': int(df_all['vessel_type_code'].mode()[0]) if not df_all['vessel_type_code'].isna().all() else 0,
        'length_m': float(df_all['length_m'].median()),
        'width_m': float(df_all['width_m'].median()),
    }
    
    with open(OUTPUT_PATH, "w") as f:
        json.dump(static_fallback, f, indent=2)
    
    print(f"Saved static_fallback to {OUTPUT_PATH}")
    print(static_fallback)

if __name__ == "__main__":
    main()