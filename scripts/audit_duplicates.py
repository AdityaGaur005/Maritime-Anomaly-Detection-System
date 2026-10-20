# audit_duplicates.py — run this from maritime/, it's read-only, just reports
from pathlib import Path
import numpy as np
from collections import defaultdict

ROOT = Path(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime")
targets = {"X_anomaly_test.npy", "y_anomaly_test.npy", "mmsi_anomaly_test.npy",
           "norm_mean.npy", "norm_std.npy", "X_normal_test_norm.npy",
           "X_anomaly_test_norm.npy"}

found = defaultdict(list)
for p in ROOT.rglob("*.npy"):
    if p.name in targets:
        found[p.name].append(p)

for name, paths in found.items():
    print(f"\n{name}: {len(paths)} copies")
    for p in paths:
        st = p.stat()
        arr = np.load(p, allow_pickle=True)
        print(f"  {p}\n    shape={arr.shape} dtype={arr.dtype} modified={pd.Timestamp(st.st_mtime, unit='s') if False else st.st_mtime}")



# compare_duplicates.py — run from maritime/
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime")
pairs = [
    ("model/X_anomaly_test.npy", "processing/X_anomaly_test.npy"),
    ("model/X_anomaly_test_norm.npy", "processing/X_anomaly_test_norm.npy"),
    ("model/X_normal_test_norm.npy", "processing/X_normal_test_norm.npy"),
    ("model/y_anomaly_test.npy", "processing/y_anomaly_test.npy"),
]

for a, b in pairs:
    pa, pb = ROOT / a, ROOT / b
    arr_a, arr_b = np.load(pa), np.load(pb)
    identical = np.array_equal(arr_a, arr_b)
    print(f"{a}  vs  {b}: {'IDENTICAL' if identical else 'DIFFERENT'}")
    if not identical:
        diff = arr_a != arr_b
        print(f"  {diff.sum()} / {arr_a.size} elements differ")