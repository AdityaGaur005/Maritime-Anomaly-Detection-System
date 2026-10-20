"""
One-off script to verify that two normalization arrays are equivalent.
Run from the repo root: python model/test.py
"""
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
MODEL_DIR = Path(__file__).resolve().parent

a = np.load(BASE_DIR / "model" / "X_train.npy")
b = np.load(BASE_DIR / "model" / "X_train_norm.npy")
c = np.load(BASE_DIR / "model" / "X_train_full.npy")
print(a.shape, b.shape, c.shape)
print(np.allclose(a[:1000], b[:1000]))
print(np.allclose(a[:1000], c[:1000]))
print(np.allclose(b[:1000], c[:1000]))
print(a.mean(), a.std())
print(b.mean(), b.std())
print(c.mean(), c.std())
