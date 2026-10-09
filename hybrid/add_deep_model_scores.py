"""
Adds Transformer-VAE (and optionally LSTM-AE) reconstruction-error features
to the hybrid feature table built by build_hybrid_features.py.

For each model, adds 8 columns per window:
  <model>_err_<channel>   (7 columns — per-channel squared error, mean over
                            the 30 timesteps; tells XGBoost WHICH channel the
                            model found unusual, not just how unusual overall)
  <model>_err_agg          (1 column — the same aggregate MSE the reconstruction
                            models were evaluated on originally)

Does not touch the raw parquet — normalizes the saved raw sequences with the
same norm_mean/norm_std used to train the deep models, so scores are on the
scale those models actually learned.

Place in : maritime/hybrid/
Run from : maritime/hybrid/
Usage    : python add_deep_model_scores.py incident_vessels
           python add_deep_model_scores.py clean_2020
Requires : build_hybrid_features.py already run for that mode.
Output   : maritime/hybrid/hybrid_features_<mode>_with_scores.parquet
"""

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
HYBRID_DIR = BASE_DIR / "hybrid"
PROCESSING_DIR = BASE_DIR / "processing"
MODEL_DIR = BASE_DIR / "model"

sys.path.insert(0, str(MODEL_DIR))
from model import TransformerVAE                      # noqa: E402
from train_lstm_ae import LSTMAutoencoder, CONFIG      # noqa: E402

CHANNELS = ['lat', 'lon', 'speed', 'course', 'computed_speed', 'acceleration', 'heading_change']

TRANS_CKPT = MODEL_DIR / "best_model_large.pt"
LSTM_CKPT = MODEL_DIR / "lstm_ae_run" / "lstm_ae_best.pt"
NORM_MEAN_PATH = PROCESSING_DIR / "norm_mean.npy"
NORM_STD_PATH = PROCESSING_DIR / "norm_std.npy"

D_MODEL, NHEAD, NUM_LAYERS, LATENT_DIM, DIM_FEEDFORWARD, DROPOUT = 256, 8, 6, 64, 512, 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transformer(n_features, seq_len):
    model = TransformerVAE(
        n_features=n_features, seq_len=seq_len, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, latent_dim=LATENT_DIM,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(TRANS_CKPT, map_location=device))
    model.eval()
    return model


def load_lstm():
    model = LSTMAutoencoder(
        n_features=CONFIG["n_features"], hidden_dim=CONFIG["hidden_dim"],
        latent_dim=CONFIG["latent_dim"], num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(LSTM_CKPT, map_location=device))
    model.eval()
    return model


def score_per_channel(model, X, reconstruct_fn, batch_size=128):
    """Returns (n_windows, 7) per-channel squared error, mean over the 30 timesteps.
    reconstruct_fn(model, x) -> recon must be DETERMINISTIC (no VAE sampling)."""
    all_err = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            recon = reconstruct_fn(model, x)
            err = ((recon - x) ** 2).mean(dim=1)  # mean over time -> (batch, 7)
            all_err.append(err.cpu().numpy())
            if i % (batch_size * 200) == 0:
                print(f"  {min(i + batch_size, len(X)):,}/{len(X):,}", flush=True)
    return np.concatenate(all_err, axis=0)


def transformer_reconstruct(model, x):
    """Deterministic: decode from mu directly, bypassing reparameterize()'s
    torch.randn_like() sampling. model.eval() does NOT disable that sampling —
    it's an explicit call in reparameterize(), not gated on self.training."""
    mu, logvar = model.encode(x)
    return model.decode(mu)


def lstm_reconstruct(model, x):
    recon, _ = model(x)
    return recon


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ('incident_vessels', 'clean_2020'):
        print("Usage: python add_deep_model_scores.py [incident_vessels|clean_2020]")
        sys.exit(1)
    mode = sys.argv[1]

    print("Loading feature table + raw sequences...")
    df = pd.read_parquet(HYBRID_DIR / f"hybrid_features_{mode}.parquet")
    X_raw = np.load(HYBRID_DIR / f"hybrid_sequences_{mode}.npy")
    assert len(df) == len(X_raw), "Row count mismatch — rerun build_hybrid_features.py"

    mean = np.load(NORM_MEAN_PATH)
    std = np.load(NORM_STD_PATH)
    X_norm = ((X_raw - mean) / (std + 1e-8)).astype(np.float32)

    n_features, seq_len = X_norm.shape[2], X_norm.shape[1]

    print("\nScoring with Transformer-VAE (deterministic, from mu)...")
    trans_model = load_transformer(n_features, seq_len)
    trans_err = score_per_channel(trans_model, X_norm, transformer_reconstruct)
    for i, ch in enumerate(CHANNELS):
        df[f'trans_err_{ch}'] = trans_err[:, i]
    df['trans_err_agg'] = trans_err.mean(axis=1)
    del trans_model
    torch.cuda.empty_cache()

    print("\nScoring with LSTM-AE...")
    lstm_model = load_lstm()
    lstm_err = score_per_channel(lstm_model, X_norm, lstm_reconstruct)
    for i, ch in enumerate(CHANNELS):
        df[f'lstm_err_{ch}'] = lstm_err[:, i]
    df['lstm_err_agg'] = lstm_err.mean(axis=1)

    out_path = HYBRID_DIR / f"hybrid_features_{mode}_with_scores.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {out_path} — {df.shape[1]} columns total (added 16: 7+1 Transformer, 7+1 LSTM)")


if __name__ == "__main__":
    main()