"""
LSTM Autoencoder for maritime AIS anomaly detection.
Matches the existing pipeline: 30-step windows, 7 feature******pected files in DATA_DIR:
    X_train.npy               -> (N_train, 30, 7), already normalized
    X_val.npy                 -> (N_val, 30, 7), already normalized
    X_normal_test_norm.npy    -> (N_normal_test, 30, 7)
    X_anomaly_test_norm.npy   -> (N_anomaly_test, 30, 7)
    y_anomaly_test.npy        -> (N_anomaly_test,) labels for the anomaly test windows
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import json
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR
OUT_DIR = BASE_DIR / "lstm_ae_run"

OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "seq_len": 30,
    "n_features": 7,
    "hidden_dim": 128,
    "latent_dim": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 80,
    "patience": 7,
    "grad_clip": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    """
    Encoder LSTM compresses the 30-step window into a fixed latent vector
    (final hidden state -> linear bottleneck). Decoder LSTM reconstructs
    the sequence from that latent vector, repeated at every time step
    (standard seq2seq-AE pattern; avoids teacher forcing so it's directly
    comparable to the Transformer-VAE's non-autoregressive decoding).
    """

    def __init__(self, n_features, hidden_dim, latent_dim, num_layers, dropout):
        super().__init__()
        self.seq_len = CONFIG["seq_len"]

        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # x: (B, T, F)
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]                      # (B, hidden_dim) - top layer's final hidden state
        z = self.to_latent(h_last)            # (B, latent_dim)

        dec_input = self.from_latent(z)                      # (B, hidden_dim)
        dec_input = dec_input.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, hidden_dim)

        dec_out, _ = self.decoder_lstm(dec_input)             # (B, T, hidden_dim)
        recon = self.output_proj(dec_out)                     # (B, T, F)
        return recon, z


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    X_train = np.load(DATA_DIR / "X_train.npy").astype(np.float32)
    X_val = np.load(DATA_DIR / "X_val.npy").astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train))
    val_ds = TensorDataset(torch.from_numpy(X_val))

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, n_batches = 0.0, 0

    with torch.set_grad_enabled(is_train):
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            recon, _ = model(x)
            loss = criterion(recon, x)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train():
    device = CONFIG["device"]
    print(f"Device: {device}")

    train_loader, val_loader = load_data()

    model = LSTMAutoencoder(
        n_features=CONFIG["n_features"],
        hidden_dim=CONFIG["hidden_dim"],
        latent_dim=CONFIG["latent_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | train_loss={train_loss:.5f} | "
              f"val_loss={val_loss:.5f} | lr={optimizer.param_groups[0]['lr']:.1e} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), OUT_DIR / "lstm_ae_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for {CONFIG['patience']} epochs).")
                break

    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump({k: v for k, v in CONFIG.items() if k != "device"}, f, indent=2)

    print(f"Best val_loss: {best_val_loss:.5f}. Saved to {OUT_DIR/'lstm_ae_best.pt'}")
    return model


# ---------------------------------------------------------------------------
# Evaluation on labeled test set (per-window reconstruction error -> AUC/AP)
# ---------------------------------------------------------------------------
def evaluate_on_test():
    device = CONFIG["device"]
    normal_path = DATA_DIR / "X_normal_test_norm.npy"
    anomaly_path = DATA_DIR / "X_anomaly_test_norm.npy"
    anomaly_label_path = DATA_DIR / "y_anomaly_test.npy"
    if not (normal_path.exists() and anomaly_path.exists() and anomaly_label_path.exists()):
        print("X_normal_test_norm.npy / X_anomaly_test_norm.npy / y_anomaly_test.npy "
              "not found — skipping AUC/AP evaluation.")
        return

    X_normal = np.load(normal_path).astype(np.float32)
    X_anomaly = np.load(anomaly_path).astype(np.float32)
    y_anomaly = np.load(anomaly_label_path).astype(int)

    # y_anomaly_test.npy should already be all 1s (or incident-type codes > 0).
    # Sanity check rather than assume — mismatched label conventions are a
    # common silent bug here (e.g. if it's incident-type codes, not a binary flag).
    assert y_anomaly.shape[0] == X_anomaly.shape[0], \
        f"Label count {y_anomaly.shape[0]} != anomaly window count {X_anomaly.shape[0]}"
    if not np.all(y_anomaly > 0):
        print(f"WARNING: {np.sum(y_anomaly == 0)} entries in y_anomaly_test.npy are 0/non-positive. "
              "Check this file actually only contains anomaly labels.")

    y_normal = np.zeros(X_normal.shape[0], dtype=int)
    y_anomaly_binary = (y_anomaly > 0).astype(int)  # collapse incident-type codes to binary if needed

    X_test = np.concatenate([X_normal, X_anomaly], axis=0)
    y_test = np.concatenate([y_normal, y_anomaly_binary], axis=0)

    print(f"Test set: {X_normal.shape[0]} normal windows, {X_anomaly.shape[0]} anomaly windows "
          f"({y_test.mean()*100:.2f}% positive)")

    model = LSTMAutoencoder(
        n_features=CONFIG["n_features"],
        hidden_dim=CONFIG["hidden_dim"],
        latent_dim=CONFIG["latent_dim"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(OUT_DIR / "lstm_ae_best.pt", map_location=device))
    model.eval()

    test_ds = TensorDataset(torch.from_numpy(X_test))
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False)

    per_window_errors = []
    with torch.no_grad():
        for (x,) in test_loader:
            x = x.to(device)
            recon, _ = model(x)
            # per-window error = mean squared error over time steps and features
            err = ((recon - x) ** 2).mean(dim=[1, 2])
            per_window_errors.append(err.cpu().numpy())

    scores = np.concatenate(per_window_errors)

    auc = roc_auc_score(y_test, scores)
    ap = average_precision_score(y_test, scores)
    normal_err_mean = scores[y_test == 0].mean()
    anomaly_err_mean = scores[y_test == 1].mean() if (y_test == 1).any() else float("nan")

    print("\n--- Test Set Evaluation ---")
    print(f"ROC-AUC:              {auc:.3f}")
    print(f"Average Precision:    {ap:.3f}")
    print(f"Normal error (mean):  {normal_err_mean:.4f}")
    print(f"Anomaly error (mean): {anomaly_err_mean:.4f}")

    np.save(OUT_DIR / "test_scores.npy", scores)  # reuse as features for the hybrid XGBoost stage
    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump({"roc_auc": auc, "average_precision": ap,
                    "normal_err_mean": float(normal_err_mean),
                    "anomaly_err_mean": float(anomaly_err_mean)}, f, indent=2)


if __name__ == "__main__":
    train()
    evaluate_on_test()
