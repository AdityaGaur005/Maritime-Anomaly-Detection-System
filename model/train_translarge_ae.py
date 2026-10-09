import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import mlflow
from sklearn.metrics import roc_auc_score, average_precision_score
from model import TransformerVAE, vae_loss

BASE_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# CONFIGURATION (Scaled Up)
# ------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256               # reduced from 512 for larger model
EPOCHS = 50
LR = 1e-4
KL_WEIGHT = 0.01               # final KL weight (we'll anneal to this)
KL_ANNEAL_EPOCHS = 20          # ramp KL from 0 to KL_WEIGHT over 20 epochs

# Larger model parameters
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 6
LATENT_DIM = 64
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
def load_data():
    X_train = np.load(BASE_DIR / "X_train.npy")
    X_val = np.load(BASE_DIR / "X_val.npy")
    X_normal_test = np.load(BASE_DIR / "X_normal_test_norm.npy")
    X_anomaly_test = np.load(BASE_DIR / "X_anomaly_test_norm.npy")
    y_anomaly_test = np.load(BASE_DIR / "y_anomaly_test.npy")
    return X_train, X_val, X_normal_test, X_anomaly_test, y_anomaly_test

def make_loader(X, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2)

def per_sample_recon_error(model, X, batch_size=1024):
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i+batch_size]).float().to(DEVICE)
            recon, mu, logvar = model(batch)
            err = torch.mean((recon - batch) ** 2, dim=[1, 2])
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)

def evaluate_anomaly_detection(model, X_normal_test, X_anomaly_test, y_anomaly_test):
    normal_err = per_sample_recon_error(model, X_normal_test)
    anomaly_err = per_sample_recon_error(model, X_anomaly_test)
    y_true = y_anomaly_test
    y_score = anomaly_err
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return {
        "held_out_normal_mean_error": float(normal_err.mean()),
        "labeled_test_roc_auc": float(auc),
        "labeled_test_avg_precision": float(ap),
    }

# ------------------------------------------------------------
# KL ANNEALING
# ------------------------------------------------------------
def get_kl_weight(epoch, max_kl=KL_WEIGHT, warmup=KL_ANNEAL_EPOCHS):
    if epoch < warmup:
        return max_kl * (epoch / warmup)
    return max_kl

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    X_train, X_val, X_normal_test, X_anomaly_test, y_anomaly_test = load_data()
    train_loader = make_loader(X_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, BATCH_SIZE, shuffle=False)

    # Initialize larger model
    model = TransformerVAE(
        n_features=X_train.shape[2],
        seq_len=X_train.shape[1],
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        latent_dim=LATENT_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(DEVICE)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, min_lr=1e-6
    )

    # MLflow tracking
    db_path = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment("maritime-anomaly-transformer-vae-large")

    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "kl_weight": KL_WEIGHT,
            "kl_anneal_epochs": KL_ANNEAL_EPOCHS,
            "d_model": D_MODEL,
            "nhead": NHEAD,
            "num_layers": NUM_LAYERS,
            "latent_dim": LATENT_DIM,
            "dim_feedforward": DIM_FEEDFORWARD,
            "dropout": DROPOUT,
            "train_size": len(X_train),
            "val_size": len(X_val)
        })

        best_val_loss = float("inf")
        patience_counter = 0
        PATIENCE = 10

        for epoch in range(1, EPOCHS + 1):
            # ---- Anneal KL ----
            current_kl = get_kl_weight(epoch)

            # ---- Training ----
            model.train()
            train_losses = []
            for (batch,) in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                recon, mu, logvar = model(batch)
                loss, recon_l, kl_l = vae_loss(recon, batch, mu, logvar, current_kl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # ---- Validation ----
            model.eval()
            val_losses = []
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(DEVICE)
                    recon, mu, logvar = model(batch)
                    loss, _, _ = vae_loss(recon, batch, mu, logvar, current_kl)
                    val_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))

            print(f"Epoch {epoch:3d}/{EPOCHS} | KL={current_kl:.4f} | train={train_loss:.5f} | val={val_loss:.5f}")
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "kl_weight": current_kl
            }, step=epoch)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            mlflow.log_metric("lr", current_lr, step=epoch)

            # ---- Checkpoint & Early Stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model_large.pt")
                mlflow.log_artifact("best_model_large.pt")
                print(f"  ✓ New best saved (val_loss={best_val_loss:.5f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        # ---- Final Evaluation ----
        model.load_state_dict(torch.load("best_model_large.pt"))
        metrics = evaluate_anomaly_detection(model, X_normal_test, X_anomaly_test, y_anomaly_test)
        print("\n=== FINAL EVALUATION ===")
        print(f"ROC-AUC: {metrics['labeled_test_roc_auc']:.4f}")
        print(f"Avg Precision: {metrics['labeled_test_avg_precision']:.4f}")
        mlflow.log_metrics(metrics)

if __name__ == "__main__":
    main()
