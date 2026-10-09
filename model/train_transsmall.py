import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import mlflow
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
from model import TransformerVAE, vae_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EPOCHS = 20
LR = 1e-3
KL_WEIGHT = 0.01

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
            batch = torch.from_numpy(X[i:i + batch_size]).float().to(DEVICE)
            recon, mu, logvar = model(batch)
            err = torch.mean((recon - batch) ** 2, dim=[1, 2])
            errors.append(err.cpu().numpy())
    return np.concatenate(errors)


def evaluate_anomaly_detection(model, X_normal_test, X_anomaly_test, y_anomaly_test):
    normal_err = per_sample_recon_error(model, X_normal_test)
    anomaly_err = per_sample_recon_error(model, X_anomaly_test)

    # Anomaly test set already contains a mix of normal/anomalous windows (y labels).
    # Score = reconstruction error; higher = more anomalous.
    y_true = y_anomaly_test
    y_score = anomaly_err
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    return {
        "held_out_normal_mean_error": float(normal_err.mean()),
        "labeled_test_roc_auc": float(auc),
        "labeled_test_avg_precision": float(ap),
    }


def main():
    X_train, X_val, X_normal_test, X_anomaly_test, y_anomaly_test = load_data()
    train_loader = make_loader(X_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, BATCH_SIZE, shuffle=False)

    model = TransformerVAE(n_features=X_train.shape[2], seq_len=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    BASE_DIR = Path(__file__).resolve().parent
    db_path = BASE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment("maritime-anomaly-transformer-vae")
    with mlflow.start_run():
        mlflow.log_params({
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lr": LR,
            "kl_weight": KL_WEIGHT, "train_size": len(X_train), "val_size": len(X_val)
        })

        best_val_loss = float("inf")
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_losses = []
            for (batch,) in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                recon, mu, logvar = model(batch)
                loss, recon_l, kl_l = vae_loss(recon, batch, mu, logvar, KL_WEIGHT)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            val_losses = []
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(DEVICE)
                    recon, mu, logvar = model(batch)
                    loss, _, _ = vae_loss(recon, batch, mu, logvar, KL_WEIGHT)
                    val_losses.append(loss.item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pt")
                mlflow.log_artifact("best_model.pt")

        # Final evaluation on labeled anomaly test set
        model.load_state_dict(torch.load("best_model.pt"))
        metrics = evaluate_anomaly_detection(model, X_normal_test, X_anomaly_test, y_anomaly_test)
        print(metrics)
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
