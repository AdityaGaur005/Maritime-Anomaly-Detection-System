"""
Per-incident-type AUC/AP for the LSTM-AE, using arrays that already exist
(X_normal_test_norm.npy, X_anomaly_test_norm.npy, y_anomaly_test.npy from
create_test_sequences.py + incident_type_anomaly_test.npy from
save_incident_types.py). Does NOT touch the raw parquet.

Place in : maritime/gaur/   (same folder as train_lstm_ae.py, needed for import)
Run from : maritime/gaur/
Requires : run save_incident_types.py first.
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

from train_lstm_ae import LSTMAutoencoder, CONFIG

BASE_DIR = Path(r"C:\Users\Aditya Gaur\Downloads\.vscode\maritime")
PROCESSING_DIR = BASE_DIR / "processing"  # confirmed canonical copy (newer, has mmsi/norm files)
CKPT_PATH = BASE_DIR / "gaur" / "lstm_ae_run" / "lstm_ae_best.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"


def score_with_lstm_ae(X, batch_size=512):
    model = LSTMAutoencoder(
        n_features=CONFIG["n_features"], hidden_dim=CONFIG["hidden_dim"],
        latent_dim=CONFIG["latent_dim"], num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i + batch_size]).float().to(device)
            recon, _ = model(x)
            err = ((recon - x) ** 2).mean(dim=[1, 2])
            scores.append(err.cpu().numpy())
    return np.concatenate(scores)


def main():
    X_normal = np.load(PROCESSING_DIR / "X_normal_test_norm.npy").astype(np.float32)
    X_anomaly = np.load(PROCESSING_DIR / "X_anomaly_test_norm.npy").astype(np.float32)
    y_anomaly = np.load(PROCESSING_DIR / "y_anomaly_test.npy")
    types_anomaly = np.load(PROCESSING_DIR / "incident_type_anomaly_test.npy", allow_pickle=True)

    assert X_anomaly.shape[0] == y_anomaly.shape[0] == types_anomaly.shape[0], \
        "Array length mismatch — rerun save_incident_types.py and check its alignment assertion."

    print("Scoring with LSTM-AE...")
    normal_scores = score_with_lstm_ae(X_normal)
    anomaly_scores = score_with_lstm_ae(X_anomaly)

    pos_mask = y_anomaly == 1
    pos_scores = anomaly_scores[pos_mask]
    pos_types = types_anomaly[pos_mask]

    print(f"Normal test windows: {len(normal_scores)}")
    print(f"Positive anomaly windows: {int(pos_mask.sum())}")

    print("\n=== Per-Incident-Type Performance (vs. full normal negative pool) ===")
    unique_types = sorted(set(pos_types))
    results = []
    for t in unique_types:
        type_scores = pos_scores[pos_types == t]
        n_pos = len(type_scores)
        if n_pos < 5:
            print(f"{t:25s} n={n_pos:5d}  (too few positives, skipped)")
            continue
        y_true = np.concatenate([np.zeros(len(normal_scores)), np.ones(n_pos)])
        y_score = np.concatenate([normal_scores, type_scores])
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        results.append((t, n_pos, auc, ap))
        print(f"{t:25s} n={n_pos:5d}  AUC={auc:.3f}  AP={ap:.4f}")

    y_true_all = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(pos_scores))])
    y_score_all = np.concatenate([normal_scores, pos_scores])
    overall_auc = roc_auc_score(y_true_all, y_score_all)
    overall_ap = average_precision_score(y_true_all, y_score_all)
    print(f"\n{'OVERALL':25s} n={len(pos_scores):5d}  AUC={overall_auc:.3f}  AP={overall_ap:.4f}")

    if results:
        best = max(results, key=lambda r: r[2])
        worst = min(results, key=lambda r: r[2])
        spread = best[2] - worst[2]
        print(f"\nBest-detected type:  {best[0]} (AUC={best[2]:.3f}, n={best[1]})")
        print(f"Worst-detected type: {worst[0]} (AUC={worst[2]:.3f}, n={worst[1]})")
        print(f"AUC spread across types: {spread:.3f}")
        if spread > 0.15:
            print("-> Large spread: performance is NOT uniform across incident types.")
        else:
            print("-> Small spread: AUC ceiling looks uniform — likely a feature/task-level limitation.")


if __name__ == "__main__":
    main()
