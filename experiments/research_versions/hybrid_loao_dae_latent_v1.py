# =========================================================
# HYBRID LOAO - DAE + LATENT DISTANCE (GPU SAFE VERSION)
# Memory Optimized | Single Run | DDoS + Infiltration
# best for ddso and infilteration till now, but not as good for portscan (maybe due to latent distance?)
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocess_cicids import load_clean_cicids
import random

device = torch.device( "cpu")
print("Using device:", device)


# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()
df.columns = df.columns.str.strip()
eps = 1e-6

if "Flow Duration" in df.columns:
    df["Packets_Per_Second"] = df["Total Fwd Packets"] / (df["Flow Duration"].abs() + eps)
    df["Bytes_Per_Second"] = df["Total Length of Fwd Packets"] / (df["Flow Duration"].abs() + eps)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].clip(-1e9, 1e9)

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ATTACKS = ["DDoS", "Infiltration", "PortScan", "Bot"]

# =========================================================
# LOOP
# =========================================================

for ZERO_DAY in ATTACKS:

    print("\n" + "="*70)
    print("ZERO-DAY:", ZERO_DAY)
    print("="*70)

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    train_df = df[df["Label"] != ZERO_DAY]
    test_zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))

    # =====================================================
    # DENOISING AE
    # =====================================================

    class DAE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 8)
            )

            self.decoder = nn.Sequential(
                nn.Linear(8, 64),
                nn.ReLU(),
                nn.Linear(64, 256),
                nn.ReLU(),
                nn.Linear(256, dim)
            )

        def forward(self, x):
            latent = self.encoder(x)
            recon = self.decoder(latent)
            return recon, latent

    model = DAE(X_benign.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
        batch_size=4096,
        shuffle=True
    )

    # =====================================================
    # TRAIN
    # =====================================================

    model.train()
    for epoch in range(20):
        for batch in loader:
            x = batch[0].to(device)
            noise = torch.randn_like(x) * 0.05
            noisy_x = x + noise

            recon, _ = model(noisy_x)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    # =====================================================
    # LATENT CENTROID
    # =====================================================

    latent_list = []
    recon_list = []

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            recon, latent = model(x)

            latent_list.append(latent.cpu())
            recon_list.append(torch.mean((recon - x)**2, dim=1).cpu())

    latent_vectors = torch.cat(latent_list).numpy()
    recon_errors = torch.cat(recon_list).numpy()

    latent_centroid = np.mean(latent_vectors, axis=0)
    threshold = np.percentile(recon_errors, 97)

    print("Anomaly Threshold:", round(threshold, 6))

    # =====================================================
    # RF TRAINING (BATCHED RESIDUAL COMPUTATION)
    # =====================================================

    rf_sample = train_df.sample(250000, random_state=seed)
    X_rf = scaler.transform(rf_sample.drop("Label", axis=1))
    y_rf = rf_sample["Label"]

    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)
    rf_loader = DataLoader(TensorDataset(X_rf_tensor), batch_size=4096)

    residual_list = []
    latent_list = []

    with torch.no_grad():
        for batch in rf_loader:
            x = batch[0].to(device)
            recon, latent = model(x)

            residual_list.append(torch.mean((recon - x)**2, dim=1).cpu())
            latent_list.append(latent.cpu())

    residual = torch.cat(residual_list).numpy()
    latent_np = torch.cat(latent_list).numpy()

    latent_dist = np.linalg.norm(latent_np - latent_centroid, axis=1)
    anomaly_score = 0.6 * residual + 0.4 * latent_dist

    X_rf_final = np.hstack([X_rf, anomaly_score.reshape(-1, 1)])

    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_rf_final, y_rf)

    # =====================================================
    # EVALUATION (BATCHED)
    # =====================================================

    eval_df = pd.concat([
        train_df.sample(60000, random_state=seed),
        test_zero_df.sample(min(10000, len(test_zero_df)), random_state=seed)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
    eval_loader = DataLoader(TensorDataset(X_eval_tensor), batch_size=4096)

    residual_list = []
    latent_list = []

    with torch.no_grad():
        for batch in eval_loader:
            x = batch[0].to(device)
            recon, latent = model(x)

            residual_list.append(torch.mean((recon - x)**2, dim=1).cpu())
            latent_list.append(latent.cpu())

    residual = torch.cat(residual_list).numpy()
    latent_np = torch.cat(latent_list).numpy()

    latent_dist = np.linalg.norm(latent_np - latent_centroid, axis=1)
    anomaly_score = 0.6 * residual + 0.4 * latent_dist

    X_eval_final = np.hstack([X_eval, anomaly_score.reshape(-1, 1)])
    rf_preds = rf.predict(X_eval_final)

    final_preds = []

    for i in range(len(X_eval)):
        if residual[i] > threshold:
            if rf_preds[i] != y_eval[i]:
                final_preds.append("ZERO_DAY")
            else:
                final_preds.append(rf_preds[i])
        else:
            final_preds.append("BENIGN")

    zero_rate = (np.array(final_preds)[y_eval == ZERO_DAY] == "ZERO_DAY").mean()
    benign_rate = (np.array(final_preds)[y_eval == "BENIGN"] == "BENIGN").mean()

    print("Zero-Day Recall:", round(zero_rate, 4))
    print("Benign Recall:", round(benign_rate, 4))