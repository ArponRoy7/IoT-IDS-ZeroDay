# =========================================================
# HYBRID LOAO - STABLE DDoS VERSION
# Residual-Based Detection (95 Percentile)
# GPU Safe | Stable Calibration
# best-till now 
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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = load_clean_cicids()
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ATTACKS = ["DDoS", "Infiltration", "PortScan", "Bot"]

for ZERO_DAY in ATTACKS:

    print("\n" + "="*70)
    print("ZERO-DAY:", ZERO_DAY)
    print("="*70)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_df = df[df["Label"] != ZERO_DAY]
    test_zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))
    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

    # =====================================================
    # DENOISING AUTOENCODER (Simple + Stable)
    # =====================================================

    class DAE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
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
            z = self.encoder(x)
            recon = self.decoder(z)
            return recon, z

    model = DAE(X_benign.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_benign_tensor),
        batch_size=4096,
        shuffle=True
    )

    model.train()
    for epoch in range(20):
        for (x,) in loader:
            x = x.to(device)
            noise = torch.randn_like(x) * 0.05
            recon, _ = model(x + noise)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    # =====================================================
    # BENIGN RECONSTRUCTION DISTRIBUTION
    # =====================================================

    recon_list = []
    latent_list = []

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, z = model(x)
            recon_list.append(torch.mean((recon - x)**2, dim=1))
            latent_list.append(z)

    recon_train = torch.cat(recon_list)
    latent_train = torch.cat(latent_list)

    latent_center = latent_train.mean(dim=0)

    # 🔥 Slightly relaxed threshold for higher recall
    threshold = torch.quantile(recon_train, 0.95).item()
    print("Detection Threshold (95%):", round(threshold,6))

    # =====================================================
    # RF TRAINING (Residual as Extra Feature)
    # =====================================================

    rf_sample = train_df.sample(250000, random_state=seed)
    X_rf = scaler.transform(rf_sample.drop("Label", axis=1))
    y_rf = rf_sample["Label"]

    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)

    residual_list = []
    latent_list = []

    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), 4096):
            x = X_rf_tensor[i:i+4096].to(device)
            recon, z = model(x)
            residual_list.append(torch.mean((recon - x)**2, dim=1).cpu())
            latent_list.append(z.cpu())

    residual = torch.cat(residual_list)
    latent_rf = torch.cat(latent_list)

    latent_dist_rf = torch.norm(latent_rf - latent_center.cpu(), dim=1)

    anomaly_feature = residual  # Keep simple

    X_rf_final = np.hstack([X_rf, anomaly_feature.numpy().reshape(-1,1)])

    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )
    rf.fit(X_rf_final, y_rf)

    # =====================================================
    # EVALUATION
    # =====================================================

    eval_df = pd.concat([
        train_df.sample(60000, random_state=seed),
        test_zero_df.sample(min(10000, len(test_zero_df)), random_state=seed)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    residual_list = []

    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), 4096):
            x = X_eval_tensor[i:i+4096].to(device)
            recon, _ = model(x)
            residual_list.append(torch.mean((recon - x)**2, dim=1).cpu())

    residual_eval = torch.cat(residual_list)

    final_preds = []

    for i in range(len(X_eval)):
        if residual_eval[i] > threshold:
            final_preds.append("ZERO_DAY")
        else:
            final_preds.append("BENIGN")

    final_preds = np.array(final_preds)

    zero_rate = (final_preds[y_eval == ZERO_DAY] == "ZERO_DAY").mean()
    benign_rate = (final_preds[y_eval == "BENIGN"] == "BENIGN").mean()

    print("Zero-Day Recall:", round(zero_rate,4))
    print("Benign Recall:", round(benign_rate,4))