# =========================================================
# HYBRID LOAO - FULL GPU ADAPTIVE VERSION
# DAE + GPU KMeans + Learned Fusion
# Strict LOAO | Multi-Attack
# =========================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocess_cicids import load_clean_cicids
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ATTACKS = ["DDoS", "Infiltration", "PortScan"]
N_CLUSTERS = 8
PERCENTILE = 97


# =========================================================
# GPU KMEANS (NO SKLEARN)
# =========================================================

def gpu_kmeans(X, k=8, iterations=15):
    X = X.to(device)
    indices = torch.randperm(X.size(0))[:k]
    centroids = X[indices]

    for _ in range(iterations):
        distances = torch.cdist(X, centroids)
        labels = torch.argmin(distances, dim=1)

        new_centroids = []
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(dim=0))
            else:
                new_centroids.append(centroids[i])
        centroids = torch.stack(new_centroids)

    return centroids, labels


# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# LOOP
# =========================================================

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
    # DAE
    # =====================================================

    class DAE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 16)
            )
            self.decoder = nn.Sequential(
                nn.Linear(16, 64),
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(X_benign_tensor),
                        batch_size=4096, shuffle=True)

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
    # BENIGN LATENT & RECON
    # =====================================================

    latent_list = []
    recon_list = []

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, z = model(x)
            latent_list.append(z)
            recon_list.append(torch.mean((recon - x)**2, dim=1))

    latent_train = torch.cat(latent_list)
    recon_train = torch.cat(recon_list)

    # =====================================================
    # GPU KMEANS
    # =====================================================

    print("Running GPU KMeans...")
    centroids, cluster_labels = gpu_kmeans(latent_train, k=N_CLUSTERS)

    cluster_thresholds = {}

    for cid in range(N_CLUSTERS):
        mask = (cluster_labels == cid)
        if mask.sum() > 0:
            cluster_scores = recon_train[mask]
            cluster_thresholds[cid] = torch.quantile(
                cluster_scores, PERCENTILE/100
            ).item()
        else:
            cluster_thresholds[cid] = torch.quantile(
                recon_train, PERCENTILE/100
            ).item()

    print("Cluster Thresholds:",
          {k: round(v,6) for k,v in cluster_thresholds.items()})

    latent_center = latent_train.mean(dim=0)

    # =====================================================
    # RF TRAINING
    # =====================================================

    rf_sample = train_df.sample(200000, random_state=seed)
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
    anomaly_score_rf = 0.6 * residual + 0.4 * latent_dist_rf

    X_rf_final = np.hstack([X_rf, anomaly_score_rf.numpy().reshape(-1,1)])

    rf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        random_state=seed
    )
    rf.fit(X_rf_final, y_rf)

    # =====================================================
    # EVALUATION
    # =====================================================

    eval_df = pd.concat([
        train_df.sample(50000, random_state=seed),
        test_zero_df.sample(min(10000, len(test_zero_df)), random_state=seed)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    residual_list = []
    latent_list = []

    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), 4096):
            x = X_eval_tensor[i:i+4096].to(device)
            recon, z = model(x)
            residual_list.append(torch.mean((recon - x)**2, dim=1).cpu())
            latent_list.append(z.cpu())

    residual = torch.cat(residual_list)
    latent_eval = torch.cat(latent_list)

    anomaly_score_eval = 0.6 * residual + 0.4 * torch.norm(
        latent_eval - latent_center.cpu(), dim=1
    )

    final_preds = []

    for i in range(len(X_eval)):
        distances = torch.norm(
            latent_eval[i] - centroids.cpu(), dim=1
        )
        cluster_id = torch.argmin(distances).item()

        if residual[i].item() > cluster_thresholds[cluster_id]:
            final_preds.append("ZERO_DAY")
        else:
            final_preds.append("BENIGN")

    final_preds = np.array(final_preds)

    zero_rate = (final_preds[y_eval == ZERO_DAY] == "ZERO_DAY").mean()
    benign_rate = (final_preds[y_eval == "BENIGN"] == "BENIGN").mean()

    print("Zero-Day Recall:", round(zero_rate,4))
    print("Benign Recall:", round(benign_rate,4))