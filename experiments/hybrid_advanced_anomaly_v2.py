# =========================================================
# ADVANCED ANOMALY MODEL (Weighted Recon + Latent Distance)
# FIXED VERSION
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocess_cicids import load_clean_cicids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# Load dataset
# =========================================================
df = load_clean_cicids()
df.columns = df.columns.str.strip()

ZERO_DAY = "DDoS"
print("Zero-Day Attack:", ZERO_DAY)

train_df = df[df["Label"] != ZERO_DAY]
benign_df = train_df[train_df["Label"] == "BENIGN"]

print("Total benign samples:", len(benign_df))

# =========================================================
# Benign-only scaler
# =========================================================
scaler = StandardScaler()
scaler.fit(benign_df.drop("Label", axis=1))

X_benign = scaler.transform(benign_df.drop("Label", axis=1))

# =========================================================
# Autoencoder
# =========================================================
class AE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

model = AE(X_benign.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(
    TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
    batch_size=8192,
    shuffle=True
)

print("\nTraining Autoencoder...")
for epoch in range(15):
    total = 0
    for batch in loader:
        x = batch[0].to(device)
        recon, _ = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total:.4f}")

model.eval()

# =========================================================
# Compute benign statistics
# =========================================================
with torch.no_grad():
    benign_tensor = torch.tensor(X_benign, dtype=torch.float32).to(device)
    recon_benign, latent_benign = model(benign_tensor)

    residual_benign = (recon_benign - benign_tensor).cpu().numpy()
    latent_benign = latent_benign.cpu().numpy()

# Feature weighting
feature_var = np.var(X_benign, axis=0) + 1e-6
feature_weights = 1 / feature_var

# Latent center
latent_center = np.mean(latent_benign, axis=0)

# =========================================================
# Evaluation dataset
# =========================================================
eval_df = pd.concat([
    train_df.sample(80000, random_state=42),
    df[df["Label"] == ZERO_DAY].sample(5000, random_state=42)
])

X_eval = scaler.transform(eval_df.drop("Label", axis=1))
y_eval = eval_df["Label"].values

eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

with torch.no_grad():
    recon_eval, latent_eval = model(eval_tensor)

    residual_eval = (recon_eval - eval_tensor).cpu().numpy()
    latent_eval = latent_eval.cpu().numpy()

# =========================================================
# Weighted Reconstruction
# =========================================================
weighted_recon_eval = np.mean(
    (residual_eval ** 2) * feature_weights,
    axis=1
)

weighted_recon_benign = np.mean(
    (residual_benign ** 2) * feature_weights,
    axis=1
)

# =========================================================
# Latent Distance
# =========================================================
latent_distance_eval = np.linalg.norm(
    latent_eval - latent_center,
    axis=1
)

latent_distance_benign = np.linalg.norm(
    latent_benign - latent_center,
    axis=1
)

# =========================================================
# Normalize both using benign statistics
# =========================================================
wr_mean, wr_std = weighted_recon_benign.mean(), weighted_recon_benign.std()
ld_mean, ld_std = latent_distance_benign.mean(), latent_distance_benign.std()

wr_eval_norm = (weighted_recon_eval - wr_mean) / (wr_std + 1e-6)
ld_eval_norm = (latent_distance_eval - ld_mean) / (ld_std + 1e-6)

wr_benign_norm = (weighted_recon_benign - wr_mean) / (wr_std + 1e-6)
ld_benign_norm = (latent_distance_benign - ld_mean) / (ld_std + 1e-6)

combined_eval_score = wr_eval_norm + ld_eval_norm
combined_benign_score = wr_benign_norm + ld_benign_norm

# =========================================================
# Train RF baseline
# =========================================================
X_rf_train = scaler.transform(train_df.drop("Label", axis=1))
y_rf_train = train_df["Label"]

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_rf_train, y_rf_train)

rf_preds = rf.predict(X_eval)

# =========================================================
# Evaluation
# =========================================================
print("\n================ ADVANCED V2 FIXED RESULTS ================")

for fpr in np.linspace(0.005, 0.04, 8):

    threshold = np.percentile(
        combined_benign_score,
        100 - fpr * 100
    )

    hybrid = np.where(
        combined_eval_score > threshold,
        "ANOMALY",
        rf_preds
    )

    zero_rate = (hybrid[y_eval == ZERO_DAY] == "ANOMALY").mean()
    benign_recall = (hybrid[y_eval == "BENIGN"] == "BENIGN").mean()

    print(f"FPR {fpr*100:.2f}% → Zero-Day {zero_rate:.4f} | Benign {benign_recall:.4f}")
