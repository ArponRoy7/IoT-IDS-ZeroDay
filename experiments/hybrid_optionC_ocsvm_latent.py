# =========================================================
# OPTION C — OCSVM IN LATENT SPACE (OPTIMIZED)
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from preprocessing.preprocess_cicids import load_clean_cicids

# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# Load dataset
# =========================================================
df = load_clean_cicids()
df.columns = df.columns.str.strip()

ZERO_DAY = "DDoS"
print("Zero-Day Attack:", ZERO_DAY)

# =========================================================
# Train split (exclude zero-day)
# =========================================================
train_df = df[df["Label"] != ZERO_DAY]
benign_df = train_df[train_df["Label"] == "BENIGN"]

print("Total benign samples:", len(benign_df))

# =========================================================
# Baseline scaler (fit on full train_df)
# =========================================================
scaler = StandardScaler()
scaler.fit(train_df.drop("Label", axis=1))

X_benign = scaler.transform(benign_df.drop("Label", axis=1))

# =========================================================
# Autoencoder (baseline architecture)
# =========================================================
class AE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
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
for epoch in range(12):
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
# Extract latent vectors for benign
# =========================================================
with torch.no_grad():
    benign_tensor = torch.tensor(X_benign, dtype=torch.float32).to(device)
    _, latent_benign = model(benign_tensor)
    latent_benign = latent_benign.cpu().numpy()

# =========================================================
# Optimized OCSVM (subsample benign latent)
# =========================================================
print("\nTraining Optimized One-Class SVM...")

max_samples = 40000  # Adjust 30k–50k if needed

if len(latent_benign) > max_samples:
    idx = np.random.choice(len(latent_benign), max_samples, replace=False)
    latent_subset = latent_benign[idx]
else:
    latent_subset = latent_benign

ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02)
ocsvm.fit(latent_subset)

print("OCSVM Training complete.")

# =========================================================
# Train baseline RF (no residual feature)
# =========================================================
print("\nTraining baseline Random Forest...")

X_rf_train = scaler.transform(train_df.drop("Label", axis=1))
y_rf_train = train_df["Label"]

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_rf_train, y_rf_train)

print("RF Training complete.")

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
    _, latent_eval = model(eval_tensor)
    latent_eval = latent_eval.cpu().numpy()

# OCSVM prediction (-1 anomaly, 1 normal)
ocsvm_preds = ocsvm.predict(latent_eval)

# Hybrid decision: OCSVM gate → RF
rf_preds = rf.predict(X_eval)

hybrid = np.where(
    ocsvm_preds == -1,
    "ANOMALY",
    rf_preds
)

# =========================================================
# Final Evaluation
# =========================================================
print("\n================ OPTION C RESULTS ================")

zero_mask = (y_eval == ZERO_DAY)
zero_rate = (hybrid[zero_mask] == "ANOMALY").mean()

benign_mask = (y_eval == "BENIGN")
benign_recall = (hybrid[benign_mask] == "BENIGN").mean()

print(f"Zero-Day Detection: {zero_rate:.4f}")
print(f"Benign Recall: {benign_recall:.4f}")
