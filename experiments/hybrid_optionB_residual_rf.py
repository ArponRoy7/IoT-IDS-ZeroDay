# OPTION B — RESIDUAL FEATURE INTO RF ONLY

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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

train_df = df[df["Label"] != ZERO_DAY]
benign_df = train_df[train_df["Label"] == "BENIGN"]

# =========================================================
# Baseline scaler (IMPORTANT — fit on train_df)
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
        return recon

model = AE(X_benign.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(
    TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
    batch_size=8192,
    shuffle=True
)

print("Training AE...")
for epoch in range(12):
    total = 0
    for batch in loader:
        x = batch[0].to(device)
        recon = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total:.4f}")

# =========================================================
# Compute reconstruction error for benign
# =========================================================
model.eval()
with torch.no_grad():
    benign_tensor = torch.tensor(X_benign, dtype=torch.float32).to(device)
    recon = model(benign_tensor)
    benign_error = torch.mean((recon - benign_tensor)**2, dim=1).cpu().numpy()

# =========================================================
# Prepare RF with residual feature
# =========================================================
print("\nTraining RF with residual feature...")

# Compute train reconstruction error
X_train_scaled = scaler.transform(train_df.drop("Label", axis=1))
train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    recon_train = model(train_tensor)
    train_error = torch.mean((recon_train - train_tensor)**2, dim=1).cpu().numpy()

# Add residual feature
X_rf_train = np.hstack([X_train_scaled, train_error.reshape(-1,1)])
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

X_eval_scaled = scaler.transform(eval_df.drop("Label", axis=1))
y_eval = eval_df["Label"].values

eval_tensor = torch.tensor(X_eval_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    recon_eval = model(eval_tensor)
    eval_error = torch.mean((recon_eval - eval_tensor)**2, dim=1).cpu().numpy()

# Add residual feature
X_rf_eval = np.hstack([X_eval_scaled, eval_error.reshape(-1,1)])
rf_preds = rf.predict(X_rf_eval)

# =========================================================
# Hybrid gating (baseline threshold)
# =========================================================
print("\nOPTION B RESULTS")

for fpr in np.linspace(0.005, 0.04, 8):

    threshold = np.percentile(benign_error, 100 - fpr*100)

    hybrid = np.where(eval_error > threshold, "ANOMALY", rf_preds)

    zero_mask = (y_eval == ZERO_DAY)
    zero_rate = (hybrid[zero_mask] == "ANOMALY").mean()

    benign_mask = (y_eval == "BENIGN")
    benign_recall = (hybrid[benign_mask] == "BENIGN").mean()

    print(f"FPR {fpr*100:.2f}% → Zero-Day {zero_rate:.4f} | Benign {benign_recall:.4f}")
