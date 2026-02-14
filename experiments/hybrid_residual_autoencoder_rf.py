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

# =========================================================
# Feature Engineering
# =========================================================
eps = 1e-6

for col in ["Flow Bytes/s", "Flow Packets/s"]:
    if col in df.columns:
        df[col] = np.log1p(np.maximum(df[col], 0))

if {"Total Length of Fwd Packets","Total Length of Bwd Packets","Flow Duration"}.issubset(df.columns):
    df["Bytes_per_Duration"] = (
        df["Total Length of Fwd Packets"] +
        df["Total Length of Bwd Packets"]
    ) / (df["Flow Duration"] + eps)

if {"Total Fwd Packets","Total Backward Packets","Flow Duration"}.issubset(df.columns):
    df["Packets_per_Duration"] = (
        df["Total Fwd Packets"] +
        df["Total Backward Packets"]
    ) / (df["Flow Duration"] + eps)

# =========================================================
# Split Training (Exclude Zero-Day)
# =========================================================
train_df = df[df["Label"] != ZERO_DAY].reset_index(drop=True)
benign_df = train_df[train_df["Label"] == "BENIGN"]

print("Total benign samples:", len(benign_df))

# =========================================================
# Scaling (fit on full training)
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df.drop("Label", axis=1))

# =========================================================
# ================= AUTOENCODER ===========================
# =========================================================

# IMPORTANT: keep on CPU
X_benign_scaled = scaler.transform(
    benign_df.drop("Label", axis=1)
)

X_tensor = torch.tensor(
    X_benign_scaled,
    dtype=torch.float32
)  # DO NOT move to GPU here

train_loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=8192,
    shuffle=True,
    pin_memory=True
)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(X_tensor.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("\nTraining Autoencoder on FULL benign...")

for epoch in range(12):
    model.train()
    total_loss = 0

    for batch in train_loader:
        x = batch[0].to(device, non_blocking=True)
        recon = model(x)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("AE Training complete.")

# =========================================================
# Compute AE errors for ALL training data (batched)
# =========================================================

print("Computing AE residual feature for full training set...")

model.eval()
train_errors = []

full_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

full_loader = DataLoader(
    TensorDataset(full_train_tensor),
    batch_size=8192,
    shuffle=False
)

with torch.no_grad():
    for batch in full_loader:
        x = batch[0].to(device)
        recon = model(x)
        error = torch.mean((recon - x) ** 2, dim=1)
        train_errors.append(error.cpu())

train_errors = torch.cat(train_errors).numpy()

train_df = train_df.copy()
train_df["AE_Error"] = train_errors

# =========================================================
# ================= BALANCED RANDOM FOREST ================
# =========================================================
print("\nTraining Residual Hybrid RF (full dataset)...")

X_rf = train_df.drop("Label", axis=1)
y_rf = train_df["Label"]

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_rf, y_rf)

print("RF Training complete.")

# =========================================================
# Evaluation Dataset
# =========================================================
eval_df = pd.concat([
    train_df.sample(n=80000, random_state=42),
    df[df["Label"] == ZERO_DAY].sample(n=5000, random_state=42)
]).reset_index(drop=True)

X_eval_scaled = scaler.transform(
    eval_df.drop(["Label","AE_Error"], axis=1)
)

# Compute AE error for eval set (batched)
eval_tensor = torch.tensor(
    X_eval_scaled,
    dtype=torch.float32
)

eval_loader = DataLoader(
    TensorDataset(eval_tensor),
    batch_size=8192,
    shuffle=False
)

eval_errors = []

with torch.no_grad():
    for batch in eval_loader:
        x = batch[0].to(device)
        recon = model(x)
        error = torch.mean((recon - x) ** 2, dim=1)
        eval_errors.append(error.cpu())

eval_errors = torch.cat(eval_errors).numpy()

eval_df = eval_df.copy()
eval_df["AE_Error"] = eval_errors

# =========================================================
# Final Prediction
# =========================================================
hybrid_preds = rf.predict(eval_df.drop("Label", axis=1))

# =========================================================
# Metrics
# =========================================================
print("\n========================================")
print("FULL DATASET RESIDUAL HYBRID RESULTS")
print("========================================")

acc = accuracy_score(eval_df["Label"], hybrid_preds)

zero_mask = (eval_df["Label"] == ZERO_DAY)
zero_detected = (hybrid_preds[zero_mask] == ZERO_DAY).sum()
zero_rate = zero_detected / zero_mask.sum()

benign_mask = (eval_df["Label"] == "BENIGN")
benign_recall = (
    (hybrid_preds[benign_mask] == "BENIGN").sum()
    / benign_mask.sum()
)

print(f"Overall Accuracy: {acc:.4f}")
print(f"Zero-Day Detection: {zero_rate:.4f}")
print(f"Benign Recall: {benign_recall:.4f}")
