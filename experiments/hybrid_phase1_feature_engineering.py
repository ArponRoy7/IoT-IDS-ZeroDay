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
ZERO_DAY = "DDoS"
print("Zero-Day Attack:", ZERO_DAY)

# =========================================================
# ================= FEATURE ENGINEERING ===================
# =========================================================

print("Applying log transform...")

log_features = [
    " Flow Bytes/s",
    " Flow Packets/s",
    " Packet Length Variance",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets"
]

for col in log_features:
    if col in df.columns:
        df[col] = np.log1p(np.maximum(df[col], 0))

print("Adding interaction features...")

# Avoid division by zero
eps = 1e-6

df["Bytes_per_Duration"] = (
    df["Total Length of Fwd Packets"] +
    df[" Total Length of Bwd Packets"]
) / (df[" Flow Duration"] + eps)

df["Packets_per_Duration"] = (
    df[" Total Fwd Packets"] +
    df[" Total Backward Packets"]
) / (df[" Flow Duration"] + eps)

df["Fwd_Bwd_Ratio"] = (
    df[" Total Fwd Packets"] /
    (df[" Total Backward Packets"] + eps)
)

df["Packet_Size_Ratio"] = (
    df[" Avg Fwd Segment Size"] /
    (df[" Avg Bwd Segment Size"] + eps)
)

df["Activity_Ratio"] = (
    df["Active Mean"] /
    (df["Idle Mean"] + eps)
)

# =========================================================
# TRAIN DATA (Exclude Zero-Day)
# =========================================================
train_df = df[df["Label"] != ZERO_DAY]

benign_train = train_df[train_df["Label"] == "BENIGN"]

rf_train_df = train_df

# =========================================================
# Scaling
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(
    rf_train_df.drop("Label", axis=1)
)

# =========================================================
# ================= AUTOENCODER ===========================
# =========================================================
X_benign_scaled = scaler.transform(
    benign_train.drop("Label", axis=1)
)

feature_var = np.var(X_benign_scaled, axis=0)
feature_var = np.where(feature_var < 1e-6, 1e-6, feature_var)

var_tensor = torch.tensor(feature_var, dtype=torch.float32).to(device)

X_tensor = torch.tensor(
    X_benign_scaled,
    dtype=torch.float32
).to(device)

train_loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=4096,
    shuffle=True
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
            nn.Dropout(0.2),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(X_tensor.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nTraining Autoencoder...")

for epoch in range(15):
    total_loss = 0
    for batch in train_loader:
        x = batch[0]
        recon = model(x)
        diff = (recon - x) ** 2
        loss = (diff / var_tensor).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("AE Training complete.")

# Compute benign errors
model.eval()
with torch.no_grad():
    recon = model(X_tensor)
    benign_errors = torch.mean(
        ((recon - X_tensor) ** 2) / var_tensor,
        dim=1
    ).cpu().numpy()

# =========================================================
# ================= RANDOM FOREST =========================
# =========================================================
print("\nTraining Balanced Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(
    X_train_scaled,
    rf_train_df["Label"]
)

print("RF Training complete.")

# =========================================================
# Evaluation Dataset
# =========================================================
eval_df = pd.concat([
    train_df.sample(n=80000, random_state=42),
    df[df["Label"] == ZERO_DAY].sample(n=5000, random_state=42)
])

X_eval = scaler.transform(eval_df.drop("Label", axis=1))
y_eval = eval_df["Label"].values

with torch.no_grad():
    X_eval_tensor = torch.tensor(
        X_eval,
        dtype=torch.float32
    ).to(device)

    recon = model(X_eval_tensor)
    eval_errors = torch.mean(
        ((recon - X_eval_tensor) ** 2) / var_tensor,
        dim=1
    ).cpu().numpy()

rf_preds = rf.predict(X_eval)

print("\n========================================")
print("PHASE 1 HYBRID EVALUATION")
print("========================================")

for fpr_target in [0.005, 0.01, 0.02]:

    threshold = np.percentile(
        benign_errors,
        100 - (fpr_target * 100)
    )

    anomaly_mask = eval_errors > threshold

    hybrid_preds = np.where(
        anomaly_mask,
        "ANOMALY",
        rf_preds
    )

    acc = accuracy_score(y_eval, hybrid_preds)

    zero_mask = (y_eval == ZERO_DAY)
    zero_rate = (
        (hybrid_preds[zero_mask] == "ANOMALY").sum()
        / zero_mask.sum()
    )

    benign_mask = (y_eval == "BENIGN")
    benign_recall = (
        (hybrid_preds[benign_mask] == "BENIGN").sum()
        / benign_mask.sum()
    )

    print(f"\nFPR Target: {fpr_target*100:.1f}%")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Zero-Day Detection: {zero_rate:.4f}")
    print(f"Benign Recall: {benign_recall:.4f}")
