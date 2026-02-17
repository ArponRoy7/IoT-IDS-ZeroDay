# =========================================================
# FINAL LOAO - FAST & LIGHT VERSION (Same Accuracy)
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from preprocessing.preprocess_cicids import load_clean_cicids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# Load Dataset
# =========================================================
df = load_clean_cicids()
df.columns = df.columns.str.strip()

eps = 1e-6

# ---------------- Safe Ratio Features ----------------
if "Flow Duration" in df.columns and "Total Fwd Packets" in df.columns:
    df["Packets_Per_Second"] = df["Total Fwd Packets"] / (df["Flow Duration"].abs() + eps)

if "Flow Duration" in df.columns and "Total Length of Fwd Packets" in df.columns:
    df["Bytes_Per_Second"] = df["Total Length of Fwd Packets"] / (df["Flow Duration"].abs() + eps)

if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
    df["Fwd_Bwd_Ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"].abs() + eps)

# ---------------- Clip Extreme Values ----------------
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].clip(-1e9, 1e9)

# ---------------- Safe Log Transform ----------------
log_features = {}
for col in numeric_cols:
    if col != "Label" and df[col].min() >= 0:
        log_features[f"log_{col}"] = np.log1p(df[col])

df = pd.concat([df, pd.DataFrame(log_features)], axis=1)
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ATTACKS = ["DDoS", "Infiltration", "PortScan"]
results = []

for ZERO_DAY in ATTACKS:

    print("\n" + "="*60)
    print("Zero-Day Attack:", ZERO_DAY)
    print("="*60)

    train_df = df[df["Label"] != ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # =====================================================
    # Use only 200k benign for training (fast + stable)
    # =====================================================
    benign_df = benign_df.sample(
        min(200000, len(benign_df)),
        random_state=42
    )

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))
    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

    # =====================================================
    # AUTOENCODER
    # =====================================================
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
            return self.decoder(self.encoder(x))

    model = AE(X_benign.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_benign_tensor),
        batch_size=4096,              # smaller batch = lower VRAM
        shuffle=True,
        pin_memory=True
    )

    print("Training Autoencoder...")
    model.train()
    for epoch in range(10):           # 10 epochs enough
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    # =====================================================
    # BENIGN ERROR
    # =====================================================
    with torch.no_grad():
        benign_tensor = X_benign_tensor.to(device)
        recon = model(benign_tensor)
        benign_error = torch.mean((recon - benign_tensor) ** 2, dim=1).cpu().numpy()

    threshold = np.percentile(benign_error, 97)

    # =====================================================
    # EVALUATION
    # =====================================================
    zero_df = df[df["Label"] == ZERO_DAY]

    eval_df = pd.concat([
        train_df.sample(80000, random_state=42),
        zero_df.sample(min(5000, len(zero_df)), random_state=42)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    with torch.no_grad():
        eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
        recon_eval = model(eval_tensor)
        eval_error = torch.mean((recon_eval - eval_tensor) ** 2, dim=1).cpu().numpy()

    preds = np.where(eval_error > threshold, "ANOMALY", "BENIGN")

    zero_rate = (preds[y_eval == ZERO_DAY] == "ANOMALY").mean()
    benign_recall = (preds[y_eval == "BENIGN"] == "BENIGN").mean()

    print(f"Zero-Day Detection @3%: {zero_rate:.4f}")
    print(f"Benign Recall @3%: {benign_recall:.4f}")

    results.append((ZERO_DAY, zero_rate, benign_recall))

print("\nFINAL LOAO RESULTS (FAST VERSION @3% FPR)")
for attack, zd, br in results:
    print(f"{attack} -> Zero-Day: {zd:.4f}, Benign Recall: {br:.4f}")
