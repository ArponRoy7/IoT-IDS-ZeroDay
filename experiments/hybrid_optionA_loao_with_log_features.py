# =========================================================
# LOAO WITH SAFE LOG & RATIO FEATURES
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

df = load_clean_cicids()
df.columns = df.columns.str.strip()

print("Adding safe ratio & log features...")
eps = 1e-6

# -------------------------------
# Ratio Features (Safe)
# -------------------------------
if "Flow Duration" in df.columns and "Total Fwd Packets" in df.columns:
    df["Packets_Per_Second"] = (
        df["Total Fwd Packets"] /
        (df["Flow Duration"].abs() + eps)
    )

if "Flow Duration" in df.columns and "Total Length of Fwd Packets" in df.columns:
    df["Bytes_Per_Second"] = (
        df["Total Length of Fwd Packets"] /
        (df["Flow Duration"].abs() + eps)
    )

if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
    df["Fwd_Bwd_Ratio"] = (
        df["Total Fwd Packets"] /
        (df["Total Backward Packets"].abs() + eps)
    )

# -------------------------------
# Clip extreme values
# -------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].clip(-1e9, 1e9)

# -------------------------------
# Safe log transform (positive only)
# -------------------------------
log_features = {}
for col in numeric_cols:
    if col != "Label":
        col_min = df[col].min()
        if col_min >= 0:   # Only log non-negative features
            log_features[f"log_{col}"] = np.log1p(df[col])

df = pd.concat([df, pd.DataFrame(log_features)], axis=1)

# Replace any remaining NaN / Inf
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
ATTACKS = ["DDoS", "Infiltration", "PortScan"]
results = []

for ZERO_DAY in ATTACKS:

    print("\n" + "="*60)
    print("Zero-Day Attack:", ZERO_DAY)
    print("="*60)

    train_df = df[df["Label"] != ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))

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
        TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
        batch_size=8192,
        shuffle=True
    )

    print("Training AE...")
    for epoch in range(12):
        for batch in loader:
            x = batch[0].to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        benign_tensor = torch.tensor(X_benign, dtype=torch.float32).to(device)
        recon = model(benign_tensor)
        benign_error = torch.mean(
            (recon - benign_tensor)**2,
            dim=1
        ).cpu().numpy()

    # Evaluation
    eval_df = pd.concat([
        train_df.sample(80000, random_state=42),
        df[df["Label"] == ZERO_DAY].sample(
            min(5000, len(df[df["Label"] == ZERO_DAY])),
            random_state=42
        )
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    with torch.no_grad():
        eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
        recon_eval = model(eval_tensor)
        eval_error = torch.mean(
            (recon_eval - eval_tensor)**2,
            dim=1
        ).cpu().numpy()

    threshold = np.percentile(benign_error, 97)

    preds = np.where(
        eval_error > threshold,
        "ANOMALY",
        "BENIGN"
    )

    zero_rate = (preds[y_eval == ZERO_DAY] == "ANOMALY").mean()
    benign_recall = (preds[y_eval == "BENIGN"] == "BENIGN").mean()

    print(f"Zero-Day Detection @3%: {zero_rate:.4f}")
    print(f"Benign Recall @3%: {benign_recall:.4f}")

    results.append((ZERO_DAY, zero_rate, benign_recall))

print("\nFINAL LOAO RESULTS WITH SAFE LOG FEATURES")
print("Attack\t\tZero-Day\tBenign Recall")
for attack, zd, br in results:
    print(f"{attack}\t\t{zd:.4f}\t\t{br:.4f}")
