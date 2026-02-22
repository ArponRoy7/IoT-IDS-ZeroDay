# =========================================================
# OPTION A — FULL LOAO WITH PORTSCAN FEATURE ENGINEERING
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

# =========================================================
# 🔥 ADD PORTSCAN-SPECIFIC INTERACTION FEATURES
# =========================================================
print("Adding interaction features...")

eps = 1e-6

if "Total Length of Fwd Packets" in df.columns and "Total Fwd Packets" in df.columns:
    df["Flow_Bytes_Per_Packet"] = (
        df["Total Length of Fwd Packets"] /
        (df["Total Fwd Packets"] + eps)
    )

if "Flow Duration" in df.columns and "Total Fwd Packets" in df.columns:
    df["Packet_Rate"] = (
        df["Total Fwd Packets"] /
        (df["Flow Duration"] + eps)
    )

if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
    df["Fwd_Backward_Ratio"] = (
        df["Total Fwd Packets"] /
        (df["Total Backward Packets"] + eps)
    )

if "Active Mean" in df.columns and "Idle Mean" in df.columns:
    df["Flow_Activity_Ratio"] = (
        df["Active Mean"] /
        (df["Idle Mean"] + eps)
    )

# =========================================================
ATTACKS = ["DDoS", "Infiltration", "PortScan"]

results = []

for ZERO_DAY in ATTACKS:

    print("\n" + "="*60)
    print("Zero-Day Attack:", ZERO_DAY)
    print("="*60)

    train_df = df[df["Label"] != ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # -----------------------------------------------------
    # Benign-only scaler
    # -----------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))

    # -----------------------------------------------------
    # Autoencoder
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # Train RF
    # -----------------------------------------------------
    X_rf_train = scaler.transform(train_df.drop("Label", axis=1))
    y_rf_train = train_df["Label"]

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_rf_train, y_rf_train)

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
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

    rf_preds = rf.predict(X_eval)

    # -----------------------------------------------------
    # Evaluate @3% FPR
    # -----------------------------------------------------
    fpr = 0.03
    threshold = np.percentile(
        benign_error,
        100 - fpr * 100
    )

    hybrid = np.where(
        eval_error > threshold,
        "ANOMALY",
        rf_preds
    )

    zero_rate = (hybrid[y_eval == ZERO_DAY] == "ANOMALY").mean()
    benign_recall = (hybrid[y_eval == "BENIGN"] == "BENIGN").mean()

    print(f"Zero-Day Detection @3%: {zero_rate:.4f}")
    print(f"Benign Recall @3%: {benign_recall:.4f}")

    results.append((ZERO_DAY, zero_rate, benign_recall))

# =========================================================
# Final Table
# =========================================================
print("\n" + "="*60)
print("FINAL LOAO RESULTS WITH SCAN FEATURES (@3%)")
print("="*60)

print("Attack\t\tZero-Day\tBenign Recall")

for attack, zd, br in results:
    print(f"{attack}\t\t{zd:.4f}\t\t{br:.4f}")
