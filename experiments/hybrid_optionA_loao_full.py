# =========================================================
# FINAL STABLE LOAO MODEL (BEST DDoS & Infiltration)
# AE + RF | 700k Benign | 12 Epochs | 200 Trees | 3% FPR
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocess_cicids import load_clean_cicids

# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
df = load_clean_cicids()
df.columns = df.columns.str.strip()
print("Dataset shape:", df.shape)

ATTACKS = ["DDoS", "Infiltration"]
results = []

# =========================================================
# LOAO LOOP
# =========================================================
for ZERO_DAY in ATTACKS:

    print("\n" + "="*60)
    print("Zero-Day Attack:", ZERO_DAY)
    print("="*60)

    # -----------------------------------------------------
    # Split (LOAO)
    # -----------------------------------------------------
    train_df = df[df["Label"] != ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"].sample(
        700000, random_state=42
    )

    # -----------------------------------------------------
    # Benign-only Scaling
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

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
        batch_size=4096,
        shuffle=True
    )

    print("Training AE...")
    model.train()
    for epoch in range(12):   # RESTORED TO 12
        for batch in train_loader:
            x = batch[0].to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    # -----------------------------------------------------
    # Compute Benign Reconstruction Error
    # -----------------------------------------------------
    benign_errors = []

    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            recon = model(x)
            err = torch.mean((recon - x) ** 2, dim=1)
            benign_errors.append(err.cpu().numpy())

    benign_error = np.concatenate(benign_errors)

    torch.cuda.empty_cache()

    # -----------------------------------------------------
    # Train RF (200 Trees RESTORED)
    # -----------------------------------------------------
    X_rf_train = scaler.transform(train_df.drop("Label", axis=1))
    y_rf_train = train_df["Label"]

    rf = RandomForestClassifier(
        n_estimators=200,   # RESTORED
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_rf_train, y_rf_train)

    # -----------------------------------------------------
    # Evaluation Set
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

    eval_loader = DataLoader(
        TensorDataset(torch.tensor(X_eval, dtype=torch.float32)),
        batch_size=4096,
        shuffle=False
    )

    eval_errors = []

    with torch.no_grad():
        for batch in eval_loader:
            x = batch[0].to(device)
            recon = model(x)
            err = torch.mean((recon - x) ** 2, dim=1)
            eval_errors.append(err.cpu().numpy())

    eval_error = np.concatenate(eval_errors)

    rf_preds = rf.predict(X_eval)

    # -----------------------------------------------------
    # Single-Sided 3% FPR Threshold (BEST CONFIG)
    # -----------------------------------------------------
    threshold = np.percentile(benign_error, 97)

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
# FINAL RESULTS
# =========================================================
print("\n" + "="*60)
print("FINAL BEST RESULTS (LOCKED CONFIG)")
print("="*60)

for attack, zd, br in results:
    print(f"{attack}\t\t{zd:.4f}\t\t{br:.4f}")
