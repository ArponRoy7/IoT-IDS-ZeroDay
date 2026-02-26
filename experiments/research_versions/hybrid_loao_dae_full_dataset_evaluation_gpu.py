# =========================================================
# HYBRID LOAO - FULL DATASET ZERO-DAY EVALUATION
# DAE Residual-Based Detection (95th Percentile)
# Strict Leave-One-Attack-Out (LOAO)
# GPU Optimized | Thesis-Ready Version
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from preprocessing.preprocess_cicids import load_clean_cicids
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# Automatically get all attack types
ALL_LABELS = df["Label"].unique()
ATTACKS = [label for label in ALL_LABELS if label != "BENIGN"]

print("\nDetected Attack Types:")
for attack in ATTACKS:
    print("-", attack)

print("\nTotal Zero-Day Experiments:", len(ATTACKS))

# =========================================================
# LOOP OVER ALL ATTACK TYPES
# =========================================================

for ZERO_DAY in ATTACKS:

    print("\n" + "="*80)
    print("ZERO-DAY ATTACK:", ZERO_DAY)
    print("="*80)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_df = df[df["Label"] != ZERO_DAY]
    test_zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # Skip if attack has too few samples
    if len(test_zero_df) < 50:
        print("Skipping (too few samples)")
        continue

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    # =====================================================
    # TRAIN DAE ON BENIGN ONLY
    # =====================================================

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))
    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

    class DAE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 8)
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 64),
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_benign_tensor),
        batch_size=4096,
        shuffle=True
    )

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
    # CALCULATE BENIGN THRESHOLD
    # =====================================================

    recon_list = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _ = model(x)
            recon_list.append(torch.mean((recon - x)**2, dim=1))

    recon_train = torch.cat(recon_list)
    threshold = torch.quantile(recon_train, 0.95).item()

    print("Detection Threshold (95%):", round(threshold, 6))

    # =====================================================
    # EVALUATION
    # =====================================================

    eval_df = pd.concat([
        benign_df.sample(min(60000, len(benign_df)), random_state=seed),
        test_zero_df.sample(min(10000, len(test_zero_df)), random_state=seed)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    residual_list = []

    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), 4096):
            x = X_eval_tensor[i:i+4096].to(device)
            recon, _ = model(x)
            residual_list.append(torch.mean((recon - x)**2, dim=1).cpu())

    residual_eval = torch.cat(residual_list)

    final_preds = []

    for i in range(len(X_eval)):
        if residual_eval[i] > threshold:
            final_preds.append("ZERO_DAY")
        else:
            final_preds.append("BENIGN")

    final_preds = np.array(final_preds)

    zero_rate = (final_preds[y_eval == ZERO_DAY] == "ZERO_DAY").mean()
    benign_rate = (final_preds[y_eval == "BENIGN"] == "BENIGN").mean()

    print("Zero-Day Recall:", round(zero_rate, 4))
    print("Benign Recall:", round(benign_rate, 4))

print("\nALL ZERO-DAY EXPERIMENTS COMPLETED")