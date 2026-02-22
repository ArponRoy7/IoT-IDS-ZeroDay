# =========================================================
# CROSS DATASET LEVEL 1 (UNIFIED FEATURE SPACE)
# TRAIN: CICIDS_UNIFIED
# TEST : BOT_UNIFIED
# ZERO-DAY: DDoS
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("\n====================================================")
print("CROSS DATASET EXPERIMENT (UNIFIED FEATURES)")
print("TRAIN DATASET  : CICIDS_UNIFIED")
print("TEST DATASET   : BOT_UNIFIED")
print("ZERO-DAY CLASS : DDoS")
print("====================================================\n")

# =========================================================
# LOAD UNIFIED DATA
# =========================================================

cicids = pd.read_csv("data/processed/cicids_unified.csv")
bot = pd.read_csv("data/processed/bot_unified.csv")

print("CICIDS shape:", cicids.shape)
print("BoT shape:", bot.shape)
print("Columns:", cicids.columns.tolist())

zero_rates = []
benign_rates = []

# =========================================================
# 3 RUNS
# =========================================================

for run in range(3):

    print(f"\n================ RUN {run+1} ================")

    seed = 42 + run
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # =====================================================
    # TRAIN ON CICIDS (REMOVE DDoS)
    # =====================================================

    train_df = cicids[cicids["Label"] != "DDoS"]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    print("Train samples:", len(train_df))
    print("Benign samples:", len(benign_df))

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))  # Fit ONLY on CICIDS benign

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))

    # =====================================================
    # AUTOENCODER
    # =====================================================

    class AE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 4)
            )
            self.decoder = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, dim)
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = AE(X_benign.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
        batch_size=8192,
        shuffle=True
    )

    model.train()
    for epoch in range(20):  # reduced epochs since features small
        for batch in loader:
            x = batch[0].to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    # =====================================================
    # THRESHOLD (99th percentile)
    # =====================================================

    errors = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            recon = model(x)
            errors.append(torch.mean((recon - x)**2, dim=1).cpu())

    errors = torch.cat(errors).numpy()
    threshold = np.percentile(errors, 99)

    print("AE Threshold:", round(threshold, 6))

    # =====================================================
    # RF TRAINING (CICIDS ONLY)
    # =====================================================

    rf_sample = train_df.sample(
        min(300000, len(train_df)),
        random_state=seed
    )

    X_rf = scaler.transform(rf_sample.drop("Label", axis=1))
    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = model(X_rf_tensor)
        residual = torch.mean((recon - X_rf_tensor)**2, dim=1).cpu().numpy()

    X_rf = np.hstack([X_rf, residual.reshape(-1,1)])
    y_rf = rf_sample["Label"]

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_rf, y_rf)

    # =====================================================
    # TEST ON BOT
    # =====================================================

    bot_benign = bot[bot["Label"] == "BENIGN"]
    bot_ddos = bot[bot["Label"] == "DDoS"]

    eval_df = pd.concat([
        bot_benign.sample(min(50000, len(bot_benign)), random_state=seed),
        bot_ddos.sample(min(50000, len(bot_ddos)), random_state=seed)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = model(X_eval_tensor)
        residual = torch.mean((recon - X_eval_tensor)**2, dim=1).cpu().numpy()

    X_eval_rf = np.hstack([X_eval, residual.reshape(-1,1)])
    rf_preds = rf.predict(X_eval_rf)

    y_eval = eval_df["Label"].values
    final_preds = []

    for i in range(len(X_eval)):
        if residual[i] > threshold:
            if rf_preds[i] != y_eval[i]:
                final_preds.append("ZERO_DAY")
            else:
                final_preds.append(rf_preds[i])
        else:
            final_preds.append("BENIGN")

    zero_rate = (np.array(final_preds)[y_eval == "DDoS"] == "ZERO_DAY").mean()
    benign_rate = (np.array(final_preds)[y_eval == "BENIGN"] == "BENIGN").mean()

    zero_rates.append(zero_rate)
    benign_rates.append(benign_rate)

    print("BoT Zero-Day Detection:", round(zero_rate,4))
    print("BoT Benign Recall     :", round(benign_rate,4))

# =====================================================
# FINAL RESULTS
# =====================================================

print("\n====================================================")
print("FINAL CROSS-DATASET RESULTS (CICIDS → BoT)")
print("====================================================")
print("BoT Zero-Day Mean :", round(np.mean(zero_rates),4))
print("BoT Zero-Day Std  :", round(np.std(zero_rates),4))
print("BoT Benign Mean   :", round(np.mean(benign_rates),4))
print("BoT Benign Std    :", round(np.std(benign_rates),4))