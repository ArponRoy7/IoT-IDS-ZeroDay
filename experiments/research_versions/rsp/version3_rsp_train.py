# =========================================================
# HIGH-CAPACITY DUAL-RESIDUAL ADAPTIVE HYBRID
# DAE TRAINED ONCE + RF=350 + EARLY STOPPING
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocess_cicids import load_clean_cicids
from collections import deque
import random
import os
import joblib

# =========================================================
# PARAMETERS
# =========================================================

WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 90
ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85

EPOCHS = 35
BATCH_SIZE = 4096
EXCLUDE_ATTACKS = []

device = torch.device("cuda")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()

for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")

for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ZERO_DAY_LIST = [
    label for label in df["Label"].unique()
    if label != "BENIGN" and label not in EXCLUDE_ATTACKS
]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# ===== TRAIN DAE ONCE USING ALL BENIGN DATA ============
# =========================================================

benign_full = df[df["Label"] == "BENIGN"]

scaler = StandardScaler()
scaler.fit(benign_full.drop("Label", axis=1))
X_benign = scaler.transform(benign_full.drop("Label", axis=1))

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
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Training DAE (once)...")
model.train()

prev_loss = float("inf")

for epoch in range(EPOCHS):

    total_loss = 0

    for (x,) in loader:
        x = x.to(device)
        noise = torch.randn_like(x) * 0.05

        optimizer.zero_grad()
        recon, _ = model(x + noise)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("Epoch", epoch + 1, "Loss:", round(total_loss, 4))

    if epoch > 5 and abs(prev_loss - total_loss) < 1e-4:
        print("Early stopping triggered")
        break

    prev_loss = total_loss

model.eval()

# =========================================================
# ================= ZERO DAY LOOP =========================
# =========================================================

base_dir = "scenarios"
os.makedirs(base_dir, exist_ok=True)

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n" + "="*80)
    print("SCENARIO:", ZERO_DAY)
    print("="*80)

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # =====================================================
    # SLIDING WINDOW INITIALIZATION
    # =====================================================

    residual_memory = deque(maxlen=WINDOW_SIZE)

    with torch.no_grad():
        for (x,) in loader:

            x = x.to(device)
            recon, _ = model(x)

            residual = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
            residual_memory.extend(residual)

    print("Sliding Window Initialized")

    # =====================================================
    # RF TRAIN
    # =====================================================

    print("Training RF...")

    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]

    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)

    residual_list = []

    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), BATCH_SIZE):
            batch = X_rf_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_rf = torch.cat(residual_list).numpy()

    variance_rf = pd.Series(residual_rf).rolling(
        window=15,
        min_periods=1
    ).var().fillna(0).values

    X_rf_aug = np.hstack([
        X_rf,
        residual_rf.reshape(-1, 1),
        variance_rf.reshape(-1, 1)
    ])

    rf = RandomForestClassifier(
        n_estimators=350,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_rf_aug, y_rf)

    print("RF trained")

    # =====================================================
    # SAVE SCENARIO (🔥 CORE CHANGE)
    # =====================================================

    scenario_path = os.path.join(base_dir, ZERO_DAY)
    os.makedirs(scenario_path, exist_ok=True)

    # Save models
    torch.save(model.state_dict(), f"{scenario_path}/dae.pt")
    joblib.dump(rf, f"{scenario_path}/rf.pkl")
    joblib.dump(scaler, f"{scenario_path}/scaler.pkl")

    # Save sliding window
    np.save(f"{scenario_path}/memory.npy", np.array(residual_memory))

    # Save test data
    eval_df = pd.concat([
        benign_df.sample(min(300000, len(benign_df)), random_state=seed),
        zero_df
    ])

    eval_df.to_csv(f"{scenario_path}/test.csv", index=False)

    print(f"✅ Saved scenario: {ZERO_DAY}")

print("\n✅ ALL SCENARIOS EXPORTED SUCCESSFULLY")