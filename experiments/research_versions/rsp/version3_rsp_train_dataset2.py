# =========================================================
# PI-OPTIMIZED EXPORT: SERVER-PARITY VERSION (FINAL)
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
import os
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocess_ciciot2023 import load_clean_ciciot2023
from collections import deque
import random
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PARAMETERS
# =========================================================
WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 99.5
ALPHA_BENIGN = 0.9999
ALPHA_ATTACK = 0.85
EPOCHS = 35
BATCH_SIZE = 8192 
seed = 42

# =========================================================
# ✅ ALL ATTACKS (FROM SCREENSHOT)
# =========================================================
TARGET_ATTACKS = [
    # Volumetric Floods
    "DDoS-RSTFINFlood",
    "DDoS-SynonymousIP_Flood",
    "DDoS-PSHACK_Flood",
    "DoS-TCP_Flood",

    # IoT Botnet
    "Mirai-greeth_flood",
    "Mirai-udpplain",
    "Mirai-greip_flood",

    # Recon & App Layer
    "DoS-HTTP_Flood",
    "Recon-HostDiscovery",
    "Recon-PortScan",
    "MITM-ArpSpoofing",   # ✅ FIXED

    # Payload Attacks
    "SqlInjection",       # ✅ FIXED
    "Backdoor_Malware",   # ✅ FIXED
    "CommandInjection"
]

# =========================================================
# SETUP
# =========================================================
device = torch.device("cpu")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

BASE_DIR = "scenarios_ciciot_pi"
os.makedirs(BASE_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
df = load_clean_ciciot2023()

# dtype optimization
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# ✅ FILTER AVAILABLE ATTACKS (IMPORTANT)
# =========================================================
available_labels = set(df["Label"].unique())
ZERO_DAY_LIST = [a for a in TARGET_ATTACKS if a in available_labels]

missing = [a for a in TARGET_ATTACKS if a not in available_labels]

print(f"✅ Available attacks: {len(ZERO_DAY_LIST)}")
print(f"⚠️ Missing attacks: {missing}")

# =========================================================
# TRAIN GLOBAL DAE
# =========================================================
benign_full = df[df["Label"] == "BENIGN"]
scaler = StandardScaler()
scaler.fit(benign_full.drop("Label", axis=1))

X_benign = scaler.transform(benign_full.drop("Label", axis=1))
X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32).to(device)

class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


model = DAE(X_benign.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()

loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=4096, shuffle=True)

print("🚀 Training Global DAE...")

best_loss = float("inf")
best_model_state = None

for epoch in range(EPOCHS):
    total_loss = 0
    model.train()

    for (x,) in loader:
        noise = torch.randn_like(x) * 0.05
        optimizer.zero_grad()
        recon, _ = model(x + noise)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if total_loss < best_loss:
        best_loss = total_loss
        best_model_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_model_state)
model.eval()

# =========================================================
# EXPORT SCENARIOS (FLAT STRUCTURE)
# =========================================================
for ZERO_DAY in ZERO_DAY_LIST:

    print(f"\n🚀 EXPORTING: {ZERO_DAY}")

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]

    with torch.no_grad():
        X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)
        recon_rf, _ = model(X_rf_tensor)
        res_rf = torch.mean((recon_rf - X_rf_tensor)**2, dim=1).numpy()

    var_rf = pd.Series(res_rf).rolling(window=25, min_periods=1).var().fillna(0).values
    X_rf_aug = np.hstack([X_rf, res_rf.reshape(-1, 1), var_rf.reshape(-1, 1)])

    rf = RandomForestClassifier(
        n_estimators=125,
        max_depth=40,
        min_samples_leaf=3,
        max_features='sqrt',
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )
    rf.fit(X_rf_aug, y_rf)

    residual_memory = deque(res_rf[y_rf == "BENIGN"][-WINDOW_SIZE:], maxlen=WINDOW_SIZE)

    # ✅ FLAT STRUCTURE (NO CATEGORY)
    scenario_path = os.path.join(BASE_DIR, ZERO_DAY)
    os.makedirs(scenario_path, exist_ok=True)

    torch.save(model.state_dict(), f"{scenario_path}/dae.pt")
    joblib.dump(rf, f"{scenario_path}/rf.pkl", compress=3)
    joblib.dump(scaler, f"{scenario_path}/scaler.pkl")
    np.save(f"{scenario_path}/memory.npy", np.array(residual_memory, dtype=np.float32))

    eval_df = pd.concat([
        benign_df.sample(min(100000, len(benign_df)), random_state=seed),
        zero_df
    ])
    eval_df.to_csv(f"{scenario_path}/test.csv", index=False)

    print(f"✅ Saved: {scenario_path}")

print("\n✅ ALL SCENARIOS EXPORTED (FLAT STRUCTURE)")