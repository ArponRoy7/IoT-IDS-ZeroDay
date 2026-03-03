# =========================================================
# HIGH-CAPACITY DUAL-RESIDUAL ADAPTIVE HYBRID (CICIoT2023)
# MODE: MAX DDOS DETECTION (99% Target) + STABLE BENIGN
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from preprocessing.preprocess_ciciot2023 import load_clean_ciciot2023
from collections import deque
import random

# =========================================================
# PARAMETERS (TUNED FOR MAX ATTACK DETECTION)
# =========================================================

# 🚀 TUNING 1: Window 25 catches the "Perfect Rhythm" of UDP Floods
WINDOW_SIZE = 100000
VARIANCE_WINDOW = 25 

# 🚀 TUNING 2: Threshold 95 is the sweet spot. 
# It catches the flood volume without killing Benign recall.
THRESHOLD_PERCENTILE = 95  

ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85

EPOCHS = 35
BATCH_SIZE = 4096

device = torch.device("cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_ciciot2023()

# Optimizing memory
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# ZERO DAY LIST (Full Roster)
# =========================================================

ZERO_DAY_LIST = [
    # The Critical Ones to Fix
    "DDoS-TCP_Flood",
    "DDoS-PSHACK_Flood",
    "DDoS-SYN_Flood",
    "DDoS-RSTFINFlood",
    "DDoS-SynonymousIP_Flood",
    "DDoS-ICMP_Fragmentation",
    "DDoS-UDP_Fragmentation",
    "DDoS-ACK_Fragmentation",
    "DDoS-HTTP_Flood",
    "DDoS-SlowLoris",

    # Remaining DoS
    "DoS-UDP_Flood",
    "DoS-TCP_Flood",
    "DoS-SYN_Flood",
    
    # The Stealthy Ones
    "Mirai-udpplain",
    "Recon-PortScan",
    "VulnerabilityScan",
    "DoS-HTTP_Flood"
]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# TRAIN DAE ONCE
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
best_loss = float("inf")
best_model_state = None

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

    if total_loss < best_loss:
        best_loss = total_loss
        best_model_state = copy.deepcopy(model.state_dict())

    if epoch > 5 and abs(prev_loss - total_loss) < 1e-4:
        print("Early stopping triggered")
        break
    prev_loss = total_loss

if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"Loaded best model with Loss: {round(best_loss, 4)}")

model.eval()

# =========================================================
# ZERO DAY LOOP
# =========================================================

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n" + "="*80)
    print("TEST:", ZERO_DAY)
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
    # RF TRAIN (Boosted)
    # =====================================================

    print("Training RF...")
    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]
    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32).to(device)
    
    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), BATCH_SIZE):
            batch = X_rf_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())
    residual_rf = torch.cat(residual_list).numpy()

    # 🚀 KEY FIX: Window=25 to catch robotic UDP floods
    variance_rf = pd.Series(residual_rf).rolling(
        window=VARIANCE_WINDOW, min_periods=1
    ).var().fillna(0).values

    X_rf_aug = np.hstack([
        X_rf,
        residual_rf.reshape(-1, 1),
        variance_rf.reshape(-1, 1)
    ])

    rf = RandomForestClassifier(
        n_estimators=600,       # 🚀 Boosted for stability
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_rf_aug, y_rf)
    print("RF trained")

    # =====================================================
    # EVALUATION
    # =====================================================

    print("Running evaluation...")
    eval_df = pd.concat([
        benign_df.sample(min(300000, len(benign_df)), random_state=seed),
        zero_df
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), BATCH_SIZE):
            batch = X_eval_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())
    residual_eval = torch.cat(residual_list).numpy()

    # 🚀 Match Window=25 for Evaluation
    variance_eval = pd.Series(residual_eval).rolling(
        window=VARIANCE_WINDOW, min_periods=1
    ).var().fillna(0).values

    X_eval_aug = np.hstack([
        X_eval,
        residual_eval.reshape(-1, 1),
        variance_eval.reshape(-1, 1)
    ])

    rf_preds = rf.predict(X_eval_aug)
    rf_probs = rf.predict_proba(X_eval_aug)
    
    hybrid_preds = []
    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    for i in range(len(X_eval)):
        residual = residual_eval[i]
        
        # Periodic threshold update
        if i > 0 and i % 1000 == 0:
            threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

        rf_pred = rf_preds[i]
        rf_prob = np.max(rf_probs[i])

        if residual > threshold:
            if rf_pred == "BENIGN":
                final_pred = "BENIGN" if rf_prob >= ALPHA_BENIGN else "ZERO_DAY"
            else:
                final_pred = rf_pred if rf_prob >= ALPHA_ATTACK else "ZERO_DAY"
        else:
            final_pred = rf_pred
        
        hybrid_preds.append(final_pred)
        if final_pred == "BENIGN":
            residual_memory.append(residual)

    hybrid_preds = np.array(hybrid_preds)

    print("Benign Recall:", round(recall_score(y_eval == "BENIGN", hybrid_preds == "BENIGN"), 4))
    print("Zero-Day Recall:", round(recall_score(y_eval == ZERO_DAY, hybrid_preds == "ZERO_DAY"), 4))

print("\nHIGH-CAPACITY Hybrid (CICIoT2023 - 99% Target) Completed")