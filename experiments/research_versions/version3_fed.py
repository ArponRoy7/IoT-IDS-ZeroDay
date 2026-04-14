# =========================================================
# HIGH-CAPACITY DUAL-RESIDUAL ADAPTIVE HYBRID
# UPGRADE: SIMULATED FEDERATED LEARNING (DISTRIBUTED IOT)
# FED-AVG ACROSS 3 EDGE GATEWAYS
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from preprocessing.preprocess_cicids import load_clean_cicids
from collections import deque
import random

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
NUM_GATEWAYS = 3  # 🔥 NEW: Simulating 3 Distributed IoT Gateways

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

fed_results_table = []

# =========================================================
# ===== PREPARE DATA FOR FEDERATED LEARNING =============
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
        recon = self.decoder(z)
        return recon, z

criterion = nn.MSELoss()

# 🔥 NEW: FEDERATED LEARNING DATA SPLIT
chunk_size = len(X_benign_tensor) // NUM_GATEWAYS
gateway_data = [
    X_benign_tensor[i * chunk_size : (i + 1) * chunk_size]
    for i in range(NUM_GATEWAYS)
]

gateway_models = [DAE(X_benign.shape[1]).to(device) for _ in range(NUM_GATEWAYS)]
gateway_optimizers = [torch.optim.AdamW(m.parameters(), lr=1e-3) for m in gateway_models]

print(f"\n[FEDERATED LEARNING] Simulating Distributed Training across {NUM_GATEWAYS} Edge Gateways...")

# =========================================================
# ===== LOCAL TRAINING AT EACH GATEWAY ==================
# =========================================================

for gw_idx in range(NUM_GATEWAYS):
    print(f"\n--- Training Local DAE on Gateway {gw_idx + 1} ---")
    local_model = gateway_models[gw_idx]
    local_opt = gateway_optimizers[gw_idx]
    local_loader = DataLoader(TensorDataset(gateway_data[gw_idx]), batch_size=BATCH_SIZE, shuffle=True)
    
    local_model.train()
    prev_loss = float("inf")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for (x,) in local_loader:
            x = x.to(device)
            noise = torch.randn_like(x) * 0.05
            local_opt.zero_grad()
            recon, _ = local_model(x + noise)
            loss = criterion(recon, x)
            loss.backward()
            local_opt.step()
            total_loss += loss.item()

        print(f"Gateway {gw_idx + 1} - Epoch {epoch + 1} Loss: {round(total_loss, 4)}")
        if epoch > 5 and abs(prev_loss - total_loss) < 1e-4:
            print("Early stopping triggered")
            break
        prev_loss = total_loss

# =========================================================
# ===== FEDERATED AVERAGING (FedAvg) ====================
# =========================================================

print("\n[FEDERATED LEARNING] Performing Federated Averaging (FedAvg) to create Global Model...")

global_model = DAE(X_benign.shape[1]).to(device)
global_dict = global_model.state_dict()

# Average the weights across all gateways
for key in global_dict.keys():
    global_dict[key] = torch.stack([m.state_dict()[key] for m in gateway_models], 0).mean(0)

global_model.load_state_dict(global_dict)
global_model.eval()

# Set the primary model for the rest of the script to the new Global Model
model = global_model

# We use the full benign loader just to initialize the sliding window baseline
loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True)

# =========================================================
# ================= ZERO DAY LOOP =========================
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
    variance_rf = pd.Series(residual_rf).rolling(window=15, min_periods=1).var().fillna(0).values

    X_rf_aug = np.hstack([X_rf, residual_rf.reshape(-1, 1), variance_rf.reshape(-1, 1)])

    rf = RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rf.fit(X_rf_aug, y_rf)
    print("RF trained")

    # =====================================================
    # EVALUATION
    # =====================================================

    print("Running evaluation using Federated Global Model...")

    eval_df = pd.concat([
        benign_df.sample(min(300000, len(benign_df)), random_state=seed),
        zero_df
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), BATCH_SIZE):
            batch = X_eval_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_eval = torch.cat(residual_list).numpy()
    variance_eval = pd.Series(residual_eval).rolling(window=15, min_periods=1).var().fillna(0).values

    X_eval_aug = np.hstack([X_eval, residual_eval.reshape(-1, 1), variance_eval.reshape(-1, 1)])

    rf_preds = rf.predict(X_eval_aug)
    rf_probs = rf.predict_proba(X_eval_aug)

    hybrid_preds = []
    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    for i in range(len(X_eval)):
        residual = residual_eval[i]
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
    
    b_rec = round(recall_score(y_eval == "BENIGN", hybrid_preds == "BENIGN"), 4)
    z_rec = round(recall_score(y_eval == ZERO_DAY, hybrid_preds == "ZERO_DAY"), 4)

    print("Federated Benign Recall:", b_rec)
    print("Federated Zero-Day Recall:", z_rec)
    
    fed_results_table.append({
        "Attack": ZERO_DAY,
        "Fed_Benign": b_rec,
        "Fed_Zero": z_rec
    })

# =========================================================
# PRINT FINAL MARKDOWN TABLE FOR PAPER
# =========================================================
print("\n\n" + "="*80)
print("FINAL FEDERATED LEARNING TABLE (COPY AND PASTE INTO THESIS)")
print("="*80)

print("| Attack Scenario | Federated Benign Recall | Federated Zero-Day Recall |")
print("| :--- | :--- | :--- |")
for res in fed_results_table:
    print(f"| {res['Attack']} | {res['Fed_Benign']:.4f} | **{res['Fed_Zero']:.4f}** |")

print("="*80 + "\n")