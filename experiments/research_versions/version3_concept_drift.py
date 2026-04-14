# =========================================================
# HIGH-CAPACITY DUAL-RESIDUAL ADAPTIVE HYBRID
# DAE TRAINED ONCE + RF=350 + EARLY STOPPING
# UPGRADE: LIFELONG LEARNING + FULL STATE LOCKDOWN + STATE ISOLATION
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
import copy  # 🔥 NEW: Needed for deep-copying model states

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

# Lifelong Learning Parameters
MICRO_BATCH_SIZE = 512  

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

# Tracker for the final table
lifelong_results_table = []

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

model = DAE(X_benign.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True)

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

# 🔥 THE FIX: Take a snapshot of the pure, clean DAE weights
print("Saving clean DAE baseline state...")
base_model_weights = copy.deepcopy(model.state_dict())

# =========================================================
# ================= ZERO DAY LOOP =========================
# =========================================================

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n" + "="*80)
    print("TEST:", ZERO_DAY)
    print("="*80)

    # 🔥 THE FIX: Reset the model back to the clean baseline for every single attack
    model.load_state_dict(copy.deepcopy(base_model_weights))
    model.eval()

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
    # EVALUATION WITH LIFELONG LEARNING + FULL STATE LOCKDOWN
    # =====================================================

    print("Running evaluation (with online adaptation + Full Lockdown Circuit Breaker)...")

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

    lifelong_batch = []
    adaptation_count = 0
    
    recent_alarms = deque(maxlen=1000)
    STORM_THRESHOLD = 0.05 

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
        
        recent_alarms.append(1 if final_pred == "ZERO_DAY" else 0)
        storm_active = False
        if len(recent_alarms) == 1000 and (sum(recent_alarms) / 1000) > STORM_THRESHOLD:
            storm_active = True

        # 🔥 THE FIX: Do not update the threshold memory IF a storm is active
        if final_pred == "BENIGN" and not storm_active:
            residual_memory.append(residual)
            
            if rf_pred == "BENIGN" and rf_prob > 0.99:
                lifelong_batch.append(X_eval[i])
                
                if len(lifelong_batch) >= MICRO_BATCH_SIZE:
                    model.train()
                    optimizer.zero_grad()
                    batch_tensor = torch.tensor(np.array(lifelong_batch), dtype=torch.float32).to(device)
                    noise = torch.randn_like(batch_tensor) * 0.05
                    recon, _ = model(batch_tensor + noise)
                    loss = criterion(recon, batch_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    model.eval()  
                    lifelong_batch = [] 
                    adaptation_count += 1

    hybrid_preds = np.array(hybrid_preds)

    b_rec = round(recall_score(y_eval == "BENIGN", hybrid_preds == "BENIGN"), 4)
    z_rec = round(recall_score(y_eval == ZERO_DAY, hybrid_preds == "ZERO_DAY"), 4)

    print(f"Lifelong Learning: Adapted {adaptation_count} times.")
    print("Benign Recall:", b_rec)
    print("Zero-Day Recall:", z_rec)
    
    lifelong_results_table.append({
        "Attack": ZERO_DAY,
        "Adaptations": adaptation_count,
        "Benign_Recall": b_rec,
        "Zero_Day_Recall": z_rec
    })

# =========================================================
# PRINT FINAL MARKDOWN TABLE FOR PAPER
# =========================================================
print("\n\n" + "="*80)
print("FINAL LIFELONG LEARNING TABLE (FULL STATE LOCKDOWN)")
print("="*80)

print("| Attack Scenario | Adaptations (Micro-Batches) | Benign Recall | Zero-Day Recall |")
print("| :--- | :--- | :--- | :--- |")
for res in lifelong_results_table:
    print(f"| {res['Attack']} | {res['Adaptations']} | {res['Benign_Recall']:.4f} | **{res['Zero_Day_Recall']:.4f}** |")

print("="*80 + "\n")