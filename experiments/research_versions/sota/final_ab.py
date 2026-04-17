# =========================================================
# HIGH-CAPACITY DUAL-RESIDUAL ADAPTIVE HYBRID
# DAE TRAINED ONCE + RF=350 + EARLY STOPPING
# (INCLUDES DYNAMIC ABLATION STUDY - PERFORMANCE OPTIMIZED)
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
import sys
import gc

# =========================================================
# PARAMETERS
# =========================================================
WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 90
ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85

EPOCHS = 35
BATCH_SIZE = 8192 
EXCLUDE_ATTACKS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================================================
# LOAD DATA
# =========================================================
print("\n[Data] Loading and preprocessing dataset...")
df = load_clean_cicids()

for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ZERO_DAY_LIST = [lbl for lbl in df["Label"].unique() if lbl != "BENIGN" and lbl not in EXCLUDE_ATTACKS]
print(f"[Data] Found {len(ZERO_DAY_LIST)} attacks for LOAO testing.")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# MODEL DEFINITION
# =========================================================
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

# =========================================================
# TRAIN GLOBAL DAE
# =========================================================
print("\n[Stage 1] Fitting Global StandardScaler...")
benign_full = df[df["Label"] == "BENIGN"]
scaler = StandardScaler()
scaler.fit(benign_full.drop("Label", axis=1))
X_benign = scaler.transform(benign_full.drop("Label", axis=1))
X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

model = DAE(X_benign.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True)

print("\n[Stage 2] Training Global DAE...")
model.train()
prev_loss = float("inf")
for epoch in range(EPOCHS):
    total_loss = 0
    for (x,) in loader:
        x = x.to(device)
        noise = torch.randn_like(x) * 0.05
        optimizer.zero_grad()
        recon, _ = model(x + noise)
        loss = nn.MSELoss()(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  -> Epoch {epoch + 1}/{EPOCHS} Loss: {round(total_loss, 4)}")
    if epoch > 5 and abs(prev_loss - total_loss) < 1e-4: break
    prev_loss = total_loss
model.eval()

del X_benign_tensor, loader
gc.collect()
torch.cuda.empty_cache()

# =========================================================
# ZERO DAY LOOP
# =========================================================
all_results = []

for idx, ZERO_DAY in enumerate(ZERO_DAY_LIST):
    print("\n" + "="*80)
    print(f"TEST [{idx+1}/{len(ZERO_DAY_LIST)}]: {ZERO_DAY}")
    print("="*80)

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    X_train_raw = scaler.transform(train_df.drop("Label", axis=1))
    y_train_raw = train_df["Label"]
    
    # 1. RF Only Baseline (FIXED: Set to 100 estimators to match SOTA benchmark)
    print("  -> Training RF Only Baseline...")
    rf_only = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rf_only.fit(X_train_raw, y_train_raw)
    
    # 2. Hybrid Preparation
    print("  -> Extracting residuals for Hybrid...")
    X_train_tensor = torch.tensor(X_train_raw, dtype=torch.float32)
    res_train_list = []
    with torch.no_grad():
        for i in range(0, len(X_train_tensor), BATCH_SIZE):
            batch = X_train_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            res_train_list.append(torch.mean((recon - batch)**2, dim=1).cpu())
    
    residual_train = torch.cat(res_train_list).numpy()
    variance_train = pd.Series(residual_train).rolling(window=15, min_periods=1).var().fillna(0).values
    X_train_aug = np.hstack([X_train_raw, residual_train.reshape(-1, 1), variance_train.reshape(-1, 1)])

    print("  -> Training Hybrid RF...")
    hybrid_rf = RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    hybrid_rf.fit(X_train_aug, y_train_raw)

    # 3. Evaluation
    print("  -> Running unified evaluation...")
    eval_df = pd.concat([benign_df.sample(min(300000, len(benign_df)), random_state=seed), zero_df])
    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    # RF Only Preds
    rf_only_preds = np.where(rf_only.predict(X_eval) == "BENIGN", "BENIGN", "ZERO_DAY")

    # DAE Residuals
    res_eval_list = []
    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), BATCH_SIZE):
            batch = X_eval_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            res_eval_list.append(torch.mean((recon - batch)**2, dim=1).cpu())
    residual_eval = torch.cat(res_eval_list).numpy()
    variance_eval = pd.Series(residual_eval).rolling(window=15, min_periods=1).var().fillna(0).values
    X_eval_aug = np.hstack([X_eval, residual_eval.reshape(-1, 1), variance_eval.reshape(-1, 1)])

    # Hybrid Preds
    hrf_preds = hybrid_rf.predict(X_eval_aug)
    hrf_probs = np.max(hybrid_rf.predict_proba(X_eval_aug), axis=1)

    # Isolated Memories
    base_memory = residual_train[y_train_raw == "BENIGN"][-WINDOW_SIZE:]
    hyb_mem = deque(base_memory, maxlen=WINDOW_SIZE)
    dae_mem = deque(base_memory, maxlen=WINDOW_SIZE)
    
    hybrid_final, dae_only_final = [], []
    
    for i in range(len(X_eval)):
        res = residual_eval[i]
        # Dynamic Threshold Update
        if i % 1000 == 0:
            t_hyb = np.percentile(hyb_mem, THRESHOLD_PERCENTILE)
            t_dae = np.percentile(dae_mem, THRESHOLD_PERCENTILE)

        # DAE Logic
        d_pred = "ZERO_DAY" if res > t_dae else "BENIGN"
        dae_only_final.append(d_pred)
        if d_pred == "BENIGN": dae_mem.append(res)

        # Hybrid Logic
        rf_p, rf_c = hrf_preds[i], hrf_probs[i]
        if res > t_hyb:
            h_pred = "BENIGN" if (rf_p == "BENIGN" and rf_c >= ALPHA_BENIGN) else "ZERO_DAY"
        else:
            h_pred = rf_p
        hybrid_final.append(h_pred)
        if h_pred == "BENIGN": hyb_mem.append(res)

    # Metrics
    hybrid_final = np.array(hybrid_final)
    dae_only_final = np.array(dae_only_final)

    res_atk = {
        "RF": recall_score(y_eval == ZERO_DAY, rf_only_preds == "ZERO_DAY"),
        "DAE": recall_score(y_eval == ZERO_DAY, dae_only_final == "ZERO_DAY"),
        "Hybrid": recall_score(y_eval == ZERO_DAY, hybrid_final == "ZERO_DAY")
    }
    res_ben = {
        "RF": recall_score(y_eval == "BENIGN", rf_only_preds == "BENIGN"),
        "DAE": recall_score(y_eval == "BENIGN", dae_only_final == "BENIGN"),
        "Hybrid": recall_score(y_eval == "BENIGN", hybrid_final == "BENIGN")
    }

    print(f"\n     --- ATTACK RECALL ---")
    print(f"     RF: {res_atk['RF']:.4f} | DAE: {res_atk['DAE']:.4f} | Hybrid: {res_atk['Hybrid']:.4f}")
    print(f"     --- BENIGN RECALL (Specificity) ---")
    print(f"     RF: {res_ben['RF']:.4f} | DAE: {res_ben['DAE']:.4f} | Hybrid: {res_ben['Hybrid']:.4f}")

    all_results.append({"atk": ZERO_DAY, "res_atk": res_atk, "res_ben": res_ben})

    del train_df, zero_df, benign_df, X_train_raw, X_eval, X_eval_tensor, X_train_tensor
    gc.collect()
    torch.cuda.empty_cache()

# =========================================================
# FINAL OUTPUT TABLE
# =========================================================
print("\n" + "="*100)
print(f"{'Isolated Zero-Day Attack':<30} | {'RF Attack/Ben':<18} | {'DAE Attack/Ben':<18} | {'Hybrid Attack/Ben':<18}")
print("-" * 100)

for r in all_results:
    atk_n = r['atk'].replace('\uFFFD', '-')
    rf_s = f"{r['res_atk']['RF']:.2f}/{r['res_ben']['RF']:.2f}"
    dae_s = f"{r['res_atk']['DAE']:.2f}/{r['res_ben']['DAE']:.2f}"
    hyb_s = f"{r['res_atk']['Hybrid']:.2f}/{r['res_ben']['Hybrid']:.2f}"
    print(f"{atk_n:<30} | {rf_s:<18} | {dae_s:<18} | {hyb_s:<18}")

# Averages
avg_hyb_atk = np.mean([r['res_atk']['Hybrid'] for r in all_results])
avg_hyb_ben = np.mean([r['res_ben']['Hybrid'] for r in all_results])
avg_dae_atk = np.mean([r['res_atk']['DAE'] for r in all_results])
avg_dae_ben = np.mean([r['res_ben']['DAE'] for r in all_results])

print("-" * 100)
print(f"{'OVERALL BALANCED ACCURACY':<30} | {'---':<18} | {(avg_dae_atk+avg_dae_ben)/2:>17.4f} | {(avg_hyb_atk+avg_hyb_ben)/2:>17.4f}")
print("="*100)