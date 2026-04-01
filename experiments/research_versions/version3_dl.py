# =========================================================
# HIGH-CAPACITY DUAL-RESIDUAL ADAPTIVE HYBRID vs DEEP MLP
# SOTA COMPARISON: PROVING DEEP LEARNING OVERFIT & LATENCY
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
import time

# =========================================================
# PARAMETERS
# =========================================================

WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 90
ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85

EPOCHS = 35
MLP_EPOCHS = 15 # MLP needs fewer epochs to overfit/memorize known attacks
BATCH_SIZE = 4096
EXCLUDE_ATTACKS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# --- DEEP MLP BASELINE CLASS ---
class DeepMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid() # Binary output: 0=Benign, 1=Attack
        )
    def forward(self, x):
        return self.network(x)

hybrid_dae = DAE(X_benign.shape[1]).to(device)
optimizer = torch.optim.AdamW(hybrid_dae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True)

print("\nTraining Hybrid DAE (once)...")
hybrid_dae.train()
prev_loss = float("inf")

for epoch in range(EPOCHS):
    total_loss = 0
    for (x,) in loader:
        x = x.to(device)
        noise = torch.randn_like(x) * 0.05
        optimizer.zero_grad()
        recon, _ = hybrid_dae(x + noise)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch > 5 and abs(prev_loss - total_loss) < 1e-4:
        break
    if (epoch+1) % 5 == 0:
        print("Epoch", epoch + 1, "Loss:", round(total_loss, 4))
    prev_loss = total_loss

hybrid_dae.eval()

# =========================================================
# ================= ZERO DAY LOOP =========================
# =========================================================

final_results = []

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n" + "="*80)
    print("TEST:", ZERO_DAY)
    print("="*80)

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # =====================================================
    # TRAINING HYBRID RF
    # =====================================================
    print("Training Proposed Hybrid RF...")
    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]
    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)

    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), BATCH_SIZE):
            batch = X_rf_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = hybrid_dae(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_rf = torch.cat(residual_list).numpy()
    variance_rf = pd.Series(residual_rf).rolling(window=15, min_periods=1).var().fillna(0).values
    X_rf_aug = np.hstack([X_rf, residual_rf.reshape(-1, 1), variance_rf.reshape(-1, 1)])

    rf = RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rf.fit(X_rf_aug, y_rf)

    # Init Sliding Window
    benign_residuals = residual_rf[y_rf == "BENIGN"]
    residual_memory = deque(benign_residuals[-WINDOW_SIZE:], maxlen=WINDOW_SIZE)

    # =====================================================
    # TRAINING DEEP MLP (SOTA Baseline)
    # =====================================================
    print("Training Deep MLP Baseline...")
    # Map to Binary: Benign=0, Attack=1
    y_train_mlp = np.where(train_df["Label"] == "BENIGN", 0, 1).astype(np.float32)
    y_mlp_tensor = torch.tensor(y_train_mlp).unsqueeze(1)
    
    mlp_loader = DataLoader(TensorDataset(X_rf_tensor, y_mlp_tensor), batch_size=BATCH_SIZE, shuffle=True)
    
    mlp_model = DeepMLP(X_rf.shape[1]).to(device)
    mlp_optim = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
    mlp_crit = nn.BCELoss()

    mlp_model.train()
    for epoch in range(MLP_EPOCHS):
        for bx, by in mlp_loader:
            bx, by = bx.to(device), by.to(device)
            mlp_optim.zero_grad()
            loss = mlp_crit(mlp_model(bx), by)
            loss.backward()
            mlp_optim.step()
    mlp_model.eval()

    # =====================================================
    # EVALUATION PREP
    # =====================================================
    print("Running evaluations...")
    eval_df = pd.concat([benign_df.sample(min(300000, len(benign_df)), random_state=seed), zero_df])
    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval_raw = eval_df["Label"].values
    y_eval_binary = np.where(y_eval_raw == "BENIGN", 0, 1)

    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    # =====================================================
    # EVALUATE DEEP MLP
    # =====================================================
    start_mlp = time.perf_counter()
    with torch.no_grad():
        mlp_probs = []
        for i in range(0, len(X_eval_tensor), BATCH_SIZE):
            batch = X_eval_tensor[i:i+BATCH_SIZE].to(device)
            mlp_probs.append(mlp_model(batch).cpu())
        mlp_probs = torch.cat(mlp_probs).numpy().flatten()
    
    mlp_preds = np.where(mlp_probs > 0.5, 1, 0)
    end_mlp = time.perf_counter()
    
    mlp_latency_us = ((end_mlp - start_mlp) / len(X_eval)) * 1_000_000
    mlp_z_recall = round(recall_score(y_eval_raw == ZERO_DAY, mlp_preds == 1), 4)
    mlp_b_recall = round(recall_score(y_eval_binary == 0, mlp_preds == 0), 4)

    # =====================================================
    # EVALUATE PROPOSED HYBRID
    # =====================================================
    start_hybrid = time.perf_counter()
    
    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), BATCH_SIZE):
            batch = X_eval_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = hybrid_dae(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_eval = torch.cat(residual_list).numpy()
    variance_eval = pd.Series(residual_eval).rolling(window=15, min_periods=1).var().fillna(0).values
    X_eval_aug = np.hstack([X_eval, residual_eval.reshape(-1, 1), variance_eval.reshape(-1, 1)])

    rf_preds = rf.predict(X_eval_aug)
    rf_probs = np.max(rf.predict_proba(X_eval_aug), axis=1)

    hybrid_preds = []
    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    for i in range(len(X_eval)):
        residual = residual_eval[i]
        if i > 0 and i % 1000 == 0:
            threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

        rf_pred = rf_preds[i]
        rf_prob = rf_probs[i]

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
    end_hybrid = time.perf_counter()

    hybrid_latency_us = ((end_hybrid - start_hybrid) / len(X_eval)) * 1_000_000
    hybrid_z_recall = round(recall_score(y_eval_raw == ZERO_DAY, hybrid_preds == "ZERO_DAY"), 4)
    hybrid_b_recall = round(recall_score(y_eval_raw == "BENIGN", hybrid_preds == "BENIGN"), 4)

    # =====================================================
    # RECORD AND PRINT
    # =====================================================
    print(f"-> [Deep MLP] Zero-Day: {mlp_z_recall} | Benign: {mlp_b_recall} | Latency: {round(mlp_latency_us, 2)} us")
    print(f"-> [Hybrid]   Zero-Day: {hybrid_z_recall} | Benign: {hybrid_b_recall} | Latency: {round(hybrid_latency_us, 2)} us")

    final_results.append({
        "Attack Category": ZERO_DAY,
        "Hybrid Zero-Day": hybrid_z_recall,
        "MLP Zero-Day": mlp_z_recall,
        "Hybrid Benign": hybrid_b_recall,
        "MLP Benign": mlp_b_recall,
        "Hybrid Latency (us)": round(hybrid_latency_us, 2),
        "MLP Latency (us)": round(mlp_latency_us, 2)
    })

# =========================================================
# FINAL OUTPUT MATRIX
# =========================================================
print("\n" + "!"*90)
print("FINAL SOTA COMPARISON MATRIX: PROPOSED HYBRID vs DEEP LEARNING (MLP)")
print("!"*90)
final_df = pd.DataFrame(final_results)
print(final_df.to_string(index=False))