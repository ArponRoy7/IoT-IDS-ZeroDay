# =========================================================
# UNIFIED SOTA COMPARISON: PROPOSED HYBRID vs TRADITIONAL ML
# APPLES-TO-APPLES LOAO EVALUATION (with Latency Tracking)
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
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
BATCH_SIZE = 4096
EXCLUDE_ATTACKS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# LOAD AND PREPARE DATA
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

# Standardize based on Benign
benign_full = df[df["Label"] == "BENIGN"]
scaler = StandardScaler()
scaler.fit(benign_full.drop("Label", axis=1))
X_benign = scaler.transform(benign_full.drop("Label", axis=1))

X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

# =========================================================
# TRAIN MODEL 1: PROPOSED HYBRID DAE
# =========================================================
class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, dim))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

hybrid_model = DAE(X_benign.shape[1]).to(device)
optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True)

print("\n--- Training Hybrid DAE (once) ---")
hybrid_model.train()
prev_loss = float("inf")

for epoch in range(EPOCHS):
    total_loss = 0
    for (x,) in loader:
        x = x.to(device)
        noise = torch.randn_like(x) * 0.05
        optimizer.zero_grad()
        recon, _ = hybrid_model(x + noise)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch > 5 and abs(prev_loss - total_loss) < 1e-4:
        print(f"Epoch {epoch + 1} Loss: {round(total_loss, 4)} (Early stopping triggered)")
        break
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1} Loss: {round(total_loss, 4)}")
    prev_loss = total_loss

hybrid_model.eval()

# =========================================================
# ZERO DAY EVALUATION LOOP
# =========================================================
final_results = []

for ZERO_DAY in ZERO_DAY_LIST:
    print("\n" + "="*80)
    print("TEST:", ZERO_DAY)
    print("="*80)

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # -----------------------------------------------------
    # 1. TRADITIONAL ML SOTA PREP (Isolation Forest & Std RF)
    # -----------------------------------------------------
    print("Training SOTA Baselines...")
    X_train_sota = scaler.transform(train_df.drop("Label", axis=1))
    y_train_sota = train_df["Label"]
    
    # Isolation Forest (Unsupervised Baseline) - Train only on a subset of Benign to save time
    X_train_iforest = scaler.transform(benign_df.sample(min(200000, len(benign_df)), random_state=seed).drop("Label", axis=1))
    iforest = IsolationForest(n_estimators=100, contamination=0.05, n_jobs=-1, random_state=seed)
    iforest.fit(X_train_iforest)

    # Standard Random Forest (Supervised Baseline)
    std_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    std_rf.fit(X_train_sota, y_train_sota)

    # -----------------------------------------------------
    # 2. HYBRID PREP (DAE Augmented RF)
    # -----------------------------------------------------
    print("Preparing Hybrid RF...")
    X_rf_tensor = torch.tensor(X_train_sota, dtype=torch.float32)
    residual_list = []

    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), BATCH_SIZE):
            batch = X_rf_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = hybrid_model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_rf = torch.cat(residual_list).numpy()
    variance_rf = pd.Series(residual_rf).rolling(window=15, min_periods=1).var().fillna(0).values
    X_rf_aug = np.hstack([X_train_sota, residual_rf.reshape(-1, 1), variance_rf.reshape(-1, 1)])

    hybrid_rf = RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    hybrid_rf.fit(X_rf_aug, y_train_sota)

    # Initialize sliding memory
    benign_residuals = residual_rf[y_train_sota == "BENIGN"]
    residual_memory = deque(benign_residuals[-WINDOW_SIZE:], maxlen=WINDOW_SIZE)

    # -----------------------------------------------------
    # 3. EVALUATION SET GENERATION
    # -----------------------------------------------------
    print("Running Evaluation...")
    eval_df = pd.concat([benign_df.sample(min(300000, len(benign_df)), random_state=seed), zero_df])
    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval_raw = eval_df["Label"].values
    y_eval_binary = np.where(y_eval_raw == "BENIGN", 0, 1)
    
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    # -----------------------------------------------------
    # 4. SOTA INFERENCE (Isolation Forest)
    # -----------------------------------------------------
    start_time_if = time.perf_counter()
    if_preds_raw = iforest.predict(X_eval) # 1 = Benign, -1 = Anomaly
    end_time_if = time.perf_counter()
    
    if_latency_us = ((end_time_if - start_time_if) / len(X_eval)) * 1_000_000
    if_preds = np.where(if_preds_raw == 1, 0, 1)

    if_b_recall = round(recall_score(y_eval_binary == 0, if_preds == 0), 4)
    if_z_recall = round(recall_score(y_eval_raw == ZERO_DAY, if_preds == 1), 4)

    # -----------------------------------------------------
    # 5. SOTA INFERENCE (Standard RF)
    # -----------------------------------------------------
    start_time_rf = time.perf_counter()
    std_rf_preds = std_rf.predict(X_eval)
    end_time_rf = time.perf_counter()

    rf_latency_us = ((end_time_rf - start_time_rf) / len(X_eval)) * 1_000_000
    rf_b_recall = round(recall_score(y_eval_raw == "BENIGN", std_rf_preds == "BENIGN"), 4)
    rf_z_recall = round(recall_score(y_eval_raw == ZERO_DAY, std_rf_preds == ZERO_DAY), 4)

    # -----------------------------------------------------
    # 6. HYBRID INFERENCE
    # -----------------------------------------------------
    start_time_hybrid = time.perf_counter()
    residual_list = []

    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), BATCH_SIZE):
            batch = X_eval_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = hybrid_model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_eval = torch.cat(residual_list).numpy()
    variance_eval = pd.Series(residual_eval).rolling(window=15, min_periods=1).var().fillna(0).values
    X_eval_aug = np.hstack([X_eval, residual_eval.reshape(-1, 1), variance_eval.reshape(-1, 1)])

    hybrid_rf_preds = hybrid_rf.predict(X_eval_aug)
    hybrid_rf_probs = np.max(hybrid_rf.predict_proba(X_eval_aug), axis=1)

    hybrid_preds = []
    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    for i in range(len(X_eval)):
        residual = residual_eval[i]
        if i > 0 and i % 1000 == 0:
            threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

        rf_pred = hybrid_rf_preds[i]
        rf_prob = hybrid_rf_probs[i]

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
    end_time_hybrid = time.perf_counter()

    hybrid_latency_us = ((end_time_hybrid - start_time_hybrid) / len(X_eval)) * 1_000_000
    hybrid_z_recall = round(recall_score(y_eval_raw == ZERO_DAY, hybrid_preds == "ZERO_DAY"), 4)
    hybrid_b_recall = round(recall_score(y_eval_raw == "BENIGN", hybrid_preds == "BENIGN"), 4)
    

    print(f"-> [iForest] Zero-Day: {if_z_recall} | Benign: {if_b_recall} | Latency: {round(if_latency_us, 2)} us")
    print(f"-> [Std RF]  Zero-Day: {rf_z_recall} | Benign: {rf_b_recall} | Latency: {round(rf_latency_us, 2)} us")
    print(f"-> [Hybrid]  Zero-Day: {hybrid_z_recall} | Benign: {hybrid_b_recall} | Latency: {round(hybrid_latency_us, 2)} us")

    final_results.append({
        "Attack Category": ZERO_DAY,
        "Hybrid Zero-Day": hybrid_z_recall,
        "Std_RF Zero-Day": rf_z_recall,
        "iForest Zero-Day": if_z_recall,
        "Hybrid Benign": hybrid_b_recall,
        "Hybrid Latency (us)": round(hybrid_latency_us, 2),
        "Std_RF Latency (us)": round(rf_latency_us, 2),
        "iForest Latency (us)": round(if_latency_us, 2)
    })

# =========================================================
# FINAL OUTPUT
# =========================================================
print("\n" + "!"*100)
print("FINAL SOTA COMPARISON MATRIX: PROPOSED HYBRID vs TRADITIONAL ML (iForest & Standard RF)")
print("!"*100)
final_df = pd.DataFrame(final_results)
print(final_df.to_string(index=False))