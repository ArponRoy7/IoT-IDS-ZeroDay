# =========================================================
# FINAL COMPREHENSIVE ZERO-DAY EVALUATION (CICIoT2023)
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
from preprocessing.preprocess_ciciot2023 import load_clean_ciciot2023
from collections import deque
import random

# =========================================================
# PARAMETERS (FINAL STABILITY TUNED)
# =========================================================
WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 99.5  # Reaching for that 99% Benign Recall
ALPHA_BENIGN = 0.9999 
ALPHA_ATTACK = 0.85
EPOCHS = 35
BATCH_SIZE = 8192 
seed = 42
EXCLUDE_ATTACKS = [] # Fix for NameError

device = torch.device("cpu")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# LOAD DATA & DEFINE COMPLETE ATTACK LIST
# =========================================================
df = load_clean_ciciot2023()

# Dynamically pull every attack label from the dataset
ZERO_DAY_LIST = [label for label in df['Label'].unique() if label != 'BENIGN']
print(f"Total Attacks to Evaluate: {len(ZERO_DAY_LIST)}")

# =========================================================
# TRAIN GLOBAL DAE (ONCE)
# =========================================================
benign_full = df[df["Label"] == "BENIGN"]
scaler = StandardScaler()
scaler.fit(benign_full.drop("Label", axis=1))

X_benign = scaler.transform(benign_full.drop("Label", axis=1))
X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32).to(device)

class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x):
        recon = self.decoder(self.encoder(x))
        return recon, None

model = DAE(X_benign.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()
loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=4096, shuffle=True)

print("Training Global DAE for IoT Baseline...")
best_loss = float("inf")
best_model_state = None
for epoch in range(EPOCHS):
    total_loss = 0
    model.train()
    for (x,) in loader:
        noise = torch.randn_like(x) * 0.05
        optimizer.zero_grad(); recon, _ = model(x + noise)
        loss = criterion(recon, x); loss.backward(); optimizer.step()
        total_loss += loss.item()
    if total_loss < best_loss:
        best_loss = total_loss
        best_model_state = copy.deepcopy(model.state_dict())
    if (epoch + 1) % 10 == 0: print(f"Epoch {epoch+1} Loss: {round(total_loss, 4)}")

model.load_state_dict(best_model_state); model.eval()

# =========================================================
# MASTER EVALUATION LOOP
# =========================================================
final_results = []

for ZERO_DAY in ZERO_DAY_LIST:
    print(f"\n" + "="*50)
    print(f"TESTING ZERO-DAY: {ZERO_DAY}")
    print("="*50)
    
    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # RF Training on Knowledge Base
    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]
    
    # Calculate Residuals for RF augmentation
    with torch.no_grad():
        X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)
        recon_rf, _ = model(X_rf_tensor)
        res_rf = torch.mean((recon_rf - X_rf_tensor)**2, dim=1).numpy()
    
    var_rf = pd.Series(res_rf).rolling(window=25, min_periods=1).var().fillna(0).values
    X_rf_aug = np.hstack([X_rf, res_rf.reshape(-1, 1), var_rf.reshape(-1, 1)])

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rf.fit(X_rf_aug, y_rf)

    # Evaluation
    eval_df = pd.concat([benign_df.sample(min(300000, len(benign_df)), random_state=seed), zero_df])
    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    with torch.no_grad():
        X_ev_tensor = torch.tensor(X_eval, dtype=torch.float32)
        recon_ev, _ = model(X_ev_tensor)
        res_ev = torch.mean((recon_ev - X_ev_tensor)**2, dim=1).numpy()

    var_ev = pd.Series(res_ev).rolling(window=25, min_periods=1).var().fillna(0).values
    X_eval_aug = np.hstack([X_eval, res_ev.reshape(-1, 1), var_ev.reshape(-1, 1)])

    rf_preds = rf.predict(X_eval_aug)
    rf_probs = np.max(rf.predict_proba(X_eval_aug), axis=1)

    # Threshold Initialization (Adaptive)
    residual_memory = deque(res_rf[y_rf == "BENIGN"][-WINDOW_SIZE:], maxlen=WINDOW_SIZE)
    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    hybrid_preds = []
    for i in range(len(X_eval)):
        r, p, pr = res_ev[i], rf_preds[i], rf_probs[i]
        if i % 5000 == 0: threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)
        
        if r > threshold:
            final = "BENIGN" if (p == "BENIGN" and pr >= ALPHA_BENIGN) else "ZERO_DAY"
        else:
            final = p
        
        hybrid_preds.append(final)
        if final == "BENIGN": residual_memory.append(r)

    b_recall = round(recall_score(y_eval == "BENIGN", np.array(hybrid_preds) == "BENIGN"), 4)
    z_recall = round(recall_score(y_eval == ZERO_DAY, np.array(hybrid_preds) == "ZERO_DAY"), 4)
    
    print(f"RESULTS for {ZERO_DAY}: Benign Recall: {b_recall} | Zero-Day Recall: {z_recall}")
    final_results.append({"Attack": ZERO_DAY, "Benign_Recall": b_recall, "ZeroDay_Recall": z_recall})

# Final Table Summary
print("\n" + "!"*50)
print("FINAL CONSOLIDATED RESULTS")
print("!"*50)
results_df = pd.DataFrame(final_results)
print(results_df.to_string(index=False))