# =========================================================
# UNIFIED SOTA COMPARISON: HYBRID (DAE+RF) vs KITNET
# Tracks Zero-Day Recall, Benign Recall, and Latency (us)
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
BATCH_SIZE = 4096
EXCLUDE_ATTACKS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# DATA PREPARATION
# =========================================================
df = load_clean_cicids()

for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ZERO_DAY_LIST = [label for label in df["Label"].unique() if label != "BENIGN" and label not in EXCLUDE_ATTACKS]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

benign_full = df[df["Label"] == "BENIGN"]
scaler = StandardScaler()
scaler.fit(benign_full.drop("Label", axis=1))

X_benign = scaler.transform(benign_full.drop("Label", axis=1))
X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True)

# =========================================================
# MODEL 1: PROPOSED HYBRID (DAE)
# =========================================================
class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, dim))

    def forward(self, x):
        return self.decoder(self.encoder(x)), None

hybrid_dae = DAE(X_benign.shape[1]).to(device)
optimizer_dae = torch.optim.AdamW(hybrid_dae.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("Training Hybrid DAE...")
hybrid_dae.train()
prev_loss = float("inf")
for epoch in range(EPOCHS):
    total_loss = 0
    for (x,) in loader:
        x = x.to(device)
        noise = torch.randn_like(x) * 0.05
        optimizer_dae.zero_grad()
        recon, _ = hybrid_dae(x + noise)
        loss = criterion(recon, x)
        loss.backward()
        optimizer_dae.step()
        total_loss += loss.item()
    if epoch > 5 and abs(prev_loss - total_loss) < 1e-4: break
    prev_loss = total_loss
hybrid_dae.eval()

# =========================================================
# MODEL 2: KITNET SIMULATION (ENSEMBLE AE)
# =========================================================
class SmallAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, dim))
    def forward(self, x): return self.decoder(self.encoder(x))

class CombinerAE(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(k, 8), nn.ReLU(), nn.Linear(8, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, k))
    def forward(self, x): return self.decoder(self.encoder(x))

NUM_GROUPS = 5
num_features = X_benign.shape[1]
indices = np.arange(num_features)
np.random.shuffle(indices)
split_size = num_features // NUM_GROUPS
FEATURE_SPLITS = [indices[i*split_size:(i+1)*split_size] if i < NUM_GROUPS-1 else indices[i*split_size:] for i in range(NUM_GROUPS)]

ensemble = [SmallAE(len(split)).to(device) for split in FEATURE_SPLITS]
optimizers = [torch.optim.Adam(ae.parameters(), lr=1e-3) for ae in ensemble]
combiner = CombinerAE(NUM_GROUPS).to(device)
combiner_optimizer = torch.optim.Adam(combiner.parameters(), lr=1e-3)

print("Training KitNET Ensemble...")
for idx, ae in enumerate(ensemble):
    for epoch in range(5):
        for (x,) in loader:
            x = x[:, FEATURE_SPLITS[idx]].to(device)
            optimizers[idx].zero_grad()
            loss = criterion(ae(x), x)
            loss.backward()
            optimizers[idx].step()

print("Training KitNET Combiner...")
for epoch in range(5):
    for (x,) in loader:
        x = x.to(device)
        errs = torch.stack([torch.mean((ae(x[:, split]) - x[:, split])**2, dim=1) for ae, split in zip(ensemble, FEATURE_SPLITS)], dim=1)
        combiner_optimizer.zero_grad()
        loss = criterion(combiner(errs), errs)
        loss.backward()
        combiner_optimizer.step()

def kitnet_score(X_tensor):
    scores = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), BATCH_SIZE):
            batch = X_tensor[i:i+BATCH_SIZE].to(device)
            errs = torch.stack([torch.mean((ae(batch[:, split]) - batch[:, split])**2, dim=1) for ae, split in zip(ensemble, FEATURE_SPLITS)], dim=1)
            final_err = torch.mean((combiner(errs) - errs)**2, dim=1)
            scores.append(final_err.cpu())
    return torch.cat(scores).numpy()

# Calculate fixed KitNET threshold from benign data only
print("Calculating Fixed KitNET Baseline Threshold...")
baseline_kitnet_scores = kitnet_score(X_benign_tensor[:50000]) # Sample to save time
KITNET_THRESHOLD = np.percentile(baseline_kitnet_scores, THRESHOLD_PERCENTILE)

# =========================================================
# ZERO DAY EVALUATION LOOP
# =========================================================
final_comparison = []

for ZERO_DAY in ZERO_DAY_LIST:
    print("\n" + "="*70)
    print("TESTING:", ZERO_DAY)
    print("="*70)

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # -----------------------------------------------------
    # PROPOSED HYBRID PREP (RF Training)
    # -----------------------------------------------------
    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]
    
    with torch.no_grad():
        recon_rf, _ = hybrid_dae(torch.tensor(X_rf, dtype=torch.float32).to(device))
        residual_rf = torch.mean((recon_rf - torch.tensor(X_rf, dtype=torch.float32).to(device)) ** 2, dim=1).cpu().numpy()

    variance_rf = pd.Series(residual_rf).rolling(window=15, min_periods=1).var().fillna(0).values
    X_rf_aug = np.hstack([X_rf, residual_rf.reshape(-1, 1), variance_rf.reshape(-1, 1)])

    rf = RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rf.fit(X_rf_aug, y_rf)

    residual_memory = deque(residual_rf[y_rf == "BENIGN"][-WINDOW_SIZE:], maxlen=WINDOW_SIZE)

    # -----------------------------------------------------
    # EVALUATION DATASET
    # -----------------------------------------------------
    eval_df = pd.concat([benign_df.sample(min(300000, len(benign_df)), random_state=seed), zero_df])
    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    # -----------------------------------------------------
    # RUN KITNET
    # -----------------------------------------------------
    start_time_kitnet = time.perf_counter()
    kitnet_residuals = kitnet_score(X_eval_tensor)
    end_time_kitnet = time.perf_counter()
    
    kitnet_latency_us = ((end_time_kitnet - start_time_kitnet) / len(X_eval)) * 1_000_000
    kitnet_preds = np.where(kitnet_residuals > KITNET_THRESHOLD, "ZERO_DAY", "BENIGN")
    
    kitnet_z_recall = round(recall_score(y_eval == ZERO_DAY, kitnet_preds == "ZERO_DAY"), 4)
    kitnet_b_recall = round(recall_score(y_eval == "BENIGN", kitnet_preds == "BENIGN"), 4)

    # -----------------------------------------------------
    # RUN PROPOSED HYBRID
    # -----------------------------------------------------
    start_time_hybrid = time.perf_counter()
    with torch.no_grad():
        recon_eval, _ = hybrid_dae(X_eval_tensor.to(device))
        residual_eval = torch.mean((recon_eval - X_eval_tensor.to(device)) ** 2, dim=1).cpu().numpy()

    variance_eval = pd.Series(residual_eval).rolling(window=15, min_periods=1).var().fillna(0).values
    X_eval_aug = np.hstack([X_eval, residual_eval.reshape(-1, 1), variance_eval.reshape(-1, 1)])
    
    rf_preds = rf.predict(X_eval_aug)
    rf_probs = np.max(rf.predict_proba(X_eval_aug), axis=1)

    hybrid_preds = []
    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    for i in range(len(X_eval)):
        r, p, pr = residual_eval[i], rf_preds[i], rf_probs[i]
        if i > 0 and i % 1000 == 0: threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)
        
        if r > threshold:
            final = "BENIGN" if (p == "BENIGN" and pr >= ALPHA_BENIGN) else "ZERO_DAY"
        else:
            final = p if (p == "BENIGN" or pr >= ALPHA_ATTACK) else "ZERO_DAY"
            
        hybrid_preds.append(final)
        if final == "BENIGN": residual_memory.append(r)
        
    end_time_hybrid = time.perf_counter()
    hybrid_preds = np.array(hybrid_preds)

    hybrid_latency_us = ((end_time_hybrid - start_time_hybrid) / len(X_eval)) * 1_000_000
    hybrid_z_recall = round(recall_score(y_eval == ZERO_DAY, hybrid_preds == "ZERO_DAY"), 4)
    hybrid_b_recall = round(recall_score(y_eval == "BENIGN", hybrid_preds == "BENIGN"), 4)

    print(f"[KitNET] Zero-Day Recall: {kitnet_z_recall} | Latency: {round(kitnet_latency_us, 2)} us")
    print(f"[Hybrid] Zero-Day Recall: {hybrid_z_recall} | Latency: {round(hybrid_latency_us, 2)} us")

    final_comparison.append({
        "Attack": ZERO_DAY,
        "KitNET_Zero_Recall": kitnet_z_recall,
        "Hybrid_Zero_Recall": hybrid_z_recall,
        "KitNET_Benign_Recall": kitnet_b_recall,
        "Hybrid_Benign_Recall": hybrid_b_recall,
        "KitNET_Latency(us)": round(kitnet_latency_us, 2),
        "Hybrid_Latency(us)": round(hybrid_latency_us, 2)
    })

# =========================================================
# FINAL OUTPUT TABLE
# =========================================================
print("\n" + "!"*80)
print("FINAL SOTA COMPARATIVE ANALYSIS MATRIX")
print("!"*80)
results_df = pd.DataFrame(final_comparison)
print(results_df.to_string(index=False))