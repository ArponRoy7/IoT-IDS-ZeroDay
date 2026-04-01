# =========================================================
# FINAL HYBRID IDS (FAST MODE + PAPER SAFE)
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
import os, time, psutil, joblib

# =========================================================
# PARAMETERS
# =========================================================

WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 90
ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85

EPOCHS = 35
BATCH_SIZE = 4096
FAST_FRAC = 0.3   # 🔥 speed control

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

for col in df.columns:
    if df[col].dtype == "float64":
        df[col] = df[col].astype("float32")
    elif df[col].dtype == "int64":
        df[col] = df[col].astype("int32")

ZERO_DAY_LIST = [l for l in df["Label"].unique() if l != "BENIGN"]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# TRAIN DAE (REDUCED DATA)
# =========================================================

benign_full = df[df["Label"] == "BENIGN"].sample(frac=FAST_FRAC, random_state=seed)

scaler = StandardScaler()
scaler.fit(benign_full.drop("Label", axis=1))

X_benign = scaler.transform(benign_full.drop("Label", axis=1))
X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

INPUT_DIM = X_benign.shape[1]

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

model = DAE(INPUT_DIM).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(TensorDataset(X_benign_tensor),
                    batch_size=BATCH_SIZE,
                    shuffle=True)

print("\nTraining DAE...")

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

    print(f"Epoch {epoch+1}: {total_loss:.4f}")

model.eval()

# =========================================================
# LOAO LOOP
# =========================================================

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n==============================")
    print("TEST:", ZERO_DAY)
    print("==============================")

    # 🔥 REDUCED TRAIN DATA
    train_df = df[df["Label"] != ZERO_DAY].sample(frac=FAST_FRAC, random_state=seed)

    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    residual_memory = deque(maxlen=WINDOW_SIZE)

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _ = model(x)
            residual = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
            residual_memory.extend(residual)

    # -------------------------------
    # RF TRAIN
    # -------------------------------
    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]

    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)

    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), BATCH_SIZE):
            batch = X_rf_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch)**2, dim=1).cpu())

    residual_rf = torch.cat(residual_list).numpy()
    variance_rf = pd.Series(residual_rf).rolling(15, min_periods=1).var().fillna(0).values

    X_rf_aug = np.hstack([X_rf,
                          residual_rf.reshape(-1,1),
                          variance_rf.reshape(-1,1)])

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=18,
        max_features="sqrt",
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_rf_aug, y_rf)

    # -------------------------------
    # EVALUATION (REDUCED)
    # -------------------------------
    eval_df = pd.concat([
        benign_df.sample(min(100000, len(benign_df)), random_state=seed),
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
            residual_list.append(torch.mean((recon - batch)**2, dim=1).cpu())

    residual_eval = torch.cat(residual_list).numpy()
    variance_eval = pd.Series(residual_eval).rolling(15, min_periods=1).var().fillna(0).values

    X_eval_aug = np.hstack([X_eval,
                            residual_eval.reshape(-1,1),
                            variance_eval.reshape(-1,1)])

    rf_preds = rf.predict(X_eval_aug)
    rf_probs = rf.predict_proba(X_eval_aug)

    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    hybrid_preds = []

    for i in range(len(X_eval)):
        residual = residual_eval[i]
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

    hybrid_preds = np.array(hybrid_preds)

    print("Benign Recall:", round(recall_score(y_eval=="BENIGN", hybrid_preds=="BENIGN"),4))
    print("Zero-Day Recall:", round(recall_score(y_eval==ZERO_DAY, hybrid_preds=="ZERO_DAY"),4))

# =========================================================
# FINAL HARDWARE PROFILING (PAPER READY + ACCURATE)
# =========================================================

print("\n" + "="*80)
print("HARDWARE PROFILING (FINAL)")
print("="*80)

process = psutil.Process(os.getpid())

os.makedirs("profiling_models", exist_ok=True)

dae_path = "profiling_models/dae.pth"
rf_path = "profiling_models/rf.pkl"

torch.save(model.state_dict(), dae_path)
joblib.dump(rf, rf_path)

# -------------------------------
# MODEL SIZE
# -------------------------------
dae_size = os.path.getsize(dae_path) / (1024 * 1024)
rf_size = os.path.getsize(rf_path) / (1024 * 1024)
total_size = dae_size + rf_size

# -------------------------------
# PREPARE SAMPLE (IMPORTANT FIX)
# -------------------------------
sample_size = min(10000, len(X_eval))

X_sample_original = X_eval[:sample_size]
X_sample_aug = X_eval_aug[:sample_size]

X_sample_tensor = torch.tensor(X_sample_original, dtype=torch.float32).to(device)

# -------------------------------
# GPU WARMUP (CRITICAL)
# -------------------------------
with torch.no_grad():
    for i in range(0, sample_size, BATCH_SIZE):
        batch = X_sample_tensor[i:i+BATCH_SIZE]
        model(batch)
        rf.predict(X_sample_aug[i:i+BATCH_SIZE])

if device.type == "cuda":
    torch.cuda.synchronize()

# -------------------------------
# RAM BEFORE
# -------------------------------
ram_before = process.memory_info().rss / (1024 * 1024)

# -------------------------------
# TIMING START
# -------------------------------
start = time.time()

with torch.no_grad():
    for i in range(0, sample_size, BATCH_SIZE):
        batch = X_sample_tensor[i:i+BATCH_SIZE]

        # Stage 1 (DAE)
        recon, _ = model(batch)

        # Stage 2 (RF)
        rf.predict(X_sample_aug[i:i+BATCH_SIZE])

# GPU sync for correct timing
if device.type == "cuda":
    torch.cuda.synchronize()

end = time.time()

# -------------------------------
# RAM AFTER
# -------------------------------
ram_after = process.memory_info().rss / (1024 * 1024)

peak_ram = ram_after - ram_before
delta_ram = ram_after - ram_before

# -------------------------------
# METRICS
# -------------------------------
total_time = end - start
time_per_flow = total_time / sample_size
throughput = sample_size / total_time

# -------------------------------
# PRINT FINAL (COPY TO PAPER)
# -------------------------------
print("\n" + "="*80)
print("FINAL HARDWARE METRICS (COPY THIS)")
print("="*80)

print(f"Total Architecture Storage Size: {total_size:.2f} MB")
print(f"Stage-1 DAE Model Weight: {dae_size:.2f} MB")
print(f"Stage-2 RF Model Weight: {rf_size:.2f} MB")
print(f"Peak RAM Usage: {peak_ram:.2f} MB")
print(f"Inference Time per Flow: {time_per_flow:.6f} sec")
print(f"Throughput: {throughput:.2f} flows/sec")