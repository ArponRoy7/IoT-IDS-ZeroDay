# =========================================================
# HYBRID IDS - DEPLOYMENT METRICS EVALUATION
# ATTACKS: DDoS, Infiltration, PortScan
# MEASURES:
# AE Time, RF Time, Hybrid Time
# CPU Usage, RAM Usage, Model Size
# =========================================================

import time
import os
import gc
import psutil
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocess_cicids import load_clean_cicids

device = torch.device("cpu")
print("Using device:", device)

ATTACKS = ["DDoS", "Infiltration", "PortScan"]

process = psutil.Process(os.getpid())

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()
df.columns = df.columns.str.strip()

eps = 1e-6

if "Flow Duration" in df.columns:
    df["Packets_Per_Second"] = df["Total Fwd Packets"] / (df["Flow Duration"].abs() + eps)
    df["Bytes_Per_Second"] = df["Total Length of Fwd Packets"] / (df["Flow Duration"].abs() + eps)

if "Total Backward Packets" in df.columns:
    df["Fwd_Bwd_Ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"].abs() + eps)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].clip(-1e9, 1e9)

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# USE ONE ATTACK FOR TRAINING (DDoS)
# =========================================================

ZERO_DAY = "DDoS"

train_df = df[df["Label"] != ZERO_DAY]
test_df = df.sample(200000, random_state=42)

benign_df = train_df[train_df["Label"] == "BENIGN"]

scaler = StandardScaler()
scaler.fit(benign_df.drop("Label", axis=1))

X_benign = scaler.transform(benign_df.drop("Label", axis=1))

# =========================================================
# AUTOENCODER
# =========================================================

class AE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = AE(X_benign.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(
    TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
    batch_size=8192,
    shuffle=True
)

# Train once
model.train()
for epoch in range(10):
    for batch in loader:
        x = batch[0]
        recon = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

# =========================================================
# RF TRAINING
# =========================================================

rf_sample = train_df.sample(500000, random_state=42)
X_rf = scaler.transform(rf_sample.drop("Label", axis=1))

with torch.no_grad():
    X_tensor = torch.tensor(X_rf, dtype=torch.float32)
    recon = model(X_tensor)
    residual = torch.mean((recon - X_tensor)**2, dim=1).numpy()

X_rf = np.hstack([X_rf, residual.reshape(-1,1)])
y_rf = rf_sample["Label"].values

rf = RandomForestClassifier(
    n_estimators=300,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_rf, y_rf)

# =========================================================
# MODEL SIZE
# =========================================================

torch.save(model.state_dict(), "ae_model.pth")
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

ae_size = os.path.getsize("ae_model.pth") / (1024*1024)
rf_size = os.path.getsize("rf_model.pkl") / (1024*1024)
total_size = ae_size + rf_size

# =========================================================
# INFERENCE TIMING
# =========================================================

X_eval = scaler.transform(test_df.drop("Label", axis=1))
X_tensor = torch.tensor(X_eval, dtype=torch.float32)

start_ram = process.memory_info().rss / (1024*1024)

cpu_before = psutil.cpu_percent(interval=None)

# AE timing
start = time.perf_counter()
with torch.no_grad():
    recon = model(X_tensor)
    residual = torch.mean((recon - X_tensor)**2, dim=1).numpy()
ae_time = time.perf_counter() - start

# RF timing
X_eval_rf = np.hstack([X_eval, residual.reshape(-1,1)])

start = time.perf_counter()
rf_preds = rf.predict(X_eval_rf)
rf_time = time.perf_counter() - start

hybrid_time = ae_time + rf_time

cpu_after = psutil.cpu_percent(interval=1)
end_ram = process.memory_info().rss / (1024*1024)

peak_ram = max(start_ram, end_ram)

throughput = len(X_eval) / hybrid_time

gpu_mem = 0
if torch.cuda.is_available():
    gpu_mem = torch.cuda.memory_allocated() / (1024*1024)

# =========================================================
# PRINT RESULTS
# =========================================================

print("\n========== DEPLOYMENT METRICS ==========")
print(f"AE Model Size: {ae_size:.2f} MB")
print(f"RF Model Size: {rf_size:.2f} MB")
print(f"Total Hybrid Size: {total_size:.2f} MB\n")

print(f"AE Inference Time: {ae_time:.4f} sec")
print(f"RF Inference Time: {rf_time:.4f} sec")
print(f"Hybrid Total Time: {hybrid_time:.4f} sec")
print(f"Hybrid Throughput: {throughput:.2f} samples/sec\n")

print(f"Peak RAM Usage: {peak_ram:.2f} MB")
print(f"CPU Utilization: {cpu_after:.2f}%")
print(f"GPU Memory Used: {gpu_mem:.2f} MB")