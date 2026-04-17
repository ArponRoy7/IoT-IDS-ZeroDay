# =========================================================
# EDGE INFERENCE (ULTIMATE RAM OPTIMIZED & DYNAMIC METRICS)
# =========================================================

import os
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import time
import psutil
import warnings
import gc
from collections import deque

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = "scenarios"  
DATASET_NAME = "CICIDS-2017" 
RF_ESTIMATORS = 350
RF_DEPTH = 40
BATCH_SIZE = 4096 
ALPHA_BENIGN = 0.999
THRESHOLD_PERCENTILE = 90

device = torch.device("cpu")
torch.set_grad_enabled(False) # 🚀 Global PyTorch memory freeze

class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# 🚀 MEMORY FIX: Pure NumPy Rolling Variance
def rolling_variance(arr, window=15):
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.var(arr[start:i+1])
    return out

# =========================
# TRACKERS
# =========================
process = psutil.Process()
peak_ram = 0
total_latency = 0
latency_count = 0
cpu_usage_samples = []

print(f"\n🚀 STARTING FULL EVALUATION & PROFILING: {DATASET_NAME}\n")

process.cpu_percent(interval=None)

attack_list = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
attack_list.sort()

for attack in attack_list:
    print(f"🚀 Running attack: {attack}")
    path = os.path.join(BASE_DIR, attack)
    
    scaler = joblib.load(f"{path}/scaler.pkl")
    rf = joblib.load(f"{path}/rf.pkl")
    
    # 🚀 MEMORY FIX: Prevent Model Cloning
    rf.n_jobs = 1 
    
    dae = DAE(scaler.mean_.shape[0])
    dae.load_state_dict(torch.load(f"{path}/dae.pt", map_location='cpu'))
    dae.eval()
    
    memory = deque(np.load(f"{path}/memory.npy"), maxlen=100000)
    
    benign_correct = benign_total = 0
    attack_correct = attack_total = 0
    last_14_residuals = []

    # 🚀 MEMORY FIX: Pre-define data types to skip heavy float64 allocations
    sample_cols = pd.read_csv(f"{path}/test.csv", nrows=1).columns
    dtype_dict = {col: np.float32 for col in sample_cols if col != "Label"}

    # 🚀 MEMORY FIX: Smaller chunksize (4096) for lower resting memory
    df_iterator = pd.read_csv(f"{path}/test.csv", chunksize=4096, dtype=dtype_dict, engine='c')
    
    for df_chunk in df_iterator:
        
        # 🚀 MEMORY FIX: Immediately strip Pandas overhead and keep pure NumPy
        true_labels_chunk = df_chunk["Label"].values
        
        # Drop label and convert to pure numpy
        features_chunk = df_chunk.drop(columns=["Label"]).values
        del df_chunk # Destroy the dataframe instantly
        
        for i in range(0, len(features_chunk), BATCH_SIZE):
            
            samples = features_chunk[i:i+BATCH_SIZE]
            true_labels = true_labels_chunk[i:i+BATCH_SIZE]
            
            start = time.perf_counter()

            # 1. DAE Inference
            x = scaler.transform(samples).astype(np.float32)
            x_tensor = torch.tensor(x, dtype=torch.float32)
            
            recon, _ = dae(x_tensor)
            res = torch.mean((recon - x_tensor) ** 2, dim=1).numpy().astype(np.float32)

            # 2. Rolling Variance (NumPy)
            if len(last_14_residuals) > 0:
                combined = np.concatenate([last_14_residuals, res])
            else:
                combined = res
            
            vars_series = rolling_variance(combined, 15)
            if len(last_14_residuals) > 0:
                variances = vars_series[len(last_14_residuals):]
            else:
                variances = vars_series
            
            last_14_residuals = res[-14:]

            # 🚀 MEMORY FIX: np.empty instead of np.hstack
            x_aug = np.empty((x.shape[0], x.shape[1] + 2), dtype=np.float32)
            x_aug[:, :-2] = x
            x_aug[:, -2] = res
            x_aug[:, -1] = variances

            rf_preds = rf.predict(x_aug)
            rf_probs = rf.predict_proba(x_aug).max(axis=1)

            # 4. Hybrid Logic
            threshold = np.percentile(memory, THRESHOLD_PERCENTILE)
            final_preds = np.copy(rf_preds)
            over_thresh = res > threshold
            confident_benign = (rf_preds == "BENIGN") & (rf_probs >= ALPHA_BENIGN)
            final_preds[over_thresh & ~confident_benign] = "ZERO_DAY"

            # End Profiling
            batch_latency = (time.perf_counter() - start) * 1e6
            total_latency += batch_latency
            latency_count += len(samples)

            # Accuracy Tracking
            for pred, true in zip(final_preds, true_labels):
                if true == "BENIGN":
                    benign_total += 1
                    if pred == "BENIGN": benign_correct += 1
                else:
                    attack_total += 1
                    if pred == "ZERO_DAY": attack_correct += 1

            # Resource Sampling
            ram = process.memory_info().rss / 1024**2
            peak_ram = max(peak_ram, ram)
            cpu_usage_samples.append(psutil.cpu_percent(interval=None))
            
            # 🚀 MEMORY FIX: Strict internal garbage collection
            del samples, true_labels, x, x_tensor, recon, res, variances, x_aug
            gc.collect()
            
        del features_chunk, true_labels_chunk
        gc.collect()

    # Print individual results
    b_rec = (benign_correct / benign_total) * 100 if benign_total > 0 else 0
    z_rec = (attack_correct / attack_total) * 100 if attack_total > 0 else 0
    print(f"{attack}: Zero-Day={round(z_rec, 2)}% | Benign={round(b_rec, 2)}%\n")
    
    del scaler, rf, dae, memory
    gc.collect()

# =========================================================
# FINAL CALCULATIONS & LATEX OUTPUT
# =========================================================
avg_latency = total_latency / latency_count if latency_count > 0 else 0
throughput = 1e6 / avg_latency if avg_latency > 0 else 0
avg_cpu = np.mean([c for c in cpu_usage_samples if c > 0])

sample_path = os.path.join(BASE_DIR, attack_list[0])
scaler_meta = joblib.load(f"{sample_path}/scaler.pkl")
dae_meta = DAE(scaler_meta.mean_.shape[0])
param_count = sum(p.numel() for p in dae_meta.parameters())
def get_size(f): return os.path.getsize(f) / (1024*1024) if os.path.exists(f) else 0.0
storage_total = get_size(f"{sample_path}/rf.pkl") + get_size(f"{sample_path}/dae.pt") + get_size(f"{sample_path}/scaler.pkl")

print("\n" + "="*50)
print(f"📊 FINAL HARDWARE COLUMN FOR {DATASET_NAME}")
print("="*50)
print(f"Input Feature Dimension      & {scaler_meta.mean_.shape[0]} \\\\")
print(f"Stage-1 DAE Parameter Count  & {param_count:,} \\\\")
print(f"Stage-2 RF Structure        & {RF_ESTIMATORS} Trees ($d$={RF_DEPTH}) \\\\")
print(f"Total Architecture Storage   & {storage_total:.2f} MB \\\\")
print(f"Active Inference Peak RAM    & {peak_ram:.2f} MB \\\\")
print(f"Inference Latency (per flow) & {avg_latency:.2f} $\\mu$s \\\\")
print(f"Throughput (Flows / second)  & $\\sim${int(throughput):,} \\\\")
print(f"CPU Utilization (Average)    & {avg_cpu:.1f}\\% \\\\")
print("="*50)