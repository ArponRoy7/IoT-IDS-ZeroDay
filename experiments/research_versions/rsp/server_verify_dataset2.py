# =========================================================
# 16GB EDGE INFERENCE (99% BENIGN TARGET / ANOMALY-FIRST)
# =========================================================

import os
# 🔥 FIX: Force all joblib multi-core worker processes to ignore warnings
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
from sklearn.metrics import recall_score

warnings.filterwarnings("ignore")

BASE_DIR = "scenarios_ciciot_pi"

# =========================================================
# 🔥 HYPERPARAMETERS: OPTIMIZED FOR 16GB PI & 150-TREE RF
# =========================================================
THRESHOLD_PERCENTILE = 99.7  # 🚀 Protects Benign recall by only overriding top 0.3%
ALPHA_BENIGN = 0.95          # 🚀 Trusts the 150-tree RF when 95% confident
BATCH_SIZE = 1024            # Optimized for Pi RAM stability

device = torch.device("cpu")

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

def get_file_size_mb(filepath):
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0.0

all_latencies = []
process = psutil.Process()
peak_ram = 0
total_flows = 0

total_size, dae_size, rf_size, param_count = 0, 0, 0, 0

if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(f"Missing '{BASE_DIR}'. Run the Training Script first!")

print("\n===== EDGE GATEWAY HIGH-FIDELITY VERIFICATION =====\n")

# Baseline CPU tracking
process.cpu_percent(interval=None)

attack_list = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
attack_list.sort()

for idx, attack in enumerate(attack_list): 
    print(f"\n🚀 Running edge inference: {attack}")
    path = os.path.join(BASE_DIR, attack)
    
    if idx == 0:
        scaler_size = get_file_size_mb(f"{path}/scaler.pkl")
        rf_size = get_file_size_mb(f"{path}/rf.pkl")
        dae_size = get_file_size_mb(f"{path}/dae.pt")
        total_size = scaler_size + rf_size + dae_size

    scaler = joblib.load(f"{path}/scaler.pkl")
    rf = joblib.load(f"{path}/rf.pkl")
    memory = deque(np.load(f"{path}/memory.npy"), maxlen=100000)

    df = pd.read_csv(f"{path}/test.csv")
    df = df[[col for col in df.columns if col != "Label"] + ["Label"]]

    dae = DAE(scaler.mean_.shape[0])
    dae.load_state_dict(torch.load(f"{path}/dae.pt", map_location=device))
    dae.eval()
    
    if idx == 0:
        param_count = sum(p.numel() for p in dae.parameters())

    y_true, y_pred = [], []
    last_24_residuals = [] 

    start_total = time.time()

    for i in range(0, len(df), BATCH_SIZE):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        true_labels = batch_df["Label"].values
        samples = batch_df.drop(columns=["Label"]).values
        
        start = time.perf_counter()

        # 1. Vectorized DAE Inference
        x = scaler.transform(samples)
        x_tensor = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            recon, _ = dae(x_tensor)
            residuals = torch.mean((recon - x_tensor) ** 2, dim=1).numpy()

        # 2. Vectorized Rolling Variance
        if len(last_24_residuals) > 0: 
            combined_res = np.concatenate([last_24_residuals, residuals])
        else:
            combined_res = residuals
            
        variances = pd.Series(combined_res).rolling(window=25, min_periods=1).var(ddof=1).fillna(0.0).values
        if len(last_24_residuals) > 0: 
            variances = variances[len(last_24_residuals):]
            
        last_24_residuals = residuals[-24:] 

        # 3. RF Classification
        x_aug = np.hstack([x, residuals.reshape(-1, 1), variances.reshape(-1, 1)])
        rf_preds = rf.predict(x_aug)
        rf_probs = rf.predict_proba(x_aug).max(axis=1)

        # 4. 🔥 UPDATED HYBRID LOGIC
        threshold = np.percentile(memory, THRESHOLD_PERCENTILE)
        final_preds = np.copy(rf_preds)
        
        over_thresh = residuals > threshold
        # Condition: trust the RF more to keep Benign recall high
        is_confident_benign = (rf_preds == "BENIGN") & (rf_probs >= ALPHA_BENIGN)
        
        # Override to ZERO_DAY only if structural anomaly is clear AND RF is unsure
        final_preds[over_thresh & ~is_confident_benign] = "ZERO_DAY"

        # Update sliding window
        memory.extend(residuals[final_preds == "BENIGN"])
            
        y_pred.extend(final_preds)
        y_true.extend(true_labels)

        batch_latency = (time.perf_counter() - start) * 1e6
        all_latencies.extend([batch_latency / len(batch_df)] * len(batch_df))
        total_flows += len(batch_df)
        
        ram = process.memory_info().rss / 1024**2
        peak_ram = max(peak_ram, ram)

        del batch_df, samples, x, x_tensor, recon, variances, x_aug
        gc.collect()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    b_recall = recall_score(y_true == "BENIGN", y_pred == "BENIGN")
    z_recall = recall_score(y_true == attack, y_pred == "ZERO_DAY")

    print(f"⏱️ Finished in {round(time.time()-start_total,2)}s | Zero-Day: {round(z_recall*100,2)}% | Benign: {round(b_recall*100,2)}%")

# Final hardware results for LaTeX table
avg_cpu = process.cpu_percent(interval=None)
avg_latency = np.mean(all_latencies)
throughput = 1e6 / avg_latency if avg_latency > 0 else 0

print("\n" + "="*50)
print("📊 FINAL HARDWARE PROFILING FOR LATEX TABLE")
print("="*50)
print(f"Total Architecture Storage : {total_size:.2f} MB")
print(f"Stage-1 DAE Model Weight   : {dae_size:.2f} MB")
print(f"Stage-2 RF Model Weight    : {rf_size:.2f} MB")
print(f"Stage-1 DAE Parameter Count: {param_count:,}")
print(f"Active Inference Peak RAM  : {peak_ram:.2f} MB")
print(f"Inference Latency (per flow): {avg_latency:.2f} µs")
print(f"Throughput (Flows / second): ~{int(throughput):,}")
print(f"CPU Utilization (Average)  : {avg_cpu:.1f}%")
print("="*50 + "\n")