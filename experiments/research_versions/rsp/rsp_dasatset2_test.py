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

BASE_DIR = "scenarios_ciciot_pi"
DATASET_NAME = "CICIoT-2023"

THRESHOLD_PERCENTILE = 99.7
ALPHA_BENIGN = 0.95
BATCH_SIZE = 1024

device = torch.device("cpu")
torch.set_grad_enabled(False) # 🚀 Global memory save for PyTorch

# =========================
# DAE MODEL
# =========================
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
    return os.path.getsize(filepath) / (1024 * 1024) if os.path.exists(filepath) else 0.0


# =========================
# FAST ROLLING VARIANCE (NUMPY)
# =========================
def rolling_variance(arr, window=25):
    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.var(arr[start:i+1])
    return out


# =========================
# TRACKERS
# =========================
process = psutil.Process()
global_peak_ram = 0

total_size, dae_size, rf_size, param_count = 0, 0, 0, 0
total_latency = 0
latency_count = 0
cpu_usage_samples = []

# Dynamic Variables for LaTeX
input_dim = 0
rf_estimators = 0
rf_depth = 0

print(f"\n===== EDGE GATEWAY HIGH-FIDELITY VERIFICATION: {DATASET_NAME} =====\n")

process.cpu_percent(interval=None)

attack_list = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
attack_list.sort()

for idx, attack in enumerate(attack_list):

    print(f"\n🚀 Running edge inference: {attack}")
    path = os.path.join(BASE_DIR, attack)

    peak_ram_attack = 0

    if idx == 0:
        scaler_size = get_file_size_mb(f"{path}/scaler.pkl")
        rf_size = get_file_size_mb(f"{path}/rf.pkl")
        dae_size = get_file_size_mb(f"{path}/dae.pt")
        total_size = scaler_size + rf_size + dae_size

    scaler = joblib.load(f"{path}/scaler.pkl")
    rf = joblib.load(f"{path}/rf.pkl")
    
    # 🚀 MEMORY FIX 1: Prevent Model Cloning
    rf.n_jobs = 1

    memory = deque(np.load(f"{path}/memory.npy"), maxlen=100000)

    dae = DAE(scaler.mean_.shape[0])
    dae.load_state_dict(torch.load(f"{path}/dae.pt", map_location=device))
    dae.eval()

    if idx == 0:
        param_count = sum(p.numel() for p in dae.parameters())
        input_dim = scaler.mean_.shape[0]
        rf_estimators = rf.n_estimators
        rf_depth = rf.max_depth

    # Metrics
    benign_correct = 0
    benign_total = 0
    attack_correct = 0
    attack_total = 0

    last_24_residuals = []

    start_total = time.time()

    # 🚀 MEMORY FIX 2: Pre-define data types to skip heavy float64 allocations
    sample_cols = pd.read_csv(f"{path}/test.csv", nrows=1).columns
    dtype_dict = {col: np.float32 for col in sample_cols if col != "Label"}

    # 🚀 MEMORY FIX 3: Smaller chunksize for lower resting memory
    df_iterator = pd.read_csv(f"{path}/test.csv", chunksize=4096, dtype=dtype_dict, engine='c')

    for df_chunk in df_iterator:

        # 🚀 MEMORY FIX 4: Immediately strip Pandas overhead and keep pure NumPy
        true_labels_chunk = df_chunk["Label"].values
        features_chunk = df_chunk.drop(columns=["Label"]).values
        del df_chunk # Destroy the dataframe instantly

        for i in range(0, len(features_chunk), BATCH_SIZE):

            samples = features_chunk[i:i+BATCH_SIZE]
            true_labels = true_labels_chunk[i:i+BATCH_SIZE]

            start = time.perf_counter()

            # =========================
            # 1. DAE (float32)
            # =========================
            x = scaler.transform(samples).astype(np.float32)
            x_tensor = torch.tensor(x, dtype=torch.float32)

            recon, _ = dae(x_tensor)
            residuals = torch.mean((recon - x_tensor) ** 2, dim=1).numpy().astype(np.float32)

            # =========================
            # 2. Rolling variance (NUMPY)
            # =========================
            if len(last_24_residuals) > 0:
                combined_res = np.concatenate([last_24_residuals, residuals])
            else:
                combined_res = residuals

            variances = rolling_variance(combined_res, 25)

            if len(last_24_residuals) > 0:
                variances = variances[len(last_24_residuals):]

            last_24_residuals = residuals[-24:]

            # =========================
            # 3. RF input (NO HSTACK)
            # =========================
            x_aug = np.empty((x.shape[0], x.shape[1] + 2), dtype=np.float32)
            x_aug[:, :-2] = x
            x_aug[:, -2] = residuals
            x_aug[:, -1] = variances

            rf_preds = rf.predict(x_aug)
            rf_probs = rf.predict_proba(x_aug).max(axis=1)

            # =========================
            # 4. Hybrid logic
            # =========================
            threshold = np.percentile(memory, THRESHOLD_PERCENTILE)
            final_preds = np.copy(rf_preds)

            over_thresh = residuals > threshold
            is_confident_benign = (rf_preds == "BENIGN") & (rf_probs >= ALPHA_BENIGN)

            final_preds[over_thresh & ~is_confident_benign] = "ZERO_DAY"

            memory.extend(residuals[final_preds == "BENIGN"])

            # =========================
            # Metrics
            # =========================
            for pred, true in zip(final_preds, true_labels):
                if true == "BENIGN":
                    benign_total += 1
                    if pred == "BENIGN":
                        benign_correct += 1
                else:
                    attack_total += 1
                    if pred == "ZERO_DAY":
                        attack_correct += 1

            # =========================
            # Latency
            # =========================
            batch_latency = (time.perf_counter() - start) * 1e6
            lat_per_flow = batch_latency / len(samples)

            total_latency += lat_per_flow * len(samples)
            latency_count += len(samples)

            # =========================
            # RAM tracking
            # =========================
            ram = process.memory_info().rss / 1024**2
            peak_ram_attack = max(peak_ram_attack, ram)
            global_peak_ram = max(global_peak_ram, ram)
            cpu_usage_samples.append(psutil.cpu_percent(interval=None))

            # 🚀 MEMORY FIX 5: Strict internal garbage collection to cap the ceiling
            del samples, true_labels, x, x_tensor, recon, variances, x_aug
            gc.collect() 
            
        del features_chunk, true_labels_chunk
        gc.collect()

    print(f"⏱️ Finished in {round(time.time()-start_total,2)}s")

    b_recall = benign_correct / benign_total if benign_total > 0 else 0
    z_recall = attack_correct / attack_total if attack_total > 0 else 0

    print(f"{attack}: Zero-Day = {round(z_recall*100,2)}% | Benign = {round(b_recall*100,2)}%")
    print(f"📌 {attack} Peak RAM: {round(peak_ram_attack,2)} MB")

    # Cleanup models
    del scaler, rf, dae, memory
    gc.collect()


# =========================
# FINAL METRICS & LATEX OUTPUT
# =========================
avg_cpu = np.mean([c for c in cpu_usage_samples if c > 0])
avg_latency = total_latency / latency_count if latency_count > 0 else 0
throughput = 1e6 / avg_latency if avg_latency > 0 else 0

print("\n" + "="*50)
print(f"📊 FINAL HARDWARE COLUMN FOR {DATASET_NAME}")
print("="*50)
print(f"Input Feature Dimension      & {input_dim} \\\\")
print(f"Stage-1 DAE Parameter Count  & {param_count:,} \\\\")
print(f"Stage-2 RF Structure         & {rf_estimators} Trees ($d$={rf_depth}) \\\\")
print(f"Total Architecture Storage   & {total_size:.2f} MB \\\\")
print(f"Active Inference Peak RAM    & {global_peak_ram:.2f} MB \\\\")
print(f"Inference Latency (per flow) & {avg_latency:.2f} $\\mu$s \\\\")
print(f"Throughput (Flows / second)  & $\\sim${int(throughput):,} \\\\")
print(f"CPU Utilization (Average)    & {avg_cpu:.1f}\\% \\\\")
print("="*50 + "\n")