# =========================================================
# HYBRID LOAO – FULL-SCALE ADAPTIVE SLIDING WINDOW
# CICIoT2023 – 15 Representative Attacks
# Full RF + Full Evaluation + Summary Table
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from preprocessing.preprocess_ciciot2023 import load_clean_ciciot2023
from collections import deque
import random
import time

# =========================================================
# SETTINGS
# =========================================================

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

start_total = time.time()

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_ciciot2023()
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# SELECTED 15 ATTACKS (~15 HOURS RUNTIME)
# =========================================================

ZERO_DAY_LIST = [

    # High Volume
    "DDoS-ICMP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-SYN_Flood",

    # Medium DoS
    "DoS-UDP_Flood",
    "DoS-TCP_Flood",
    "DoS-HTTP_Flood",

    # IoT Botnet
    "Mirai-greeth_flood",
    "Mirai-udpplain",

    # Fragmentation
    "DDoS-UDP_Fragmentation",
    "DDoS-ICMP_Fragmentation",

    # Recon
    "Recon-PortScan",
    "Recon-HostDiscovery",

    # Application Layer
    "SqlInjection",
    "XSS"
]

results_summary = []

# =========================================================
# MAIN LOOP
# =========================================================

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n" + "="*80)
    print(f"FULL-SCALE ZERO-DAY TEST: {ZERO_DAY}")
    print("="*80)

    start_attack = time.time()

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df  = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    # =====================================================
    # TRAIN DENOISING AUTOENCODER (BENIGN ONLY)
    # =====================================================

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))
    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

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

    model = DAE(X_benign.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_benign_tensor),
        batch_size=4096,
        shuffle=True
    )

    model.train()

    for epoch in range(50):
        for (x,) in loader:
            x = x.to(device)
            noise = torch.randn_like(x) * 0.05
            recon, _ = model(x + noise)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    # =====================================================
    # INITIALIZE LARGE SLIDING WINDOW
    # =====================================================

    residual_memory = deque(maxlen=100000)

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _ = model(x)
            residual_batch = torch.mean((recon - x)**2, dim=1).cpu().numpy()
            for r in residual_batch:
                residual_memory.append(r)

    print("Sliding window initialized (100k memory).")

    # =====================================================
    # FULL RANDOM FOREST TRAINING
    # =====================================================

    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]

    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)

    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), 4096):
            x = X_rf_tensor[i:i+4096].to(device)
            recon, _ = model(x)
            residual_list.append(torch.mean((recon - x)**2, dim=1).cpu())

    residual_rf = torch.cat(residual_list).numpy()
    X_rf_aug = np.hstack([X_rf, residual_rf.reshape(-1,1)])

    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )

    print("Training FULL RF...")
    rf.fit(X_rf_aug, y_rf)
    print("RF trained.")

    # =====================================================
    # FULL EVALUATION
    # =====================================================

    eval_df = pd.concat([train_df, zero_df])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), 4096):
            batch = X_eval_tensor[i:i+4096].to(device)
            recon, _ = model(batch)
            batch_residual = torch.mean((recon - batch)**2, dim=1).cpu()
            residual_list.append(batch_residual)

    residual_eval = torch.cat(residual_list).numpy()

    X_eval_aug = np.hstack([X_eval, residual_eval.reshape(-1,1)])
    rf_preds_all = rf.predict(X_eval_aug)

    hybrid_preds = []

    for i in range(len(X_eval)):

        residual = residual_eval[i]
        threshold = np.percentile(residual_memory, 97)

        rf_pred = rf_preds_all[i]

        if residual > threshold:
            if rf_pred == "BENIGN":
                final_pred = "ZERO_DAY"
            else:
                final_pred = rf_pred
        else:
            final_pred = rf_pred

        hybrid_preds.append(final_pred)

        if final_pred == "BENIGN":
            residual_memory.append(residual)

    hybrid_preds = np.array(hybrid_preds)

    benign_recall = recall_score(
        y_eval == "BENIGN",
        hybrid_preds == "BENIGN"
    )

    zero_recall = recall_score(
        y_eval == ZERO_DAY,
        hybrid_preds == "ZERO_DAY"
    )

    attack_time = round((time.time()-start_attack)/60,2)

    print("Benign Recall:", round(benign_recall, 4))
    print("Zero-Day Recall:", round(zero_recall, 4))
    print("Time for attack:", attack_time, "minutes")

    results_summary.append({
        "Attack": ZERO_DAY,
        "Benign_Recall": round(benign_recall, 4),
        "ZeroDay_Recall": round(zero_recall, 4),
        "Time_Minutes": attack_time
    })

# =========================================================
# FINAL SUMMARY TABLE
# =========================================================

print("\n" + "="*80)
print("FINAL SUMMARY TABLE (FULL-SCALE LOAO)")
print("="*80)

summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

print("\nAverage Benign Recall:",
      round(summary_df["Benign_Recall"].mean(), 4))

print("Average Zero-Day Recall:",
      round(summary_df["ZeroDay_Recall"].mean(), 4))

print("Total Runtime (minutes):",
      round((time.time()-start_total)/60,2))