# =========================================================
# HIGH-CAPACITY DUAL-RESIDUAL ADAPTIVE HYBRID (CICIoT2023)
# HARDWARE ACCELERATED (AMP + VRAM CACHE + OPTIMIZED LOOP)
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

# =========================================================
# HIGH CAPACITY PARAMETERS (MICRO-TUNED)
# =========================================================

WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 90
ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85

EPOCHS = 35
RF_SAMPLE_SIZE = None   # FULL training data
BATCH_SIZE = 4096       # Optimal for sharp gradient descent

# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_ciciot2023()

# Memory optimization
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")

for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# CICIoT2023 ZERO-DAY BENCHMARK ROSTER
# =========================================================

ZERO_DAY_LIST = [
    # 1. The Volumetric Floods (DAE Magnitude Tests)
    "DDoS-ICMP_Flood",      # Classic IoT Ping Flood
    "DDoS-UDP_Flood",       # Proves protocol-agnostic volumetric detection

    # 2. The Decentralized Botnets (Concept Drift Tests)
    "Mirai-greeth_flood",   # Advanced IoT Botnet behavior
    "Mirai-udpplain",       # Standard Mirai botnet variant

    # 3. The Robotic / Stealth Scripts (Temporal Variance Tests)
    "DoS-HTTP_Flood",       # Application layer resource exhaustion
    "DictionaryBruteForce", # Telnet/SSH password cracking (Perfect flat variance)
    
    # 4. IoT Specific Lateral Movement
    "MITM-ArpSpoofing",     # Tests how the model handles internal network deception

    # 5. Early Warning / Reconnaissance
    "Recon-PortScan",       # Standard port probing
    "VulnerabilityScan",    # Slower, more methodical IoT scanning

    # 6. The Boundary Test (The "Honest Failure")
    "SqlInjection"          # Payload-based attack to prove your model doesn't overfit
]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n" + "="*80)
    print("TEST:", ZERO_DAY)
    print("="*80)

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # =====================================================
    # SCALER
    # =====================================================

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))
    
    X_benign = scaler.transform(benign_df.drop("Label", axis=1))

    # 🚀 SPEED HACK 1: Move entire dataset to GPU VRAM immediately
    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32).to(device)

    # =====================================================
    # DAE MODEL
    # =====================================================

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
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # =====================================================
    # TRAIN DAE (WITH AMP)
    # =====================================================

    print("Training DAE...")
    model.train()

    # 🚀 SPEED HACK 2: Automatic Mixed Precision (AMP)
    scaler_amp = torch.amp.GradScaler('cuda') 

    for epoch in range(EPOCHS):
        total_loss = 0
        for (x,) in loader:
            # x is already on device
            noise = torch.randn_like(x) * 0.05
            
            optimizer.zero_grad()
            
            # Forward pass in 16-bit math
            with torch.amp.autocast('cuda'):
                recon, _ = model(x + noise)
                loss = criterion(recon, x)

            # Backward pass via AMP
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            total_loss += loss.item()

        print("Epoch", epoch + 1, "Loss:", round(total_loss, 4))

    model.eval()

    # =====================================================
    # SLIDING WINDOW INITIALIZATION
    # =====================================================

    residual_memory = deque(maxlen=WINDOW_SIZE)

    with torch.no_grad():
        for (x,) in loader:
            recon, _ = model(x)
            residual = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
            residual_memory.extend(residual)

    print("Sliding Window Initialized")

    # =====================================================
    # RF TRAIN FULL DATA
    # =====================================================

    print("Training RF...")

    rf_sample = train_df
    X_rf = scaler.transform(rf_sample.drop("Label", axis=1))
    y_rf = rf_sample["Label"]

    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32).to(device)
    residual_list = []

    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), BATCH_SIZE):
            batch = X_rf_tensor[i:i+BATCH_SIZE]
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_rf = torch.cat(residual_list).numpy()

    # variance feature
    variance_rf = pd.Series(residual_rf).rolling(
        window=15, min_periods=1
    ).var().fillna(0).values

    X_rf_aug = np.hstack([
        X_rf,
        residual_rf.reshape(-1, 1),
        variance_rf.reshape(-1, 1)
    ])

    rf = RandomForestClassifier(
        n_estimators=500,       # 🚀 Tuned to 500
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_rf_aug, y_rf)
    print("RF trained")

    # =====================================================
    # FULL EVALUATION
    # =====================================================

    print("Running evaluation...")

    eval_df = pd.concat([
        benign_df.sample(min(300000, len(benign_df)), random_state=seed),
        zero_df
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

    residual_list = []

    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), BATCH_SIZE):
            batch = X_eval_tensor[i:i+BATCH_SIZE]
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_eval = torch.cat(residual_list).numpy()

    # 🚀 FIXED: Variance window matched to 15
    variance_eval = pd.Series(residual_eval).rolling(
        window=15, min_periods=1
    ).var().fillna(0).values

    X_eval_aug = np.hstack([
        X_eval,
        residual_eval.reshape(-1, 1),
        variance_eval.reshape(-1, 1)
    ])

    rf_preds = rf.predict(X_eval_aug)
    rf_probs = rf.predict_proba(X_eval_aug)

    hybrid_preds = []

    # =====================================================
    # OPTIMIZED HYBRID LOGIC
    # =====================================================

    # 🚀 SPEED HACK 3: Calculate initial threshold outside the loop
    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    for i in range(len(X_eval)):
        residual = residual_eval[i]
        
        # Recalculate threshold only every 1000 items to save massive CPU time
        if i > 0 and i % 1000 == 0:
            threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

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

        if final_pred == "BENIGN":
            residual_memory.append(residual)

    hybrid_preds = np.array(hybrid_preds)

    print(
        "Benign Recall:",
        round(recall_score(y_eval == "BENIGN", hybrid_preds == "BENIGN"), 4)
    )

    print(
        "Zero-Day Recall:",
        round(recall_score(y_eval == ZERO_DAY, hybrid_preds == "ZERO_DAY"), 4)
    )

print("\nHIGH-CAPACITY Hardware-Accelerated Hybrid (CICIoT2023) Completed")