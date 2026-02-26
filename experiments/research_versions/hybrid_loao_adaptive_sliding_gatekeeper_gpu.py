# =========================================================
# HYBRID LOAO – ADAPTIVE SLIDING WINDOW GATEKEEPER
# Dynamic 95th Percentile Threshold
# Strict LOAO | Adaptive Structural Zero-Day Detector
# (LOGIC UNCHANGED — LOOP FIXED)
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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ZERO_DAY_LIST = [
    "DDoS",
    "Infiltration",
    "DoS GoldenEye",
    "DoS Hulk"
]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# LOOP OVER ZERO-DAY ATTACKS
# =========================================================

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n" + "="*80)
    print(f"ADAPTIVE ZERO-DAY TEST: {ZERO_DAY}")
    print("="*80)

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df  = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    # =====================================================
    # TRAIN DAE (BENIGN ONLY)
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
    for epoch in range(20):
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
    # INITIALIZE SLIDING WINDOW MEMORY
    # =====================================================

    residual_memory = deque(maxlen=10000)

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _ = model(x)
            residual_batch = torch.mean((recon - x)**2, dim=1).cpu().numpy()
            for r in residual_batch:
                residual_memory.append(r)

    print("Initialized sliding window with benign residuals.")

    # =====================================================
    # RF TRAINING (UNCHANGED)
    # =====================================================

    rf_sample = train_df.sample(250000, random_state=seed)
    X_rf = scaler.transform(rf_sample.drop("Label", axis=1))
    y_rf = rf_sample["Label"]

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
        n_estimators=400,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_rf_aug, y_rf)
    print("RF trained.")

    # =====================================================
    # EVALUATION
    # =====================================================

    eval_df = pd.concat([
        train_df.sample(60000, random_state=seed),
        zero_df.sample(min(10000, len(zero_df)), random_state=seed)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    hybrid_preds = []

    with torch.no_grad():

        for i in range(len(X_eval_tensor)):

            x = X_eval_tensor[i].unsqueeze(0).to(device)
            recon, _ = model(x)
            residual = torch.mean((recon - x)**2).item()

            current_threshold = np.percentile(residual_memory, 95)

            rf_input = np.hstack([X_eval[i], residual]).reshape(1,-1)
            rf_pred = rf.predict(rf_input)[0]
            rf_prob = np.max(rf.predict_proba(rf_input))

            # ASYMMETRIC GATEKEEPER (UNCHANGED)
            if residual > current_threshold:

                if rf_pred == "BENIGN":
                    if rf_prob >= 0.995:
                        final_pred = "BENIGN"
                    else:
                        final_pred = "ZERO_DAY"
                else:
                    if rf_prob >= 0.85:
                        final_pred = rf_pred
                    else:
                        final_pred = "ZERO_DAY"
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

    print("Benign Recall:", round(benign_recall, 4))
    print("Zero-Day Recall:", round(zero_recall, 4))

print("\nAdaptive Sliding Window Evaluation Completed.")