# =========================================================
# FULL ABLATION STUDY: STANDALONE RF vs STANDALONE DAE vs HYBRID
# PROVING SYNERGISTIC CASCADING LOGIC
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

device = torch.device("cuda")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()

for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")

for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ZERO_DAY_LIST = [
    label for label in df["Label"].unique()
    if label != "BENIGN" and label not in EXCLUDE_ATTACKS
]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Tracker for the final table
ablation_results = []

# =========================================================
# ===== TRAIN DAE ONCE USING ALL BENIGN DATA ============
# =========================================================

benign_full = df[df["Label"] == "BENIGN"]

scaler = StandardScaler()
scaler.fit(benign_full.drop("Label", axis=1))
X_benign = scaler.transform(benign_full.drop("Label", axis=1))

X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

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
        recon = self.decoder(z)
        return recon, z

model = DAE(X_benign.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True)

print("Training DAE (once)...")
model.train()

prev_loss = float("inf")

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

    print("Epoch", epoch + 1, "Loss:", round(total_loss, 4))

    if epoch > 5 and abs(prev_loss - total_loss) < 1e-4:
        print("Early stopping triggered")
        break
    prev_loss = total_loss

model.eval()

# =========================================================
# ================= ZERO DAY LOOP =========================
# =========================================================

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n" + "="*80)
    print("ABLATION TEST:", ZERO_DAY)
    print("="*80)

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # =====================================================
    # SLIDING WINDOW INITIALIZATION
    # =====================================================
    residual_memory = deque(maxlen=WINDOW_SIZE)

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _ = model(x)
            residual = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
            residual_memory.extend(residual)

    print("Sliding Window Initialized")

    # =====================================================
    # RF TRAIN
    # =====================================================
    print("Training RF...")

    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]
    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)

    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), BATCH_SIZE):
            batch = X_rf_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_rf = torch.cat(residual_list).numpy()
    variance_rf = pd.Series(residual_rf).rolling(window=15, min_periods=1).var().fillna(0).values

    X_rf_aug = np.hstack([X_rf, residual_rf.reshape(-1, 1), variance_rf.reshape(-1, 1)])

    rf = RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rf.fit(X_rf_aug, y_rf)
    print("RF trained")

    # =====================================================
    # EVALUATION (ALL 3 MODELS SIMULTANEOUSLY)
    # =====================================================
    print("Running Ablation Evaluation...")

    eval_df = pd.concat([
        benign_df.sample(min(300000, len(benign_df)), random_state=seed),
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
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_eval = torch.cat(residual_list).numpy()
    variance_eval = pd.Series(residual_eval).rolling(window=15, min_periods=1).var().fillna(0).values

    X_eval_aug = np.hstack([X_eval, residual_eval.reshape(-1, 1), variance_eval.reshape(-1, 1)])

    rf_preds_raw = rf.predict(X_eval_aug)
    rf_probs_raw = rf.predict_proba(X_eval_aug)

    dae_preds = []
    rf_preds = []
    hybrid_preds = []

    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    for i in range(len(X_eval)):
        residual = residual_eval[i]

        if i > 0 and i % 1000 == 0:
            threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

        rf_pred = rf_preds_raw[i]
        rf_prob = np.max(rf_probs_raw[i])

        # 1. STANDALONE DAE
        dae_final = "ZERO_DAY" if residual > threshold else "BENIGN"
        
        # 2. STANDALONE RF
        rf_final = rf_pred

        # 3. HYBRID
        if residual > threshold:
            if rf_pred == "BENIGN":
                hybrid_final = "BENIGN" if rf_prob >= ALPHA_BENIGN else "ZERO_DAY"
            else:
                hybrid_final = rf_pred if rf_prob >= ALPHA_ATTACK else "ZERO_DAY"
        else:
            hybrid_final = rf_pred

        dae_preds.append(dae_final)
        rf_preds.append(rf_final)
        hybrid_preds.append(hybrid_final)

        # Update sliding window purely based on Hybrid (Proposed System)
        if hybrid_final == "BENIGN":
            residual_memory.append(residual)

    dae_preds = np.array(dae_preds)
    rf_preds = np.array(rf_preds)
    hybrid_preds = np.array(hybrid_preds)

    # Metric Calculations
    dae_zero = recall_score(y_eval == ZERO_DAY, dae_preds == "ZERO_DAY")
    rf_zero = recall_score(y_eval == ZERO_DAY, rf_preds == "ZERO_DAY")
    hybrid_zero = recall_score(y_eval == ZERO_DAY, hybrid_preds == "ZERO_DAY")

    dae_benign = recall_score(y_eval == "BENIGN", dae_preds == "BENIGN")
    rf_benign = recall_score(y_eval == "BENIGN", rf_preds == "BENIGN")
    hybrid_benign = recall_score(y_eval == "BENIGN", hybrid_preds == "BENIGN")
    
    print(f"Std_RF Zero-Day: {round(rf_zero*100, 2)}% | Std_DAE Zero-Day: {round(dae_zero*100, 2)}% | Hybrid Zero-Day: {round(hybrid_zero*100, 2)}%")

    ablation_results.append({
        "Attack": ZERO_DAY,
        "RF_Zero": rf_zero, "DAE_Zero": dae_zero, "Hybrid_Zero": hybrid_zero,
        "RF_Benign": rf_benign, "DAE_Benign": dae_benign, "Hybrid_Benign": hybrid_benign
    })

# =========================================================
# PRINT FINAL MARKDOWN TABLE FOR PAPER
# =========================================================
print("\n\n" + "="*80)
print("FINAL ABLATION STUDY TABLE (COPY AND PASTE INTO THESIS)")
print("="*80)

print("| Attack Scenario | Standalone RF Zero-Day | Standalone DAE Zero-Day | Hybrid Zero-Day |")
print("| :--- | :--- | :--- | :--- |")
for res in ablation_results:
    print(f"| {res['Attack']} | {res['RF_Zero']:.4f} | {res['DAE_Zero']:.4f} | **{res['Hybrid_Zero']:.4f}** |")

print("| --- | --- | --- | --- |")
avg_rf_benign = np.mean([r['RF_Benign'] for r in ablation_results])
avg_dae_benign = np.mean([r['DAE_Benign'] for r in ablation_results])
avg_hybrid_benign = np.mean([r['Hybrid_Benign'] for r in ablation_results])

print(f"| **Average Benign Recall** | {avg_rf_benign:.4f} | {avg_dae_benign:.4f} | **{avg_hybrid_benign:.4f}** |")
print("="*80 + "\n")