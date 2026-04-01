# =========================================================
# HIGH-CAPACITY DUAL-RESIDUAL HYBRID + BASELINE COMPARISON
# (OPTIMIZED VERSION - FAST + STABLE)
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
import warnings

warnings.filterwarnings("ignore")

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

# Seed
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# TRAIN DAE ON BENIGN DATA
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

print("\nTraining DAE...")
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

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    if epoch > 5 and abs(prev_loss - total_loss) < 1e-4:
        print("Early stopping")
        break

    prev_loss = total_loss

model.eval()

# =========================================================
# LOAO LOOP
# =========================================================

print("\n" + "="*80)
print("STARTING LOAO EVALUATION + BASELINES")
print("="*80)

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n" + "="*60)
    print(f"TEST ATTACK: {ZERO_DAY}")
    print("="*60)

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    residual_memory = deque(maxlen=WINDOW_SIZE)

    # Populate residual memory
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
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_rf = torch.cat(residual_list).numpy()
    variance_rf = pd.Series(residual_rf).rolling(15, min_periods=1).var().fillna(0).values

    X_rf_aug = np.hstack([
        X_rf,
        residual_rf.reshape(-1, 1),
        variance_rf.reshape(-1, 1)
    ])

    rf = RandomForestClassifier(
        n_estimators=150,              # reduced
        class_weight="balanced_subsample",
        n_jobs=1,                      # IMPORTANT FIX
        random_state=seed
    )

    rf.fit(X_rf_aug, y_rf)

    # -------------------------------
    # EVAL DATA (REDUCED SIZE)
    # -------------------------------
    eval_df = pd.concat([
        benign_df.sample(min(100000, len(benign_df)), random_state=seed),
        zero_df.sample(min(50000, len(zero_df)), random_state=seed)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    # -------------------------------
    # Residuals
    # -------------------------------
    residual_list = []

    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), BATCH_SIZE):
            batch = X_eval_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())

    residual_eval = torch.cat(residual_list).numpy()
    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    # -------------------------------
    # DAE ONLY
    # -------------------------------
    dae_preds = np.where(residual_eval > threshold, "ZERO_DAY", "BENIGN")

    dae_recall = recall_score(
        y_eval == ZERO_DAY,
        dae_preds != "BENIGN"
    )

    # -------------------------------
    # RF ONLY (BATCHED)
    # -------------------------------
    X_eval_aug = np.hstack([
        X_eval,
        residual_eval.reshape(-1, 1),
        pd.Series(residual_eval).rolling(15, min_periods=1).var().fillna(0).values.reshape(-1,1)
    ])

    rf_preds = []
    rf_probs = []

    for i in range(0, len(X_eval_aug), 50000):
        batch = X_eval_aug[i:i+50000]
        rf_preds.append(rf.predict(batch))
        rf_probs.append(rf.predict_proba(batch))

    rf_preds = np.concatenate(rf_preds)
    rf_probs = np.vstack(rf_probs)

    rf_recall = recall_score(
        y_eval == ZERO_DAY,
        rf_preds != "BENIGN"
    )

    # -------------------------------
    # HYBRID
    # -------------------------------
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

        if final_pred == "BENIGN":
            residual_memory.append(residual)

    hybrid_preds = np.array(hybrid_preds)

    hybrid_recall = recall_score(
        y_eval == ZERO_DAY,
        hybrid_preds != "BENIGN"
    )

    benign_recall = recall_score(
        y_eval == "BENIGN",
        hybrid_preds == "BENIGN"
    )

    # -------------------------------
    # PRINT
    # -------------------------------
    print("\n--- RESULT COMPARISON ---")
    print(f"RF Only Recall:   {rf_recall:.4f}")
    print(f"DAE Only Recall:  {dae_recall:.4f}")
    print(f"Hybrid Recall:    {hybrid_recall:.4f}")
    print(f"Benign Recall:    {benign_recall:.4f}")
    print("-" * 50)

print("\nCOMPARISON COMPLETE")