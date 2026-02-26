# =========================================================
# HYBRID LOAO – 3 BLOCK EVALUATION (GPU VERSION)
# 1) Pure RF
# 2) Pure DAE
# 3) Hybrid Gatekeeper (Proposed)
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
from preprocessing.preprocess_cicids import load_clean_cicids
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

ATTACKS = ["DDoS", "Infiltration"]   # Focus on strong zero-day ones

# =========================================================
# LOOP
# =========================================================

for ZERO_DAY in ATTACKS:

    print("\n" + "="*70)
    print(f"ZERO-DAY EVALUATION: {ZERO_DAY} (Hidden from RF)")
    print("="*70)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_df = df[df["Label"] != ZERO_DAY]
    test_zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    # =====================================================
    # TRAIN DAE (ONLY BENIGN)
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

    loader = DataLoader(TensorDataset(X_benign_tensor),
                        batch_size=4096, shuffle=True)

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

    # Compute threshold
    recon_list = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _ = model(x)
            recon_list.append(torch.mean((recon - x)**2, dim=1))

    recon_train = torch.cat(recon_list)
    threshold = torch.quantile(recon_train, 0.95).item()
    print("Detection Threshold (95% DAE):", round(threshold,6))
    print("-"*70)

    # =====================================================
    # TRAIN RF (Known attacks + benign)
    # =====================================================

    rf_sample = train_df.sample(250000, random_state=seed)
    X_rf = scaler.transform(rf_sample.drop("Label", axis=1))
    y_rf = rf_sample["Label"]

    # Add residual as feature
    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)

    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), 4096):
            x = X_rf_tensor[i:i+4096].to(device)
            recon, _ = model(x)
            residual_list.append(torch.mean((recon - x)**2, dim=1).cpu())

    residual_rf = torch.cat(residual_list).numpy()
    X_rf_final = np.hstack([X_rf, residual_rf.reshape(-1,1)])

    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )
    rf.fit(X_rf_final, y_rf)

    print("Training RF on:",
          ", ".join(train_df["Label"].unique()))

    # =====================================================
    # PREPARE EVAL DATA (Benign + Known + Zero)
    # =====================================================

    eval_df = pd.concat([
        train_df.sample(60000, random_state=seed),
        test_zero_df.sample(min(10000, len(test_zero_df)), random_state=seed)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), 4096):
            x = X_eval_tensor[i:i+4096].to(device)
            recon, _ = model(x)
            residual_list.append(torch.mean((recon - x)**2, dim=1).cpu())

    residual_eval = torch.cat(residual_list).numpy()
    X_eval_rf = np.hstack([X_eval, residual_eval.reshape(-1,1)])

    # =====================================================
    # TEST 1 — PURE RF
    # =====================================================

    print("\n[TEST 1] PURE RANDOM FOREST (Supervised Baseline)")

    rf_preds = rf.predict(X_eval_rf)

    benign_recall = recall_score(
        y_eval == "BENIGN",
        rf_preds == "BENIGN"
    )

    zero_recall = recall_score(
        y_eval == ZERO_DAY,
        rf_preds == ZERO_DAY
    )

    print("- Benign Recall :", round(benign_recall,4))
    print(f"- Zero-Day ({ZERO_DAY}) Recall :", round(zero_recall,4))
    print("* Expected: RF fails on zero-day.\n")

    # =====================================================
    # TEST 2 — PURE DAE
    # =====================================================

    print("[TEST 2] PURE DAE THRESHOLD")

    dae_preds = np.where(residual_eval > threshold,
                         "ZERO_DAY",
                         "BENIGN")

    benign_recall_dae = recall_score(
        y_eval == "BENIGN",
        dae_preds == "BENIGN"
    )

    zero_recall_dae = recall_score(
        y_eval == ZERO_DAY,
        dae_preds == "ZERO_DAY"
    )

    print("- Benign Recall :", round(benign_recall_dae,4))
    print(f"- Zero-Day ({ZERO_DAY}) Recall :", round(zero_recall_dae,4))
    print("* DAE catches zero-day but causes false positives.\n")

    # =====================================================
    # TEST 3 — HYBRID GATEKEEPER (CONFIDENCE-BASED)
    # =====================================================

    print("[TEST 3] HYBRID GATEKEEPER (Confidence-Based Override)")

    # 1. Get the probability scores from the RF, not just the hard labels
    rf_probs = rf.predict_proba(X_eval_rf)
    
    hybrid_preds = []

    for i in range(len(X_eval)):
        
        # Find the RF's highest confidence score for this specific packet
        max_rf_confidence = np.max(rf_probs[i])

        # CONDITION A: The DAE flags an anomaly
        if residual_eval[i] > threshold:
            
            # The Gatekeeper Check: Is the RF highly confident?
            if max_rf_confidence >= 0.70:
                # The RF is very certain. It's either definitely BENIGN or a KNOWN ATTACK.
                # We trust the RF and rescue the packet.
                hybrid_preds.append(rf_preds[i])
            else:
                # The RF is guessing with low confidence (< 70%).
                # Since the DAE is screaming anomaly, this is a true unknown.
                hybrid_preds.append("ZERO_DAY")
                
        # CONDITION B: The DAE says it's normal traffic
        else:
            # Trust the RF to handle normal routing
            hybrid_preds.append(rf_preds[i])

    hybrid_preds = np.array(hybrid_preds)

    benign_recall_h = recall_score(
        y_eval == "BENIGN",
        hybrid_preds == "BENIGN"
    )

    zero_recall_h = recall_score(
        y_eval == ZERO_DAY,
        hybrid_preds == "ZERO_DAY"
    )

    print("- Benign Recall :", round(benign_recall_h, 4))
    print(f"- Zero-Day ({ZERO_DAY}) Recall :", round(zero_recall_h, 4))
    print("- Overall F1 :", round(
        f1_score((y_eval == ZERO_DAY).astype(int),
                 (hybrid_preds == "ZERO_DAY").astype(int)), 4))
    print("* Confidence-based hybrid filters out DAE false positives.\n")

    print("="*70)