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
    # 3-STAGE EVALUATION & DIAGNOSTICS
    # =====================================================

    from sklearn.metrics import recall_score, f1_score

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
    
    # Append residual for the RF
    X_eval_rf = np.hstack([X_eval, residual_eval.reshape(-1, 1)])

    # Get hard predictions and probability scores from RF
    rf_preds = rf.predict(X_eval_rf)
    rf_probs = rf.predict_proba(X_eval_rf)

    # -----------------------------------------------------
    # STAGE 1 & 2 (Quick Calculation)
    # -----------------------------------------------------
    dae_preds = np.where(residual_eval > threshold, "ZERO_DAY", "BENIGN")
    
    print(f"\n[TEST 1] PURE RF Zero-Day Recall : {round(recall_score(y_eval == ZERO_DAY, rf_preds == ZERO_DAY), 4)}")
    print(f"[TEST 2] PURE DAE Zero-Day Recall: {round(recall_score(y_eval == ZERO_DAY, dae_preds == 'ZERO_DAY'), 4)}")
    print(f"[TEST 2] PURE DAE Benign Recall  : {round(recall_score(y_eval == 'BENIGN', dae_preds == 'BENIGN'), 4)}")

    # -----------------------------------------------------
    # STAGE 3: HYBRID GATEKEEPER WITH DIAGNOSTICS
    # -----------------------------------------------------
    print("\n[TEST 3] HYBRID GATEKEEPER (Confidence Threshold = 0.90)")
    
    hybrid_preds = []
    diagnostic_logs = [] # To store the 5-line diagnostic

    for i in range(len(X_eval)):
        max_rf_conf = np.max(rf_probs[i])
        
        # CONDITION A: DAE Flags Anomaly (Residual > Threshold)
        if residual_eval[i] > threshold:
            
            if rf_preds[i] == "BENIGN":
                # ASYMMETRIC TRUST: The RF claims it's normal, but DAE disagrees.
                # We demand near-absolute perfection (99%) to let the RF override.
                if max_rf_conf >= 0.99:
                    hybrid_preds.append("BENIGN")
                else:
                    hybrid_preds.append("ZERO_DAY")
            else:
                # The RF predicts a KNOWN ATTACK.
                # Since the DAE also flagged an anomaly, we trust the RF's specific label.
                hybrid_preds.append(rf_preds[i])
                
        # CONDITION B: DAE Says Normal (Residual <= Threshold)
        else:
            # Trust the RF for normal routing and low-residual known attacks
            hybrid_preds.append(rf_preds[i])

        # =================================================
        # THE DIAGNOSTIC TRACKER
        # =================================================
        if y_eval[i] == ZERO_DAY and hybrid_preds[-1] != "ZERO_DAY":
            if len(diagnostic_logs) < 5:
                diagnostic_logs.append(
                    f"  -> RF Guessed: '{rf_preds[i]}' | Confidence: {max_rf_conf*100:.1f}% | DAE Residual: {residual_eval[i]:.6f}"
                )

    hybrid_preds = np.array(hybrid_preds)

    benign_recall_h = recall_score(y_eval == "BENIGN", hybrid_preds == "BENIGN")
    zero_recall_h = recall_score(y_eval == ZERO_DAY, hybrid_preds == "ZERO_DAY")
    hybrid_f1 = f1_score((y_eval == ZERO_DAY).astype(int), (hybrid_preds == "ZERO_DAY").astype(int))

    print(f"- Benign Recall : {round(benign_recall_h, 4)}")
    print(f"- Zero-Day Rec  : {round(zero_recall_h, 4)}")
    print(f"- Overall F1    : {round(hybrid_f1, 4)}")
    
    if len(diagnostic_logs) > 0:
        print("\n[DIAGNOSTIC] Why did we miss the Zero-Day?")
        for log in diagnostic_logs:
            print(log)
    print("="*70)