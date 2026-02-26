# =========================================================
# HYBRID LOAO – MULTI-THRESHOLD COMPARISON
# Threshold Sensitivity Study (95%, 90%, 85%)
# Logic UNCHANGED
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

# Only structural / important attacks for table
ATTACK_LIST = [
    "DDoS",
    "Infiltration",
    "DoS GoldenEye",
    "DoS Hulk"
]

THRESHOLDS = [0.95, 0.90, 0.85]

results = []

for ZERO_DAY in ATTACK_LIST:

    print("\n" + "="*80)
    print(f"ZERO-DAY TEST: {ZERO_DAY}")
    print("="*80)

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_df = df[df["Label"] != ZERO_DAY]
    zero_df  = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    # ===============================
    # TRAIN DAE (BENIGN ONLY)
    # ===============================

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
            return self.decoder(z), z

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

    # Reconstruction distribution
    recon_list = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _ = model(x)
            recon_list.append(torch.mean((recon - x)**2, dim=1))

    recon_train = torch.cat(recon_list)

    # ===============================
    # RF TRAINING (UNCHANGED)
    # ===============================

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

    # ===============================
    # EVALUATION DATA
    # ===============================

    eval_df = pd.concat([
        train_df.sample(60000, random_state=seed),
        zero_df.sample(min(10000, len(zero_df)), random_state=seed)
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
    X_eval_aug = np.hstack([X_eval, residual_eval.reshape(-1,1)])

    rf_preds = rf.predict(X_eval_aug)
    rf_probs = rf.predict_proba(X_eval_aug)

    # ===============================
    # LOOP OVER THRESHOLDS
    # ===============================

    for perc in THRESHOLDS:

        threshold = torch.quantile(recon_train, perc).item()

        hybrid_preds = []

        for i in range(len(X_eval)):

            max_rf_conf = np.max(rf_probs[i])

            if residual_eval[i] > threshold:

                if rf_preds[i] == "BENIGN":

                    if max_rf_conf >= 0.995:
                        hybrid_preds.append("BENIGN")
                    else:
                        hybrid_preds.append("ZERO_DAY")

                else:
                    if max_rf_conf >= 0.85:
                        hybrid_preds.append(rf_preds[i])
                    else:
                        hybrid_preds.append("ZERO_DAY")
            else:
                hybrid_preds.append(rf_preds[i])

        hybrid_preds = np.array(hybrid_preds)

        benign_recall = recall_score(
            y_eval == "BENIGN",
            hybrid_preds == "BENIGN"
        )

        zero_recall = recall_score(
            y_eval == ZERO_DAY,
            hybrid_preds == "ZERO_DAY"
        )

        results.append([
            ZERO_DAY,
            f"{int(perc*100)}%",
            round(zero_recall*100,2),
            round(benign_recall*100,2)
        ])

# =========================================================
# PRINT FINAL TABLE
# =========================================================

results_df = pd.DataFrame(
    results,
    columns=["Attack", "Threshold", "Zero-Day Recall (%)", "Benign Recall (%)"]
)

print("\n" + "="*80)
print("FINAL COMPARISON TABLE")
print("="*80)
print(results_df.to_string(index=False))