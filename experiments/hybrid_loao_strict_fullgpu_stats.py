# =========================================================
# HYBRID LOAO STRICT FULL GPU (STATISTICAL VERSION)
# 3 Runs + Mean ± Std
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocessing.preprocess_cicids import load_clean_cicids
import random

# =========================
# CONFIGURATION
# =========================
CONF_THRESHOLD = 0.55
AE_EPOCHS = 25
RF_TREES = 200
MAX_RF_SAMPLES = 500000
N_RUNS = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA ONCE
# =========================================================
df = load_clean_cicids()
df.columns = df.columns.str.strip()
print("Dataset shape:", df.shape)

eps = 1e-6

# =========================================================
# FEATURE ENGINEERING (ONCE)
# =========================================================
if "Flow Duration" in df.columns:
    df["Packets_Per_Second"] = df["Total Fwd Packets"] / (df["Flow Duration"].abs() + eps)
    df["Bytes_Per_Second"] = df["Total Length of Fwd Packets"] / (df["Flow Duration"].abs() + eps)

if "Total Backward Packets" in df.columns:
    df["Fwd_Bwd_Ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"].abs() + eps)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].clip(-1e9, 1e9)

log_dict = {
    f"log_{col}": np.log1p(df[col])
    for col in numeric_cols
    if col != "Label" and df[col].min() >= 0
}

df = pd.concat([df, pd.DataFrame(log_dict)], axis=1)
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ATTACKS = ["DDoS", "Infiltration"]

# =========================================================
# MAIN STATISTICAL LOOP
# =========================================================
for ZERO_DAY in ATTACKS:

    print("\n" + "="*60)
    print("ZERO-DAY ATTACK:", ZERO_DAY)
    print("="*60)

    zero_scores = []
    benign_scores = []

    for run in range(N_RUNS):

        print(f"\n----- RUN {run+1} -----")

        # Different seed each run
        seed = 42 + run
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        train_df = df[df["Label"] != ZERO_DAY].copy()
        test_zero_df = df[df["Label"] == ZERO_DAY].copy()
        benign_df = train_df[train_df["Label"] == "BENIGN"].copy()

        scaler = StandardScaler()
        scaler.fit(benign_df.drop("Label", axis=1))

        X_benign = scaler.transform(benign_df.drop("Label", axis=1))

        # =============================
        # AUTOENCODER
        # =============================
        class AE(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(8, 16),
                    nn.ReLU(),
                    nn.Linear(16, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, dim)
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = AE(X_benign.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
            batch_size=8192,
            shuffle=True
        )

        model.train()
        for epoch in range(AE_EPOCHS):
            for batch in loader:
                x = batch[0].to(device)
                recon = model(x)
                loss = criterion(recon, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()

        # =============================
        # THRESHOLD
        # =============================
        benign_errors = []

        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                recon = model(x)
                benign_errors.append(torch.mean((recon - x)**2, dim=1).cpu())

        benign_errors = torch.cat(benign_errors).numpy()
        threshold = np.percentile(benign_errors, 96)

        # =============================
        # RF TRAINING
        # =============================
        if len(train_df) > MAX_RF_SAMPLES:
            frac = MAX_RF_SAMPLES / len(train_df)
            rf_sample = (
                train_df
                .groupby("Label", group_keys=False)
                .sample(frac=frac, random_state=seed)
                .reset_index(drop=True)
            )
        else:
            rf_sample = train_df.copy()

        X_train_rf = scaler.transform(rf_sample.drop("Label", axis=1))
        y_train_rf = rf_sample["Label"]

        rf = RandomForestClassifier(
            n_estimators=RF_TREES,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed
        )

        rf.fit(X_train_rf, y_train_rf)

        # =============================
        # EVALUATION
        # =============================
        eval_df = pd.concat([
            train_df.sample(100000, random_state=seed),
            test_zero_df.sample(min(10000, len(test_zero_df)), random_state=seed)
        ])

        X_eval = scaler.transform(eval_df.drop("Label", axis=1))
        y_eval = eval_df["Label"].values

        eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

        with torch.no_grad():
            recon = model(eval_tensor)
            eval_error = torch.mean((recon - eval_tensor)**2, dim=1).cpu().numpy()

        final_preds = []

        for i in range(len(X_eval)):

            if eval_error[i] > threshold:
                probs = rf.predict_proba(X_eval[i].reshape(1, -1))[0]
                max_prob = np.max(probs)
                rf_pred = rf.classes_[np.argmax(probs)]

                if rf_pred == "BENIGN" or max_prob < CONF_THRESHOLD:
                    final_preds.append("ZERO_DAY")
                else:
                    final_preds.append(rf_pred)
            else:
                final_preds.append("BENIGN")

        final_preds = np.array(final_preds)

        zero_rate = (final_preds[y_eval == ZERO_DAY] == "ZERO_DAY").mean()
        benign_recall = (final_preds[y_eval == "BENIGN"] == "BENIGN").mean()

        zero_scores.append(zero_rate)
        benign_scores.append(benign_recall)

        print(f"Run {run+1} Zero-Day: {zero_rate:.4f}")
        print(f"Run {run+1} Benign Recall: {benign_recall:.4f}")

    # =============================
    # FINAL STATISTICS
    # =============================
    print("\n===== FINAL STATISTICS =====")
    print(f"Zero-Day Mean: {np.mean(zero_scores):.4f}")
    print(f"Zero-Day Std : {np.std(zero_scores):.4f}")
    print(f"Benign Mean  : {np.mean(benign_scores):.4f}")
    print(f"Benign Std   : {np.std(benign_scores):.4f}")
