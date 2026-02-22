# =========================================================
# OPTION B - FULL GPU LOAO HYBRID (STABLE VERSION)
# ALL ATTACK TYPES
# 3 RUNS EACH + MEAN + STD TABLE
# MEMORY SAFE + GPU STABLE
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocess_cicids import load_clean_cicids
import random
import gc

device = torch.device("cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()
df.columns = df.columns.str.strip()

eps = 1e-6

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

# =========================================================
# DETECT ALL ATTACK TYPES
# =========================================================

ALL_LABELS = df["Label"].unique().tolist()
ATTACKS = [label for label in ALL_LABELS if label != "BENIGN"]

print("\nDetected Attack Types:")
for a in ATTACKS:
    print("-", a)

results_summary = []

# =========================================================
# LOOP OVER ALL ATTACKS
# =========================================================

for ZERO_DAY in ATTACKS:

    print("\n" + "="*80)
    print("ZERO-DAY ATTACK:", ZERO_DAY)
    print("="*80)

    zero_rates = []
    benign_rates = []

    for run in range(3):

        print(f"\n------ RUN {run+1} ------")

        seed = 42 + run
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        train_df = df[df["Label"] != ZERO_DAY]
        test_zero_df = df[df["Label"] == ZERO_DAY]
        benign_df = train_df[train_df["Label"] == "BENIGN"]

        if len(test_zero_df) < 100:
            print("Skipping (too few samples)")
            continue

        scaler = StandardScaler()
        scaler.fit(benign_df.drop("Label", axis=1))

        X_benign = scaler.transform(benign_df.drop("Label", axis=1))

        # =====================================================
        # AUTOENCODER
        # =====================================================

        class AE(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 8)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(8, 32),
                    nn.ReLU(),
                    nn.Linear(32, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Linear(256, dim)
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = AE(X_benign.shape[1]).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()

        benign_loader = DataLoader(
            TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
            batch_size=8192,
            shuffle=True
        )

        model.train()
        for epoch in range(30):
            for batch in benign_loader:
                x = batch[0].to(device)
                recon = model(x)
                loss = criterion(recon, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()

        # =====================================================
        # THRESHOLD (99th percentile)
        # =====================================================

        errors = []
        with torch.no_grad():
            for batch in benign_loader:
                x = batch[0].to(device)
                recon = model(x)
                err = torch.mean((recon - x)**2, dim=1)
                errors.append(err.cpu())

        errors = torch.cat(errors).numpy()
        threshold = np.percentile(errors, 99)

        # =====================================================
        # RANDOM FOREST (BATCHED RESIDUAL)
        # =====================================================

        rf_sample = train_df.sample(min(300000, len(train_df)), random_state=seed)

        X_rf = scaler.transform(rf_sample.drop("Label", axis=1))
        y_rf = rf_sample["Label"].values

        rf_loader = DataLoader(
            torch.tensor(X_rf, dtype=torch.float32),
            batch_size=8192,
            shuffle=False
        )

        residual_list = []

        with torch.no_grad():
            for batch in rf_loader:
                batch = batch.to(device)
                recon = model(batch)
                res = torch.mean((recon - batch)**2, dim=1)
                residual_list.append(res.cpu())

        residual = torch.cat(residual_list).numpy()
        X_rf = np.hstack([X_rf, residual.reshape(-1,1)])

        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed
        )

        rf.fit(X_rf, y_rf)

        # =====================================================
        # EVALUATION (BATCHED)
        # =====================================================

        eval_df = pd.concat([
            train_df.sample(min(100000, len(train_df)), random_state=seed),
            test_zero_df.sample(min(10000, len(test_zero_df)), random_state=seed)
        ])

        X_eval = scaler.transform(eval_df.drop("Label", axis=1))
        y_eval = eval_df["Label"].values

        eval_loader = DataLoader(
            torch.tensor(X_eval, dtype=torch.float32),
            batch_size=8192,
            shuffle=False
        )

        residual_list = []

        with torch.no_grad():
            for batch in eval_loader:
                batch = batch.to(device)
                recon = model(batch)
                res = torch.mean((recon - batch)**2, dim=1)
                residual_list.append(res.cpu())

        residual = torch.cat(residual_list).numpy()
        X_eval_rf = np.hstack([X_eval, residual.reshape(-1,1)])

        rf_preds = rf.predict(X_eval_rf)

        final_preds = []

        for i in range(len(X_eval)):
            if residual[i] > threshold:
                if rf_preds[i] != y_eval[i]:
                    final_preds.append("ZERO_DAY")
                else:
                    final_preds.append(rf_preds[i])
            else:
                final_preds.append("BENIGN")

        zero_rate = (np.array(final_preds)[y_eval == ZERO_DAY] == "ZERO_DAY").mean()
        benign_rate = (np.array(final_preds)[y_eval == "BENIGN"] == "BENIGN").mean()

        zero_rates.append(zero_rate)
        benign_rates.append(benign_rate)

        print("Zero-Day Rate:", round(zero_rate,4))
        print("Benign Recall:", round(benign_rate,4))

        torch.cuda.empty_cache()
        gc.collect()

    if len(zero_rates) == 0:
        continue

    results_summary.append({
        "Attack": ZERO_DAY,
        "ZeroDay_Mean": np.mean(zero_rates),
        "ZeroDay_Std": np.std(zero_rates),
        "Benign_Mean": np.mean(benign_rates),
        "Benign_Std": np.std(benign_rates)
    })

# =========================================================
# FINAL TABLE
# =========================================================

print("\n" + "="*80)
print("FINAL PERFORMANCE TABLE (3 RUNS EACH)")
print("="*80)

summary_df = pd.DataFrame(results_summary)
summary_df = summary_df.sort_values(by="ZeroDay_Mean", ascending=False)

print(summary_df.round(4).to_string(index=False))