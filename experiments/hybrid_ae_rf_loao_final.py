# =========================================================
# FINAL HYBRID IDS (AE + RF) - LOAO VERSION (CORRECTED)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================
df = load_clean_cicids()
df.columns = df.columns.str.strip()

eps = 1e-6

# ---------------- Safe Ratio Features ----------------
if "Flow Duration" in df.columns and "Total Fwd Packets" in df.columns:
    df["Packets_Per_Second"] = df["Total Fwd Packets"] / (df["Flow Duration"].abs() + eps)

if "Flow Duration" in df.columns and "Total Length of Fwd Packets" in df.columns:
    df["Bytes_Per_Second"] = df["Total Length of Fwd Packets"] / (df["Flow Duration"].abs() + eps)

if "Total Fwd Packets" in df.columns and "Total Backward Packets" in df.columns:
    df["Fwd_Bwd_Ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"].abs() + eps)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].clip(-1e9, 1e9)

# ---------------- Safe Log Transform ----------------
log_features = {}
for col in numeric_cols:
    if col != "Label" and df[col].min() >= 0:
        log_features[f"log_{col}"] = np.log1p(df[col])

df = pd.concat([df, pd.DataFrame(log_features)], axis=1)
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# ATTACKS FOR LOAO
# =========================================================
ATTACKS = ["DDoS", "Infiltration", "PortScan"]

for ZERO_DAY in ATTACKS:

    print("\n" + "="*60)
    print("ZERO-DAY ATTACK:", ZERO_DAY)
    print("="*60)

    torch.cuda.empty_cache()

    # -----------------------------------------------------
    # Remove zero-day from training
    # -----------------------------------------------------
    train_df = df[df["Label"] != ZERO_DAY]
    test_zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # -----------------------------------------------------
    # SCALE USING BENIGN ONLY
    # -----------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))

    # -----------------------------------------------------
    # AUTOENCODER
    # -----------------------------------------------------
    class AE(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(dim, 64),
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
                nn.Linear(64, dim)
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = AE(X_benign.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
        batch_size=4096,
        shuffle=True,
        pin_memory=True
    )

    print("Training Autoencoder...")
    model.train()
    for epoch in range(10):
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    # -----------------------------------------------------
    # BENIGN ERROR (BATCHED)
    # -----------------------------------------------------
    benign_loader = DataLoader(
        TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
        batch_size=4096,
        shuffle=False,
        pin_memory=True
    )

    benign_error_list = []

    with torch.no_grad():
        for batch in benign_loader:
            x = batch[0].to(device, non_blocking=True)
            recon = model(x)
            error = torch.mean((recon - x)**2, dim=1)
            benign_error_list.append(error.cpu())

    benign_error = torch.cat(benign_error_list).numpy()
    threshold = np.percentile(benign_error, 97)

    # -----------------------------------------------------
    # TRAIN RANDOM FOREST
    # -----------------------------------------------------
    X_train_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_train_rf = train_df["Label"]

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    print("Training Random Forest...")
    rf.fit(X_train_rf, y_train_rf)

    # -----------------------------------------------------
    # EVALUATION SET
    # -----------------------------------------------------
    eval_df = pd.concat([
        train_df.sample(80000, random_state=42),
        test_zero_df.sample(min(5000, len(test_zero_df)), random_state=42)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    eval_loader = DataLoader(
        TensorDataset(torch.tensor(X_eval, dtype=torch.float32)),
        batch_size=4096,
        shuffle=False,
        pin_memory=True
    )

    eval_error_list = []

    with torch.no_grad():
        for batch in eval_loader:
            x = batch[0].to(device, non_blocking=True)
            recon = model(x)
            error = torch.mean((recon - x)**2, dim=1)
            eval_error_list.append(error.cpu())

    eval_error = torch.cat(eval_error_list).numpy()

    # -----------------------------------------------------
    # HYBRID DECISION (REALISTIC ZERO-DAY LOGIC)
    # -----------------------------------------------------
    final_preds = []

    for i in range(len(X_eval)):

        if eval_error[i] > threshold:
            rf_pred = rf.predict(X_eval[i].reshape(1, -1))[0]

            # If RF says BENIGN after anomaly detection,
            # treat as ZERO_DAY
            if rf_pred == "BENIGN":
                final_preds.append("ZERO_DAY")
            else:
                final_preds.append(rf_pred)

        else:
            final_preds.append("BENIGN")

    final_preds = np.array(final_preds)

    # -----------------------------------------------------
    # Adjust ground truth for proper reporting
    # -----------------------------------------------------
    y_eval_adjusted = y_eval.copy()
    y_eval_adjusted[y_eval_adjusted == ZERO_DAY] = "ZERO_DAY"

    print("\nClassification Report:")
    print(classification_report(y_eval_adjusted, final_preds, zero_division=0))

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    zero_detected = (
        final_preds[y_eval == ZERO_DAY] == "ZERO_DAY"
    ).mean()

    benign_recall = (
        final_preds[y_eval == "BENIGN"] == "BENIGN"
    ).mean()

    print(f"Zero-Day Detection Rate: {zero_detected:.4f}")
    print(f"Benign Recall: {benign_recall:.4f}")
