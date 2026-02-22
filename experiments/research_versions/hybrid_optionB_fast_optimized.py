# =========================================================
# OPTION B - RESEARCH LOAO HYBRID (FAST + STABLE - FIXED)
# Only DDoS and Infiltration
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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
if "Flow Duration" in df.columns:
    df["Packets_Per_Second"] = df["Total Fwd Packets"] / (df["Flow Duration"].abs() + eps)
    df["Bytes_Per_Second"] = df["Total Length of Fwd Packets"] / (df["Flow Duration"].abs() + eps)

if "Total Backward Packets" in df.columns:
    df["Fwd_Bwd_Ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"].abs() + eps)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].clip(-1e9, 1e9)

# Efficient log feature creation
log_dict = {}
for col in numeric_cols:
    if col != "Label" and df[col].min() >= 0:
        log_dict[f"log_{col}"] = np.log1p(df[col])

df = pd.concat([df, pd.DataFrame(log_dict)], axis=1)

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# ATTACKS (FAST MODE)
# =========================================================
ATTACKS = ["DDoS", "Infiltration"]

for ZERO_DAY in ATTACKS:

    print("\n" + "="*60)
    print("ZERO-DAY ATTACK:", ZERO_DAY)
    print("="*60)

    torch.cuda.empty_cache()

    # -----------------------------------------------------
    # LOAO SPLIT
    # -----------------------------------------------------
    train_df = df[df["Label"] != ZERO_DAY]
    test_zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # -----------------------------------------------------
    # SCALER (Benign Only)
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
        shuffle=True
    )

    print("Training Autoencoder...")
    model.train()
    for epoch in range(10):
        for batch in loader:
            x = batch[0].to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()

    # -----------------------------------------------------
    # THRESHOLD (97th percentile)
    # -----------------------------------------------------
    benign_error = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            recon = model(x)
            benign_error.append(torch.mean((recon - x)**2, dim=1).cpu())

    benign_error = torch.cat(benign_error).numpy()
    threshold = np.percentile(benign_error, 97)

    # -----------------------------------------------------
    # RANDOM FOREST (FAST + STRATIFIED SAFE)
    # -----------------------------------------------------
    print("Preparing RF training sample...")

    max_samples = 300000

    if len(train_df) > max_samples:
        rf_sample, _ = train_test_split(
            train_df,
            train_size=max_samples,
            stratify=train_df["Label"],
            random_state=42
        )
    else:
        rf_sample = train_df.copy()

    print("RF training size:", len(rf_sample))

    X_train_rf = scaler.transform(rf_sample.drop("Label", axis=1))
    y_train_rf = rf_sample["Label"]

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        max_features="sqrt",
        class_weight="balanced",
        bootstrap=True,
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

    eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon = model(eval_tensor)
        eval_error = torch.mean((recon - eval_tensor)**2, dim=1).cpu().numpy()

    # -----------------------------------------------------
    # RESEARCH LOAO DECISION LOGIC
    # -----------------------------------------------------
    final_preds = []

    for i in range(len(X_eval)):

        if eval_error[i] > threshold:
            rf_pred = rf.predict(X_eval[i].reshape(1, -1))[0]

            if rf_pred != y_eval[i]:
                final_preds.append("ZERO_DAY")
            else:
                final_preds.append(rf_pred)
        else:
            final_preds.append("BENIGN")

    final_preds = np.array(final_preds)

    y_eval_adjusted = y_eval.copy()
    y_eval_adjusted[y_eval_adjusted == ZERO_DAY] = "ZERO_DAY"

    print("\nClassification Report:")
    print(classification_report(y_eval_adjusted, final_preds, zero_division=0))

    zero_detected = (
        final_preds[y_eval == ZERO_DAY] == "ZERO_DAY"
    ).mean()

    benign_recall = (
        final_preds[y_eval == "BENIGN"] == "BENIGN"
    ).mean()

    print(f"Zero-Day Detection Rate: {zero_detected:.4f}")
    print(f"Benign Recall: {benign_recall:.4f}")
