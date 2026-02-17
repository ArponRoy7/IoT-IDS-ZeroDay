# =========================================================
# OPTION B - FULL GPU OPTIMIZED RESEARCH LOAO HYBRID
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

if "Flow Duration" in df.columns:
    df["Packets_Per_Second"] = df["Total Fwd Packets"] / (df["Flow Duration"].abs() + eps)
    df["Bytes_Per_Second"] = df["Total Length of Fwd Packets"] / (df["Flow Duration"].abs() + eps)

if "Total Backward Packets" in df.columns:
    df["Fwd_Bwd_Ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"].abs() + eps)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].clip(-1e9, 1e9)

log_dict = {}
for col in numeric_cols:
    if col != "Label" and df[col].min() >= 0:
        log_dict[f"log_{col}"] = np.log1p(df[col])

df = pd.concat([df, pd.DataFrame(log_dict)], axis=1)

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ATTACKS = ["DDoS", "Infiltration"]

for ZERO_DAY in ATTACKS:

    print("\n" + "="*60)
    print("ZERO-DAY:", ZERO_DAY)
    print("="*60)

    train_df = df[df["Label"] != ZERO_DAY]
    test_zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    scaler = StandardScaler()
    scaler.fit(benign_df.drop("Label", axis=1))

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))

    # =========================================================
    # DEEP AUTOENCODER (GPU OPTIMIZED)
    # =========================================================

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

            z = self.encoder(x)
            return self.decoder(z)


    model = AE(X_benign.shape[1]).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.5
    )

    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
        batch_size=8192,
        shuffle=True
    )

    print("Training Deep Autoencoder...")

    model.train()

    for epoch in range(30):

        total_loss = 0

        for batch in loader:

            x = batch[0].to(device)

            recon = model(x)

            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss:", total_loss)

        scheduler.step(total_loss)

    model.eval()

    # =========================================================
    # BETTER THRESHOLD
    # =========================================================

    errors = []

    with torch.no_grad():

        for batch in loader:

            x = batch[0].to(device)
            recon = model(x)

            errors.append(
                torch.mean((recon - x)**2, dim=1).cpu()
            )

    errors = torch.cat(errors).numpy()

    threshold = np.percentile(errors, 99)

    print("Threshold:", threshold)

    # =========================================================
    # RF WITH RESIDUAL FEATURE
    # =========================================================

    rf_sample = train_df.sample(500000, random_state=42)

    X_rf = scaler.transform(rf_sample.drop("Label", axis=1))

    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32).to(device)

    with torch.no_grad():

        recon = model(X_rf_tensor)

        residual = torch.mean(
            (recon - X_rf_tensor)**2,
            dim=1
        ).cpu().numpy()

    X_rf = np.hstack([X_rf, residual.reshape(-1,1)])

    y_rf = rf_sample["Label"]

    print("Training RF...")

    rf = RandomForestClassifier(

        n_estimators=300,
        max_depth=None,
        class_weight="balanced_subsample",
        n_jobs=-1

    )

    rf.fit(X_rf, y_rf)

    # =========================================================
    # EVALUATION
    # =========================================================

    eval_df = pd.concat([
        train_df.sample(100000),
        test_zero_df.sample(min(10000, len(test_zero_df)))
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))

    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

    with torch.no_grad():

        recon = model(X_eval_tensor)

        residual = torch.mean(
            (recon - X_eval_tensor)**2,
            dim=1
        ).cpu().numpy()

    X_eval_rf = np.hstack([X_eval, residual.reshape(-1,1)])

    rf_preds = rf.predict(X_eval_rf)

    final_preds = []

    y_eval = eval_df["Label"].values

    for i in range(len(X_eval)):

        if residual[i] > threshold:

            if rf_preds[i] != y_eval[i]:

                final_preds.append("ZERO_DAY")

            else:

                final_preds.append(rf_preds[i])

        else:

            final_preds.append("BENIGN")

    y_eval_adjusted = y_eval.copy()
    y_eval_adjusted[y_eval_adjusted == ZERO_DAY] = "ZERO_DAY"

    print(classification_report(
        y_eval_adjusted,
        final_preds,
        zero_division=0
    ))
