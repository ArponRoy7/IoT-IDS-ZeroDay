# =========================================================
# STAGE-1 AUTOENCODER VISUAL ANALYSIS (FINAL PRESENTATION VERSION)
# 3 RUN ROC (MEAN ± STD)
# 1 RUN THRESHOLD ANALYSIS (LOG SCALE + 99% LINE)
# SAVES 2 PLOTS PER ATTACK INTO /Plots/run2
# =========================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from preprocessing.preprocess_cicids import load_clean_cicids
import matplotlib.pyplot as plt
import random
import matplotlib
matplotlib.use("Agg")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ATTACKS = ["Infiltration", "DDoS", "PortScan"]

# Create run2 folder
os.makedirs("Plots/run2", exist_ok=True)

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
# LOOP OVER ATTACKS
# =========================================================

for ZERO_DAY in ATTACKS:

    print("\n==============================")
    print("Processing:", ZERO_DAY)
    print("==============================")

    auc_scores = []
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []

    # ======================================
    # 3 RUNS FOR ROC
    # ======================================

    for run in range(3):

        seed = 42 + run
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        train_df = df[df["Label"] != ZERO_DAY]
        test_zero_df = df[df["Label"] == ZERO_DAY]
        benign_df = train_df[train_df["Label"] == "BENIGN"]

        scaler = StandardScaler()
        scaler.fit(benign_df.drop("Label", axis=1))

        X_benign = scaler.transform(benign_df.drop("Label", axis=1))

        class AE(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.ReLU(),
                    nn.Linear(32, 8)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(8, 32),
                    nn.ReLU(),
                    nn.Linear(32, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, dim)
                )

            def forward(self, x):
                return self.decoder(self.encoder(x))

        model = AE(X_benign.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
            batch_size=8192,
            shuffle=True
        )

        model.train()
        for epoch in range(30):
            for batch in loader:
                x = batch[0].to(device)
                recon = model(x)
                loss = criterion(recon, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()

        eval_df = pd.concat([
            benign_df.sample(min(10000, len(benign_df)), random_state=seed),
            test_zero_df.sample(min(10000, len(test_zero_df)), random_state=seed)
        ])

        X_eval = scaler.transform(eval_df.drop("Label", axis=1))
        y_eval = (eval_df["Label"] == ZERO_DAY).astype(int).values

        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

        with torch.no_grad():
            recon = model(X_eval_tensor)
            residual = torch.mean((recon - X_eval_tensor)**2, dim=1).cpu().numpy()

        fpr, tpr, thresholds = roc_curve(y_eval, residual)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # ======================================
    # SAVE ROC (MEAN)
    # ======================================

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr,
             label=f"AUC = {mean_auc:.3f} ± {std_auc:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("Benign False Positive Rate")
    plt.ylabel("Zero-Day Detection Rate")
    plt.title(f"ROC Curve - {ZERO_DAY}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"Plots/run2/roc_mean3_{ZERO_DAY.lower().replace(' ','_')}.png", dpi=300)
    plt.close()

    # ======================================
    # THRESHOLD ANALYSIS (LOG SCALE + 99% LINE)
    # ======================================

    # Compute 99th percentile threshold from benign reconstruction errors
    benign_errors = residual[y_eval == 0]
    threshold = np.percentile(benign_errors, 99)

    thresholds_safe = thresholds + 1e-12

    plt.figure()
    plt.plot(np.log10(thresholds_safe), tpr, label="Zero-Day Detection Rate")
    plt.plot(np.log10(thresholds_safe), fpr, label="Benign False Positive Rate")

    # Vertical line at chosen threshold
    plt.axvline(np.log10(threshold), linestyle='--', label="99th Percentile Threshold")

    plt.xlabel("log10(Residual Threshold)")
    plt.ylabel("Detection / False Positive Rate")
    plt.title(f"Threshold Tradeoff - {ZERO_DAY}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"Plots/run2/threshold_log_{ZERO_DAY.lower().replace(' ','_')}.png", dpi=300)
    plt.close()

    print(f"Saved ROC and Threshold plot for {ZERO_DAY}")

print("\nAll plots saved inside /Plots/run2 folder.")