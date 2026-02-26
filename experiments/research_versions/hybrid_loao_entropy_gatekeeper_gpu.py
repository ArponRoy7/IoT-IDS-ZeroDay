# =========================================================
# HYBRID LOAO – ENTROPY-BASED GATEKEEPER
# DAE + RF + Entropy Uncertainty
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = load_clean_cicids()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ZERO_DAY = "DDoS"

train_df = df[df["Label"] != ZERO_DAY]
zero_df  = df[df["Label"] == ZERO_DAY]
benign_df = train_df[train_df["Label"] == "BENIGN"]

scaler = StandardScaler()
scaler.fit(benign_df.drop("Label", axis=1))

# ---------------- DAE ----------------

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=4096, shuffle=True)

model.train()
for epoch in range(20):
    for (x,) in loader:
        x = x.to(device)
        recon, _ = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

# Threshold
recon_list = []
with torch.no_grad():
    for (x,) in loader:
        x = x.to(device)
        recon, _ = model(x)
        recon_list.append(torch.mean((recon - x)**2, dim=1))

threshold = torch.quantile(torch.cat(recon_list), 0.95).item()
print("DAE Threshold:", threshold)

# ---------------- RF ----------------

rf_sample = train_df.sample(250000, random_state=42)
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

rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)
rf.fit(X_rf_aug, y_rf)

# ---------------- EVALUATION ----------------

eval_df = pd.concat([
    train_df.sample(60000, random_state=42),
    zero_df.sample(min(10000, len(zero_df)), random_state=42)
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

hybrid_preds = []

for i in range(len(X_eval)):

    entropy = -np.sum(rf_probs[i] * np.log(rf_probs[i] + 1e-10))

    if residual_eval[i] > threshold:

        # If RF highly uncertain → trust DAE
        if entropy > 0.5:
            hybrid_preds.append("ZERO_DAY")
        else:
            hybrid_preds.append(rf_preds[i])
    else:
        hybrid_preds.append(rf_preds[i])

hybrid_preds = np.array(hybrid_preds)

print("Benign Recall:",
      recall_score(y_eval=="BENIGN", hybrid_preds=="BENIGN"))

print("Zero-Day Recall:",
      recall_score(y_eval==ZERO_DAY, hybrid_preds=="ZERO_DAY"))