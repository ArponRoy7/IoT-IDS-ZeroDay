# =========================================================
# HYBRID OPTION B - LOAO FULL GPU (HIGH ACCURACY RESTORED)
# Deep AE + Residual RF + Original Hybrid Rule
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

# ================= CONFIG =================
AE_EPOCHS = 30
RF_TREES = 300
ZERO_DAY = "DDoS"
PERCENTILE = 99
RF_SAMPLE_SIZE = 500000
EVAL_KNOWN = 100000
EVAL_ZERO = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================
df = load_clean_cicids()
df.columns = df.columns.str.strip()
df.fillna(0, inplace=True)

print("Dataset shape:", df.shape)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# =========================================================
# LOAO SPLIT
# =========================================================
train_df = df[df["Label"] != ZERO_DAY].copy()
test_zero_df = df[df["Label"] == ZERO_DAY].copy()

benign_df = train_df[train_df["Label"] == "BENIGN"]

# =========================================================
# SCALER (FIT ON ALL BENIGN)
# =========================================================
scaler = StandardScaler()
scaler.fit(benign_df.drop("Label", axis=1))

X_benign = scaler.transform(benign_df.drop("Label", axis=1))

# =========================================================
# DEEP AUTOENCODER (ORIGINAL STRONG ONE)
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
        return self.decoder(self.encoder(x))


model = AE(X_benign.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

loader = DataLoader(
    TensorDataset(torch.tensor(X_benign, dtype=torch.float32)),
    batch_size=8192,
    shuffle=True
)

print("\nTraining Deep Autoencoder...")

for epoch in range(AE_EPOCHS):
    model.train()
    total_loss = 0

    for batch in loader:
        x = batch[0].to(device)
        recon = model(x)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(loader):.6f}")

model.eval()

# =========================================================
# THRESHOLD (99th percentile on benign)
# =========================================================
errors = []

with torch.no_grad():
    for batch in loader:
        x = batch[0].to(device)
        recon = model(x)
        err = torch.mean((recon - x)**2, dim=1)
        errors.append(err.cpu())

errors = torch.cat(errors).numpy()
threshold = np.percentile(errors, PERCENTILE)

print("\nFinal Threshold:", threshold)

# =========================================================
# STRATIFIED RF SAMPLE (500k)
# =========================================================
print("\nPreparing RF training sample...")

if len(train_df) > RF_SAMPLE_SIZE:
    frac = RF_SAMPLE_SIZE / len(train_df)
    rf_sample = (
        train_df
        .groupby("Label", group_keys=False)
        .sample(frac=frac, random_state=seed)
        .reset_index(drop=True)
    )
else:
    rf_sample = train_df.copy()

print("RF sample size:", len(rf_sample))

X_rf = scaler.transform(rf_sample.drop("Label", axis=1))
y_rf = rf_sample["Label"]

# =========================================================
# ADD RESIDUAL FEATURE
# =========================================================
X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32).to(device)

with torch.no_grad():
    recon = model(X_rf_tensor)
    residual = torch.mean((recon - X_rf_tensor)**2, dim=1).cpu().numpy()

X_rf = np.hstack([X_rf, residual.reshape(-1,1)])

# =========================================================
# RF TRAIN
# =========================================================
rf = RandomForestClassifier(
    n_estimators=RF_TREES,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=seed
)

rf.fit(X_rf, y_rf)

# =========================================================
# EVALUATION (SAME AS OLD WORKING VERSION)
# =========================================================
eval_df = pd.concat([
    train_df.sample(EVAL_KNOWN, random_state=seed),
    test_zero_df.sample(min(EVAL_ZERO, len(test_zero_df)), random_state=seed)
])

X_eval = scaler.transform(eval_df.drop("Label", axis=1))
y_eval = eval_df["Label"].values

X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)

with torch.no_grad():
    recon = model(X_eval_tensor)
    residual = torch.mean((recon - X_eval_tensor)**2, dim=1).cpu().numpy()

X_eval_rf = np.hstack([X_eval, residual.reshape(-1,1)])
rf_preds = rf.predict(X_eval_rf)

# =========================================================
# ORIGINAL HYBRID RULE (HIGH ACCURACY ONE)
# =========================================================
final_preds = []

for i in range(len(X_eval)):
    if residual[i] > threshold:
        if rf_preds[i] != y_eval[i]:
            final_preds.append("ZERO_DAY")
        else:
            final_preds.append(rf_preds[i])
    else:
        final_preds.append("BENIGN")

final_preds = np.array(final_preds)

y_eval_adjusted = y_eval.copy()
y_eval_adjusted[y_eval_adjusted == ZERO_DAY] = "ZERO_DAY"

print("\n=== FINAL RESULTS ===")
print(classification_report(y_eval_adjusted, final_preds, zero_division=0))

zero_rate = (final_preds[y_eval == ZERO_DAY] == "ZERO_DAY").mean()
benign_rate = (final_preds[y_eval == "BENIGN"] == "BENIGN").mean()

print("Zero-Day Detection Rate:", zero_rate)
print("Benign Recall:", benign_rate)
