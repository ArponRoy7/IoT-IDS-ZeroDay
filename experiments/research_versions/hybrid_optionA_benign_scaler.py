# OPTION A — BENIGN-ONLY SCALER

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from preprocessing.preprocess_cicids import load_clean_cicids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = load_clean_cicids()
df.columns = df.columns.str.strip()

ZERO_DAY = "DDoS"
print("Zero-Day Attack:", ZERO_DAY)

train_df = df[df["Label"] != ZERO_DAY]
benign_df = train_df[train_df["Label"] == "BENIGN"]

# 🔥 Only change here
scaler = StandardScaler()
scaler.fit(benign_df.drop("Label", axis=1))

X_benign = scaler.transform(benign_df.drop("Label", axis=1))

# Autoencoder (baseline architecture)
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
    batch_size=8192,
    shuffle=True
)

print("Training AE...")
for epoch in range(12):
    total = 0
    for batch in loader:
        x = batch[0].to(device)
        recon = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total:.4f}")

model.eval()
with torch.no_grad():
    benign_tensor = torch.tensor(X_benign, dtype=torch.float32).to(device)
    recon = model(benign_tensor)
    benign_error = torch.mean((recon - benign_tensor)**2, dim=1).cpu().numpy()

# RF baseline
X_rf = scaler.transform(train_df.drop("Label", axis=1))
y_rf = train_df["Label"]

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_rf, y_rf)

# Evaluation
eval_df = pd.concat([
    train_df.sample(80000, random_state=42),
    df[df["Label"] == ZERO_DAY].sample(5000, random_state=42)
])

X_eval = scaler.transform(eval_df.drop("Label", axis=1))
y_eval = eval_df["Label"].values

with torch.no_grad():
    eval_tensor = torch.tensor(X_eval, dtype=torch.float32).to(device)
    recon = model(eval_tensor)
    eval_error = torch.mean((recon - eval_tensor)**2, dim=1).cpu().numpy()

rf_preds = rf.predict(X_eval)

print("\nOPTION A RESULTS")

for fpr in np.linspace(0.005, 0.04, 8):
    threshold = np.percentile(benign_error, 100 - fpr*100)
    hybrid = np.where(eval_error > threshold, "ANOMALY", rf_preds)

    zero_mask = (y_eval == ZERO_DAY)
    zero_rate = (hybrid[zero_mask] == "ANOMALY").mean()

    benign_mask = (y_eval == "BENIGN")
    benign_recall = (hybrid[benign_mask] == "BENIGN").mean()

    print(f"FPR {fpr*100:.2f}% → Zero-Day {zero_rate:.4f} | Benign {benign_recall:.4f}")
