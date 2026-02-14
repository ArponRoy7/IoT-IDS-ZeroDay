# experiments/anomaly_autoencoder_attackwise_selected.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

from preprocessing.preprocess_cicids import load_clean_cicids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = load_clean_cicids()

benign_df = df[df["Label"] == "BENIGN"].sample(n=200000, random_state=42)
attack_df = df[df["Label"] != "BENIGN"]

# -------- Feature Selection --------
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(benign_df.drop("Label", axis=1), np.zeros(len(benign_df)))

importances = rf.feature_importances_
features = benign_df.drop("Label", axis=1).columns

top_idx = np.argsort(importances)[::-1][:25]
selected_features = features[top_idx]

print("Using top 25 features.")

benign_selected = benign_df[selected_features]
attack_selected = attack_df[selected_features]

# -------- Train AE --------
scaler = StandardScaler()
X_train = scaler.fit_transform(benign_selected.values)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)

input_dim = X_train.shape[1]

class AE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = AE(input_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(20):
    perm = torch.randperm(X_train.size(0))
    for i in range(0, X_train.size(0), 1024):
        idx = perm[i:i+1024]
        batch = X_train[idx]

        optimizer.zero_grad()
        loss = criterion(model(batch), batch)
        loss.backward()
        optimizer.step()

print("Training complete.")

model.eval()

attacks = ["DDoS", "PortScan", "Infiltration"]

for attack in attacks:

    attack_subset = attack_selected[attack_df["Label"] == attack]
    if len(attack_subset) == 0:
        continue

    benign_eval = benign_selected.sample(n=len(attack_subset), random_state=42)

    combined = pd.concat([benign_eval, attack_subset])
    y_binary = np.array([0]*len(benign_eval) + [1]*len(attack_subset))

    X_eval = scaler.transform(combined.values)
    X_eval = torch.tensor(X_eval, dtype=torch.float32).to(device)

    with torch.no_grad():
        mse = torch.mean((X_eval - model(X_eval))**2, dim=1)

    scores = mse.cpu().numpy()

    fpr, tpr, _ = roc_curve(y_binary, scores)
    roc_auc = auc(fpr, tpr)

    print(f"AUC for {attack} (Top 25 Features): {roc_auc:.4f}")
