# experiments/anomaly_autoencoder_attackwise_full.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from preprocessing.preprocess_cicids import load_clean_cicids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = load_clean_cicids()

benign_df = df[df["Label"] == "BENIGN"].sample(n=200000, random_state=42)
attack_df = df[df["Label"] != "BENIGN"]

X_train = benign_df.drop("Label", axis=1).values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)

input_dim = X_train.shape[1]

class AE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = AE(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    attack_subset = attack_df[attack_df["Label"] == attack]
    if len(attack_subset) == 0:
        continue

    benign_eval = benign_df.sample(n=len(attack_subset), random_state=42)

    combined = pd.concat([benign_eval, attack_subset])
    y_binary = np.array([0]*len(benign_eval) + [1]*len(attack_subset))

    X_eval = scaler.transform(combined.drop("Label", axis=1).values)
    X_eval = torch.tensor(X_eval, dtype=torch.float32).to(device)

    with torch.no_grad():
        mse = torch.mean((X_eval - model(X_eval))**2, dim=1)

    scores = mse.cpu().numpy()

    fpr, tpr, _ = roc_curve(y_binary, scores)
    roc_auc = auc(fpr, tpr)

    print(f"AUC for {attack} (Full Features): {roc_auc:.4f}")
