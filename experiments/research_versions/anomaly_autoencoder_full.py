# experiments/anomaly_autoencoder_full.py

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

print("\nLoading dataset...")
df = load_clean_cicids()

benign_df = df[df["Label"] == "BENIGN"].sample(n=200000, random_state=42)
attack_df = df[df["Label"] != "BENIGN"]

X_train = benign_df.drop("Label", axis=1).values
X_test = pd.concat([benign_df, attack_df]).drop("Label", axis=1).values

y_test = (pd.concat([benign_df, attack_df])["Label"] != "BENIGN").astype(int).values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

input_dim = X_train.shape[1]

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 20
batch_size = 1024

print("\nTraining Autoencoder...")

for epoch in range(epochs):
    perm = torch.randperm(X_train.size(0))
    epoch_loss = 0
    for i in range(0, X_train.size(0), batch_size):
        idx = perm[i:i+batch_size]
        batch = X_train[idx]

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")

model.eval()
with torch.no_grad():
    reconstructed = model(X_test)
    mse = torch.mean((X_test - reconstructed)**2, dim=1)

scores = mse.cpu().numpy()

fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

print("\nOverall AUC (Full Features):", roc_auc)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--')
plt.title("Autoencoder ROC (Full Features)")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid()
plt.show()
