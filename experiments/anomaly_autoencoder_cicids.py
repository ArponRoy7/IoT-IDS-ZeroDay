import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from preprocessing.preprocess_cicids import load_clean_cicids

# ==========================================================
# Device Configuration (GPU if available)
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# Load Dataset
# ==========================================================

print("\nLoading dataset...")
df = load_clean_cicids()

# Downsample BENIGN for training
benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

attack_df = df[df["Label"] != "BENIGN"]

print("Benign training shape:", benign_df.shape)
print("Attack shape:", attack_df.shape)

# ==========================================================
# Prepare Data
# ==========================================================

X_train = benign_df.drop("Label", axis=1).values
X_test = pd.concat([benign_df, attack_df]).drop("Label", axis=1).values

y_test = (pd.concat([benign_df, attack_df])["Label"] != "BENIGN").astype(int).values

# Scaling is VERY important for neural networks
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

input_dim = X_train.shape[1]

# ==========================================================
# Autoencoder Model
# ==========================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

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
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = Autoencoder(input_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================================
# Training
# ==========================================================

epochs = 20
batch_size = 1024

print("\nTraining Autoencoder...")

for epoch in range(epochs):

    permutation = torch.randperm(X_train.size()[0])

    epoch_loss = 0

    for i in range(0, X_train.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch = X_train[indices]

        optimizer.zero_grad()

        outputs = model(batch)
        loss = criterion(outputs, batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")

# ==========================================================
# Reconstruction Error
# ==========================================================

print("\nComputing reconstruction errors...")

model.eval()

with torch.no_grad():
    reconstructed = model(X_test)
    mse = torch.mean((X_test - reconstructed) ** 2, dim=1)

scores = mse.cpu().numpy()

# Higher error = more anomalous

# ==========================================================
# ROC Curve
# ==========================================================

fpr, tpr, thresholds = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

print("\nAutoencoder AUC:", roc_auc)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Autoencoder")
plt.legend()
plt.grid()
plt.show()

# ==========================================================
# Optional: Threshold Example (1% FPR)
# ==========================================================

target_fpr = 0.01
threshold = np.percentile(scores[y_test == 0], 100 - target_fpr * 100)

y_pred = (scores > threshold).astype(int)

true_positive = ((y_pred == 1) & (y_test == 1)).sum()
total_attack = (y_test == 1).sum()

print("\nDetection Rate at 1% FPR:",
      true_positive / total_attack)
