import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from preprocessing.preprocess_cicids import load_clean_cicids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# Autoencoder (Bottleneck = 8)
# ===============================

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ===============================
# Load Dataset
# ===============================

print("Loading dataset...")
df = load_clean_cicids()

benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

print("Benign training shape:", benign_df.shape)

X_train = benign_df.drop("Label", axis=1)

# ===============================
# Scaling
# ===============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)

# ===============================
# Train Autoencoder
# ===============================

model = Autoencoder(X_train_tensor.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining Autoencoder (bottleneck=8)...")

epochs = 20
batch_size = 1024

for epoch in range(epochs):

    perm = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):

        indices = perm[i:i+batch_size]
        batch = X_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")

# ===============================
# Compute Training Reconstruction Errors
# ===============================

with torch.no_grad():
    recon = model(X_train_tensor)
    train_errors = torch.mean((recon - X_train_tensor) ** 2, dim=1).cpu().numpy()

# -------------------------------
# Set 1% FPR Threshold
# -------------------------------

threshold = np.percentile(train_errors, 99)
print("\nThreshold for 1% FPR:", threshold)

# ===============================
# Attack-wise Evaluation
# ===============================

attacks = ["DDoS", "PortScan", "Infiltration"]

print("\n==============================")
print("DETECTION RATE @ 1% FPR")
print("==============================")

for attack in attacks:

    attack_df = df[df["Label"] == attack]

    X_attack = attack_df.drop("Label", axis=1)
    X_attack_scaled = scaler.transform(X_attack)
    X_attack_tensor = torch.tensor(X_attack_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        recon_attack = model(X_attack_tensor)
        attack_errors = torch.mean((recon_attack - X_attack_tensor) ** 2, dim=1).cpu().numpy()

    detected = (attack_errors > threshold).sum()
    total = len(attack_errors)

    detection_rate = detected / total

    print(f"\nAttack: {attack}")
    print(f"Total Samples: {total}")
    print(f"Detected: {detected}")
    print(f"Detection Rate: {detection_rate:.4f}")
