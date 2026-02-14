import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from preprocessing.preprocess_cicids import load_clean_cicids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# Autoencoder (Bottleneck = 8)
# ==========================================

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
        return self.decoder(self.encoder(x))

# ==========================================
# Load Dataset
# ==========================================

print("Loading dataset...")
df = load_clean_cicids()

benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

print("Benign training shape:", benign_df.shape)

X_train = benign_df.drop("Label", axis=1)

# ==========================================
# Scaling
# ==========================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_train_tensor = torch.tensor(
    X_train_scaled, dtype=torch.float32
).to(device)

# ==========================================
# Train AE
# ==========================================

model = Autoencoder(X_train_tensor.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining Autoencoder...")

epochs = 20
batch_size = 1024

for epoch in range(epochs):

    perm = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):

        idx = perm[i:i+batch_size]
        batch = X_train_tensor[idx]

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")

# ==========================================
# Compute Benign Errors
# ==========================================

with torch.no_grad():
    recon = model(X_train_tensor)
    benign_errors = torch.mean(
        (recon - X_train_tensor) ** 2, dim=1
    ).cpu().numpy()

# ==========================================
# FPR Targets
# ==========================================

fpr_targets = [0.01, 0.02, 0.05]
attacks = ["DDoS", "PortScan", "Infiltration"]

results = []

print("\n==============================")
print("DETECTION vs FPR ANALYSIS")
print("==============================")

for fpr in fpr_targets:

    threshold = np.percentile(benign_errors, 100 - (fpr * 100))
    print(f"\nFPR Target: {fpr*100:.1f}%")
    print("Threshold:", threshold)

    for attack in attacks:

        attack_df = df[df["Label"] == attack]

        X_attack = attack_df.drop("Label", axis=1)
        X_attack_scaled = scaler.transform(X_attack)
        X_attack_tensor = torch.tensor(
            X_attack_scaled, dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            recon_attack = model(X_attack_tensor)
            attack_errors = torch.mean(
                (recon_attack - X_attack_tensor) ** 2,
                dim=1
            ).cpu().numpy()

        detected = (attack_errors > threshold).sum()
        total = len(attack_errors)

        detection_rate = detected / total

        print(f"{attack}: {detection_rate:.4f}")

        results.append({
            "FPR": fpr,
            "Attack": attack,
            "DetectionRate": detection_rate
        })

# ==========================================
# Convert to DataFrame
# ==========================================

results_df = pd.DataFrame(results)

# ==========================================
# Plot Detection vs FPR
# ==========================================

plt.figure(figsize=(8,6))

for attack in attacks:
    subset = results_df[results_df["Attack"] == attack]
    plt.plot(
        subset["FPR"],
        subset["DetectionRate"],
        marker="o",
        label=attack
    )

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("Detection Rate (TPR)")
plt.title("Detection Rate vs FPR (Autoencoder)")
plt.legend()
plt.grid(True)
plt.show()

print("\nFinal Table:")
print(results_df)
