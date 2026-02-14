import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from preprocessing.preprocess_cicids import load_clean_cicids

# ==============================
# Device
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# Load Dataset
# ==============================

print("Loading dataset...")
df = load_clean_cicids()

# Downsample BENIGN
benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

print("Benign training shape:", benign_df.shape)

# ==============================
# Prepare Data
# ==============================

X_train = benign_df.drop("Label", axis=1).values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)

input_dim = X_train.shape[1]
bottleneck_dim = 8

# ==============================
# Autoencoder Model
# ==============================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


model = Autoencoder(input_dim, bottleneck_dim).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==============================
# Train Model
# ==============================

print("\nTraining Autoencoder (bottleneck=8)...")

epochs = 20
batch_size = 1024

for epoch in range(epochs):

    permutation = torch.randperm(X_train_tensor.size()[0])
    epoch_loss = 0

    for i in range(0, X_train_tensor.size()[0], batch_size):

        indices = permutation[i:i+batch_size]
        batch = X_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

print("Training complete.")

# ==============================
# Function: Compute AUC per attack
# ==============================

def compute_attack_auc(attack_name):

    attack_df = df[df["Label"] == attack_name]

    if len(attack_df) == 0:
        print(f"No samples for {attack_name}")
        return

    # Combine benign + this attack
    test_df = pd.concat([benign_df, attack_df])

    X_test = test_df.drop("Label", axis=1).values
    y_binary = (test_df["Label"] == attack_name).astype(int).values

    X_test = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructed = model(X_test_tensor)
        errors = torch.mean((reconstructed - X_test_tensor)**2, dim=1)

    errors = errors.cpu().numpy()

    auc_score = roc_auc_score(y_binary, errors)

    print(f"AUC for {attack_name}: {auc_score:.4f}")


# ==============================
# Attack-wise Evaluation
# ==============================

print("\n==============================")
print("ATTACK-WISE AUC (Bottleneck=8)")
print("==============================")

compute_attack_auc("DDoS")
compute_attack_auc("PortScan")
compute_attack_auc("Infiltration")
