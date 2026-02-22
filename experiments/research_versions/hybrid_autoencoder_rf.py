import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from preprocessing.preprocess_cicids import load_clean_cicids

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# Autoencoder (Bottleneck = 8)
# =====================================================

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

# =====================================================
# Load Dataset
# =====================================================

print("Loading dataset...")
df = load_clean_cicids()

# Downsample benign for training
benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

attack_df = df[df["Label"] != "BENIGN"]

print("Benign:", benign_df.shape)
print("Attack:", attack_df.shape)

# =====================================================
# Scaling
# =====================================================

scaler = StandardScaler()

X_benign = benign_df.drop("Label", axis=1)
X_benign_scaled = scaler.fit_transform(X_benign)
X_benign_tensor = torch.tensor(
    X_benign_scaled, dtype=torch.float32
).to(device)

# =====================================================
# Train Autoencoder
# =====================================================

model = Autoencoder(X_benign_tensor.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nTraining Autoencoder...")

epochs = 20
batch_size = 1024

for epoch in range(epochs):

    perm = torch.randperm(X_benign_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_benign_tensor.size(0), batch_size):

        idx = perm[i:i+batch_size]
        batch = X_benign_tensor[idx]

        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

print("AE Training complete.")

# =====================================================
# Set Threshold (1% FPR)
# =====================================================

with torch.no_grad():
    recon = model(X_benign_tensor)
    benign_errors = torch.mean(
        (recon - X_benign_tensor) ** 2, dim=1
    ).cpu().numpy()

threshold = np.percentile(benign_errors, 99)
print("Anomaly Threshold (1% FPR):", threshold)

# =====================================================
# Train Random Forest on ALL known data
# =====================================================

print("\nTraining Random Forest...")

X_all = df.drop("Label", axis=1)
y_all = df["Label"]

X_all_scaled = scaler.transform(X_all)

rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_all_scaled, y_all)

print("RF training complete.")

# =====================================================
# Hybrid Evaluation
# =====================================================

print("\nRunning Hybrid Evaluation...")

# Sample evaluation set
eval_df = df.sample(n=100000, random_state=42)

X_eval = eval_df.drop("Label", axis=1)
y_eval = eval_df["Label"]

X_eval_scaled = scaler.transform(X_eval)
X_eval_tensor = torch.tensor(
    X_eval_scaled, dtype=torch.float32
).to(device)

# Stage 1: Anomaly Detection
with torch.no_grad():
    recon_eval = model(X_eval_tensor)
    eval_errors = torch.mean(
        (recon_eval - X_eval_tensor) ** 2,
        dim=1
    ).cpu().numpy()

anomaly_mask = eval_errors > threshold

# Stage 2: Classification
rf_preds = rf.predict(X_eval_scaled)

# Hybrid Decision
hybrid_preds = []

for i in range(len(eval_df)):
    if anomaly_mask[i]:
        hybrid_preds.append("ANOMALY")
    else:
        hybrid_preds.append(rf_preds[i])

# Convert to pandas series
hybrid_preds = pd.Series(hybrid_preds)

# =====================================================
# Evaluation Metrics
# =====================================================

print("\nHybrid Classification Report:")
print(classification_report(
    y_eval,
    hybrid_preds,
    zero_division=0
))

# =====================================================
# Zero-Day Specific Check (PortScan example)
# =====================================================

portscan_mask = (y_eval == "PortScan")
if portscan_mask.sum() > 0:
    detected_portscan = (
        hybrid_preds[portscan_mask] == "ANOMALY"
    ).sum()
    total_portscan = portscan_mask.sum()

    print("\nPortScan Recovery (Hybrid):")
    print(f"Detected: {detected_portscan}/{total_portscan}")
    print(f"Rate: {detected_portscan/total_portscan:.4f}")
