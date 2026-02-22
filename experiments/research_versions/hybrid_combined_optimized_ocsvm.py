import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
from preprocessing.preprocess_cicids import load_clean_cicids

# =========================================================
# Device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# Load dataset
# =========================================================
df = load_clean_cicids()
df.columns = df.columns.str.strip()

ZERO_DAY = "DDoS"
print("Zero-Day Attack:", ZERO_DAY)

train_df = df[df["Label"] != ZERO_DAY]
benign_df = train_df[train_df["Label"] == "BENIGN"]

print("Total benign samples:", len(benign_df))

# =========================================================
# Benign-only scaler
# =========================================================
scaler = StandardScaler()
scaler.fit(benign_df.drop("Label", axis=1))

X_benign_scaled = scaler.transform(
    benign_df.drop("Label", axis=1)
)

# =========================================================
# Autoencoder
# =========================================================
X_tensor = torch.tensor(X_benign_scaled, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=8192,
    shuffle=True
)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
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
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

model = Autoencoder(X_tensor.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

print("\nTraining Autoencoder...")
for epoch in range(12):
    total_loss = 0
    for batch in train_loader:
        x = batch[0].to(device)
        recon, _ = model(x)
        loss = criterion(recon, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("AE Training complete.")

# =========================================================
# Compute benign latent vectors
# =========================================================
model.eval()
with torch.no_grad():
    benign_tensor = torch.tensor(
        X_benign_scaled,
        dtype=torch.float32
    ).to(device)

    recon, latent = model(benign_tensor)

    recon_error = torch.mean(
        (recon - benign_tensor) ** 2,
        dim=1
    ).cpu().numpy()

    latent_vectors = latent.cpu().numpy()

# =========================================================
# 🔥 Optimized OCSVM (subsample benign)
# =========================================================
print("\nTraining Optimized One-Class SVM...")

max_samples = 50000  # adjust 30k–50k as needed
if len(latent_vectors) > max_samples:
    idx = np.random.choice(len(latent_vectors), max_samples, replace=False)
    latent_subset = latent_vectors[idx]
else:
    latent_subset = latent_vectors

ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02)
ocsvm.fit(latent_subset)

print("OCSVM Training complete.")

# =========================================================
# Prepare evaluation dataset
# =========================================================
eval_df = pd.concat([
    train_df.sample(80000, random_state=42),
    df[df["Label"] == ZERO_DAY].sample(
        n=min(5000, len(df[df["Label"] == ZERO_DAY])),
        random_state=42
    )
])

X_eval_scaled = scaler.transform(eval_df.drop("Label", axis=1))
y_eval = eval_df["Label"].values

with torch.no_grad():
    eval_tensor = torch.tensor(
        X_eval_scaled,
        dtype=torch.float32
    ).to(device)

    recon_eval, latent_eval = model(eval_tensor)

    eval_recon_error = torch.mean(
        (recon_eval - eval_tensor) ** 2,
        dim=1
    ).cpu().numpy()

    latent_eval = latent_eval.cpu().numpy()

# OCSVM prediction
ocsvm_preds = ocsvm.predict(latent_eval)
ocsvm_anomaly = (ocsvm_preds == -1).astype(int)

# =========================================================
# Residual RF
# =========================================================
print("\nTraining Residual Random Forest...")

X_train_scaled = scaler.transform(train_df.drop("Label", axis=1))

with torch.no_grad():
    train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    recon_train, _ = model(train_tensor)
    train_error = torch.mean(
        (recon_train - train_tensor) ** 2,
        dim=1
    ).cpu().numpy()

X_rf_train = np.hstack([X_train_scaled, train_error.reshape(-1,1)])
y_rf_train = train_df["Label"]

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_rf_train, y_rf_train)

print("RF Training complete.")

# =========================================================
# Hybrid Evaluation
# =========================================================
print("\n================ OPTIMIZED COMBINED HYBRID ================")

X_rf_eval = np.hstack([X_eval_scaled, eval_recon_error.reshape(-1,1)])
rf_preds = rf.predict(X_rf_eval)

fpr_values = np.linspace(0.005, 0.04, 8)

for fpr_target in fpr_values:

    threshold = np.percentile(
        recon_error,
        100 - (fpr_target * 100)
    )

    combined_score = (
        (eval_recon_error - recon_error.mean()) /
        (recon_error.std() + 1e-6)
        + ocsvm_anomaly
    )

    hybrid_preds = np.where(
        combined_score > threshold,
        "ANOMALY",
        rf_preds
    )

    zero_mask = (y_eval == ZERO_DAY)
    zero_rate = (
        (hybrid_preds[zero_mask] == "ANOMALY").mean()
    )

    benign_mask = (y_eval == "BENIGN")
    benign_recall = (
        (hybrid_preds[benign_mask] == "BENIGN").mean()
    )

    print(f"\nFPR {fpr_target*100:.2f}%")
    print(f"Zero-Day Detection: {zero_rate:.4f}")
    print(f"Benign Recall: {benign_recall:.4f}")
