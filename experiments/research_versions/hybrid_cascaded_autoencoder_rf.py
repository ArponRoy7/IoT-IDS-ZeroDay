import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
ZERO_DAY = "DDoS"
print("Zero-Day Attack:", ZERO_DAY)

# =========================================================
# Train data (exclude zero-day)
# =========================================================
train_df = df[df["Label"] != ZERO_DAY]

benign_train = train_df[train_df["Label"] == "BENIGN"]
rf_train_df = train_df

# =========================================================
# Scaling
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(
    rf_train_df.drop("Label", axis=1)
)

# =========================================================
# AUTOENCODER
# =========================================================
X_benign_scaled = scaler.transform(
    benign_train.drop("Label", axis=1)
)

feature_var = np.var(X_benign_scaled, axis=0)
feature_var = np.where(feature_var < 1e-6, 1e-6, feature_var)

# Softer weighting
feature_weight = np.sqrt(feature_var)
weight_tensor = torch.tensor(feature_weight, dtype=torch.float32).to(device)

X_tensor = torch.tensor(
    X_benign_scaled,
    dtype=torch.float32
).to(device)

train_loader = DataLoader(
    TensorDataset(X_tensor),
    batch_size=4096,
    shuffle=True
)

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(X_tensor.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("\nTraining Cascaded Autoencoder...")

for epoch in range(15):
    model.train()
    total_loss = 0
    for batch in train_loader:
        x = batch[0]
        recon = model(x)

        diff = (recon - x) ** 2
        loss = (diff / weight_tensor).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("AE Training complete.")

# =========================================================
# Compute benign reconstruction errors
# =========================================================
model.eval()
with torch.no_grad():
    recon = model(X_tensor)
    benign_errors = torch.mean(
        ((recon - X_tensor) ** 2) / weight_tensor,
        dim=1
    ).cpu().numpy()

# =========================================================
# RANDOM FOREST (Balanced)
# =========================================================
print("\nTraining Balanced Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(
    X_train_scaled,
    rf_train_df["Label"]
)

print("RF Training complete.")

# =========================================================
# Evaluation Dataset
# =========================================================
eval_df = pd.concat([
    train_df.sample(n=80000, random_state=42),
    df[df["Label"] == ZERO_DAY].sample(n=5000, random_state=42)
])

X_eval = scaler.transform(eval_df.drop("Label", axis=1))
y_eval = eval_df["Label"].values

with torch.no_grad():
    X_eval_tensor = torch.tensor(
        X_eval,
        dtype=torch.float32
    ).to(device)

    recon = model(X_eval_tensor)
    eval_errors = torch.mean(
        ((recon - X_eval_tensor) ** 2) / weight_tensor,
        dim=1
    ).cpu().numpy()

rf_preds = rf.predict(X_eval)

# =========================================================
# TWO-THRESHOLD CASCADED HYBRID
# =========================================================
print("\n========================================")
print("CASCADED HYBRID EVALUATION")
print("========================================")

for fpr_target in [0.005, 0.01, 0.02]:

    T_high = np.percentile(
        benign_errors,
        100 - (fpr_target * 100)
    )

    # small safety margin
    T_low = np.percentile(
        benign_errors,
        100 - ((fpr_target * 100) + 1.0)
    )

    hybrid_preds = []

    for i in range(len(eval_errors)):

        if eval_errors[i] > T_high:
            hybrid_preds.append("ANOMALY")

        elif eval_errors[i] < T_low:
            hybrid_preds.append("BENIGN")

        else:
            hybrid_preds.append(rf_preds[i])

    hybrid_preds = np.array(hybrid_preds)

    acc = accuracy_score(y_eval, hybrid_preds)

    zero_mask = (y_eval == ZERO_DAY)
    zero_rate = (
        (hybrid_preds[zero_mask] == "ANOMALY").sum()
        / zero_mask.sum()
    )

    benign_mask = (y_eval == "BENIGN")
    benign_recall = (
        (hybrid_preds[benign_mask] == "BENIGN").sum()
        / benign_mask.sum()
    )

    print(f"\nFPR Target: {fpr_target*100:.1f}%")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Zero-Day Detection: {zero_rate:.4f}")
    print(f"Benign Recall: {benign_recall:.4f}")
