# =========================================================
# HYBRID OPTION A - STRICT LOAO FULL GPU
# SINGLE RUN + FULL TRAIN/VAL/TEST DIAGNOSTICS
# Deep AE + RF
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from preprocessing.preprocess_cicids import load_clean_cicids
import random

# ================= CONFIG =================
CONF_THRESHOLD = 0.55
AE_EPOCHS = 25
RF_TREES = 200
ZERO_DAY = "DDoS"  # Change to Infiltration if needed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================
df = load_clean_cicids()
df.columns = df.columns.str.strip()
df.fillna(0, inplace=True)

print("Dataset shape:", df.shape)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# =========================================================
# LOAO SPLIT
# =========================================================
train_full = df[df["Label"] != ZERO_DAY].copy()
zero_test = df[df["Label"] == ZERO_DAY].copy()

benign = train_full[train_full["Label"] == "BENIGN"]
attacks = train_full[train_full["Label"] != "BENIGN"]

# BENIGN SPLIT
benign_train, benign_temp = train_test_split(
    benign, test_size=0.30, random_state=seed
)
benign_val, benign_test = train_test_split(
    benign_temp, test_size=0.50, random_state=seed
)

# ATTACK SPLIT
attack_train, attack_test = train_test_split(
    attacks, test_size=0.30, random_state=seed
)

# =========================================================
# SCALER (FIT ONLY ON benign_train)
# =========================================================
scaler = StandardScaler()
scaler.fit(benign_train.drop("Label", axis=1))

X_benign_train = scaler.transform(benign_train.drop("Label", axis=1))
X_benign_val = scaler.transform(benign_val.drop("Label", axis=1))

# =========================================================
# AUTOENCODER
# =========================================================
class AE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


model = AE(X_benign_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()

train_loader = DataLoader(
    TensorDataset(torch.tensor(X_benign_train, dtype=torch.float32)),
    batch_size=8192,
    shuffle=True
)

val_tensor = torch.tensor(X_benign_val, dtype=torch.float32).to(device)

print("\nTraining Autoencoder with Validation Monitoring\n")

for epoch in range(AE_EPOCHS):

    model.train()
    train_loss = 0

    for batch in train_loader:
        x = batch[0].to(device)
        recon = model(x)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    with torch.no_grad():
        recon_val = model(val_tensor)
        val_loss = criterion(recon_val, val_tensor).item()

    gap = train_loss - val_loss

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Gap: {gap:.6f}"
    )

# =========================================================
# THRESHOLD FROM VALIDATION
# =========================================================
with torch.no_grad():
    val_errors = torch.mean(
        (model(val_tensor) - val_tensor) ** 2,
        dim=1
    ).cpu().numpy()

threshold = np.percentile(val_errors, 96)
print("\nFinal Threshold:", threshold)

# =========================================================
# RF TRAINING (KNOWN ATTACKS ONLY)
# =========================================================
X_rf_train = scaler.transform(attack_train.drop("Label", axis=1))
y_rf_train = attack_train["Label"]

rf = RandomForestClassifier(
    n_estimators=RF_TREES,
    class_weight="balanced",
    n_jobs=-1,
    random_state=seed
)

rf.fit(X_rf_train, y_rf_train)

# ================= TRAIN PERFORMANCE =================
train_preds = rf.predict(X_rf_train)

print("\n=== RF TRAIN PERFORMANCE ===")
print("Train Accuracy:",
      accuracy_score(y_rf_train, train_preds))

print("Train Macro F1:",
      f1_score(y_rf_train, train_preds,
               average="macro", zero_division=0))

print("Train Confusion Matrix:\n",
      confusion_matrix(y_rf_train, train_preds))

# =========================================================
# TEST SET
# =========================================================
test_df = pd.concat([
    benign_test,
    attack_test,
    zero_test
])

X_test = scaler.transform(test_df.drop("Label", axis=1))
y_test = test_df["Label"].values

rf_preds = rf.predict(X_test)

# STRICT ZERO DAY RELABELING
y_test_adjusted = y_test.copy()
y_test_adjusted[y_test_adjusted == ZERO_DAY] = "ZERO_DAY"

# =========================================================
# TEST METRICS
# =========================================================
print("\n=== TEST PERFORMANCE ===")

print("Test Accuracy:",
      accuracy_score(y_test_adjusted, rf_preds))

print("Test Macro F1:",
      f1_score(y_test_adjusted, rf_preds,
               average="macro", zero_division=0))

print("Test Precision:",
      precision_score(y_test_adjusted, rf_preds,
                      average="macro", zero_division=0))

print("Test Recall:",
      recall_score(y_test_adjusted, rf_preds,
                   average="macro", zero_division=0))

print("\nTest Confusion Matrix:\n",
      confusion_matrix(y_test_adjusted, rf_preds))
