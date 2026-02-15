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
# Load dataset once
# =========================================================
df = load_clean_cicids()
df.columns = df.columns.str.strip()

# =========================================================
# Attacks to evaluate
# =========================================================
ATTACK_LIST = ["DDoS", "Infiltration", "PortScan"]

# =========================================================
# Loop over attacks (LOAO)
# =========================================================
for ZERO_DAY in ATTACK_LIST:

    print("\n" + "="*60)
    print("Zero-Day Attack:", ZERO_DAY)
    print("="*60)

    # -----------------------------------------------------
    # Train data (exclude zero-day)
    # -----------------------------------------------------
    train_df = df[df["Label"] != ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    print("Total benign samples:", len(benign_df))

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(train_df.drop("Label", axis=1))

    X_benign_scaled = scaler.transform(
        benign_df.drop("Label", axis=1)
    )

    # -----------------------------------------------------
    # Autoencoder (same architecture as before)
    # -----------------------------------------------------
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
                nn.Dropout(0.2),
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
            return self.decoder(self.encoder(x))

    model = Autoencoder(X_tensor.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print("Training Autoencoder...")
    for epoch in range(12):
        total_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    print("AE Training complete.")

    # -----------------------------------------------------
    # Benign reconstruction errors
    # -----------------------------------------------------
    model.eval()
    with torch.no_grad():
        X_benign_tensor = torch.tensor(
            X_benign_scaled,
            dtype=torch.float32
        ).to(device)

        recon = model(X_benign_tensor)
        benign_errors = torch.mean(
            (recon - X_benign_tensor) ** 2,
            dim=1
        ).cpu().numpy()

    # -----------------------------------------------------
    # Train Balanced RF
    # -----------------------------------------------------
    print("Training Balanced Random Forest...")

    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]

    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    rf.fit(X_rf, y_rf)
    print("RF Training complete.")

    # -----------------------------------------------------
    # Evaluation Dataset
    # -----------------------------------------------------
    eval_df = pd.concat([
        train_df.sample(n=80000, random_state=42),
        df[df["Label"] == ZERO_DAY].sample(
            n=min(5000, len(df[df["Label"] == ZERO_DAY])),
            random_state=42
        )
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
            (recon - X_eval_tensor) ** 2,
            dim=1
        ).cpu().numpy()

    rf_preds = rf.predict(X_eval)

    # -----------------------------------------------------
    # Cascaded Hybrid Evaluation
    # -----------------------------------------------------
    print("\nTWO-THRESHOLD CASCADED HYBRID")

    for fpr_target in [0.01, 0.02]:

        T_high = np.percentile(benign_errors, 100 - (fpr_target * 100))
        T_low = np.percentile(benign_errors, (fpr_target * 100))

        hybrid_preds = []
        rf_used = 0

        for i in range(len(eval_errors)):

            if eval_errors[i] > T_high:
                hybrid_preds.append("ANOMALY")

            elif eval_errors[i] < T_low:
                hybrid_preds.append("BENIGN")

            else:
                hybrid_preds.append(rf_preds[i])
                rf_used += 1

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
        print(f"RF Usage Rate: {rf_used/len(eval_errors):.4f}")
