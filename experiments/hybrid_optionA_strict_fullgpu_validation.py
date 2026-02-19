# =========================================================
# OPTION A - STRICT FULL GPU WITH VALIDATION + 3 RUNS
# Deep AE + Stratified RF
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from preprocessing.preprocess_cicids import load_clean_cicids
import random

# =========================
# CONFIG
# =========================
CONF_THRESHOLD = 0.55
AE_EPOCHS = 25
RF_TREES = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = load_clean_cicids()
df.columns = df.columns.str.strip()
print("Dataset shape:", df.shape)

df.fillna(0, inplace=True)

ATTACKS = ["DDoS", "Infiltration"]

# =========================================================
# LOOP ZERO DAY
# =========================================================

for ZERO_DAY in ATTACKS:

    print("\n" + "="*80)
    print("ZERO-DAY:", ZERO_DAY)
    print("="*80)

    zero_scores = []
    benign_scores = []

    for run in range(3):

        print(f"\n========= RUN {run+1} =========")

        seed = 42 + run
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        train_full = df[df["Label"] != ZERO_DAY].copy()
        zero_test = df[df["Label"] == ZERO_DAY].copy()

        benign = train_full[train_full["Label"] == "BENIGN"]
        attacks = train_full[train_full["Label"] != "BENIGN"]

        # ---------------- SPLITS ----------------
        benign_train, benign_temp = train_test_split(
            benign, test_size=0.30, random_state=seed
        )
        benign_val, benign_test = train_test_split(
            benign_temp, test_size=0.50, random_state=seed
        )

        attack_train, attack_temp = train_test_split(
            attacks, test_size=0.30, random_state=seed
        )
        attack_val, attack_test = train_test_split(
            attack_temp, test_size=0.50, random_state=seed
        )

        # =====================================================
        # SCALER (FIT ONLY ON benign_train)
        # =====================================================
        scaler = StandardScaler()
        scaler.fit(benign_train.drop("Label", axis=1))

        X_benign_train = scaler.transform(benign_train.drop("Label", axis=1))
        X_benign_val = scaler.transform(benign_val.drop("Label", axis=1))

        # =====================================================
        # AUTOENCODER
        # =====================================================
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

        print("\nTraining AE with validation monitoring")

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

            model.eval()
            with torch.no_grad():
                recon_val = model(val_tensor)
                val_loss = criterion(recon_val, val_tensor).item()

            print(
                f"Epoch {epoch+1} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        # =====================================================
        # THRESHOLD FROM VALIDATION
        # =====================================================
        with torch.no_grad():
            recon_val = model(val_tensor)
            val_errors = torch.mean(
                (recon_val - val_tensor)**2, dim=1
            ).cpu().numpy()

        threshold = np.percentile(val_errors, 96)
        print("Validation Threshold:", threshold)

        # =====================================================
        # RF TRAINING (KNOWN ATTACKS ONLY)
        # =====================================================
        rf_train = attack_train.copy()

        X_rf_train = scaler.transform(rf_train.drop("Label", axis=1))
        y_rf_train = rf_train["Label"]

        rf = RandomForestClassifier(
            n_estimators=RF_TREES,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed
        )

        rf.fit(X_rf_train, y_rf_train)

        # =====================================================
        # TEST SET
        # =====================================================
        test_df = pd.concat([
            benign_test,
            attack_test,
            zero_test
        ])

        X_test = scaler.transform(test_df.drop("Label", axis=1))
        y_test = test_df["Label"].values

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        with torch.no_grad():
            recon_test = model(X_test_tensor)
            test_errors = torch.mean(
                (recon_test - X_test_tensor)**2, dim=1
            ).cpu().numpy()

        final_preds = []

        for i in range(len(X_test)):

            if test_errors[i] > threshold:

                probs = rf.predict_proba(X_test[i].reshape(1,-1))[0]
                max_prob = np.max(probs)
                rf_pred = rf.classes_[np.argmax(probs)]

                if rf_pred == "BENIGN" or max_prob < CONF_THRESHOLD:
                    final_preds.append("ZERO_DAY")
                else:
                    final_preds.append(rf_pred)
            else:
                final_preds.append("BENIGN")

        y_test_adjusted = y_test.copy()
        y_test_adjusted[y_test_adjusted == ZERO_DAY] = "ZERO_DAY"

        test_acc = accuracy_score(y_test_adjusted, final_preds)
        macro_f1 = f1_score(y_test_adjusted, final_preds, average="macro")

        zero_rate = (
            np.array(final_preds)[y_test == ZERO_DAY] == "ZERO_DAY"
        ).mean()

        benign_rate = (
            np.array(final_preds)[y_test == "BENIGN"] == "BENIGN"
        ).mean()

        print("\nTest Accuracy:", round(test_acc,4))
        print("Macro F1:", round(macro_f1,4))
        print("Zero-Day Recall:", round(zero_rate,4))
        print("Benign Recall:", round(benign_rate,4))

        zero_scores.append(zero_rate)
        benign_scores.append(benign_rate)

    print("\n===== FINAL STATS (3 RUNS) =====")
    print("Zero-Day Mean:", round(np.mean(zero_scores),4))
    print("Zero-Day Std:", round(np.std(zero_scores),4))
    print("Benign Mean:", round(np.mean(benign_scores),4))
    print("Benign Std:", round(np.std(benign_scores),4))
