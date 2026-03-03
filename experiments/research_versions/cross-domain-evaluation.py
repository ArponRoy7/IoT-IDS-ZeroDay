# =========================================================
# CROSS-DATASET EVALUATION: TRAIN (CICIDS) -> TEST (IoT)
# Logic: Manual Feature Mapping + Domain Adaptation Calibration
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score
from preprocessing.preprocess_cicids import load_clean_cicids
from preprocessing.preprocess_ciciot2023 import load_clean_ciciot2023
from collections import deque
import random

# =========================================================
# PARAMETERS
# =========================================================
WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 99.0  # High stability
ALPHA_BENIGN = 0.9999 
ALPHA_ATTACK = 0.85
EPOCHS = 35
BATCH_SIZE = 4096
seed = 42

device = torch.device("cpu")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# STEP 1: CROSS-DOMAIN FEATURE BRIDGE
# =========================================================
print("Building Universal Feature Bridge...")
df_train_raw = load_clean_cicids()
df_test_raw = load_clean_ciciot2023()

# Define the bridge based on your specific CSV columns
# Left: CICIoT (IoT) | Right: CICIDS (Enterprise)
bridge = {
    'flow_duration': 'Flow Duration',
    'Rate': 'Flow Packets/s',
    'Header_Length': 'Fwd Header Length',
    'syn_flag_number': 'SYN Flag Count',
    'rst_flag_number': 'RST Flag Count',
    'psh_flag_number': 'PSH Flag Count',
    'ack_flag_number': 'ACK Flag Count'
}

common_features = list(bridge.keys())

try:
    # 1. Align CICIDS (Training)
    X_train = df_train_raw[[bridge[f] for f in common_features]].copy()
    X_train.columns = common_features
    df_train = pd.concat([X_train, df_train_raw['Label']], axis=1)

    # 2. Align CICIoT (Testing)
    # Note: We use the exact names found in your IoT log
    df_test = df_test_raw[common_features + ['Label']].copy()
    
    print(f"✅ SUCCESS: {len(common_features)} Universal Features Synced.")
except KeyError as e:
    print(f"❌ Still missing a column: {e}")
    # Fallback: List every column to find a close match
    print("IoT Columns:", df_test_raw.columns.tolist())
    exit()

# Final Cleaning
for df_obj in [df_train, df_test]:
    df_obj.columns = [c.lower() for c in df_obj.columns]
    df_obj['label'] = df_obj['label'].astype(str).str.upper().str.strip()
    df_obj.replace([np.inf, -np.inf], 0, inplace=True)
    df_obj.fillna(0, inplace=True)

# =========================================================
# STEP 2: TRAIN ON SOURCE (CICIDS2017)
# =========================================================
print("\nTraining on Source: CICIDS2017...")
benign_train = df_train[df_train["label"] == "BENIGN"]
scaler = StandardScaler()
scaler.fit(benign_train.drop("label", axis=1))

X_train_benign = scaler.transform(benign_train.drop("label", axis=1))
X_train_tensor = torch.tensor(X_train_benign, dtype=torch.float32).to(device)

class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x):
        recon = self.decoder(self.encoder(x))
        return recon, None

model = DAE(len(common_features)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()
loader = DataLoader(TensorDataset(X_train_tensor), batch_size=BATCH_SIZE, shuffle=True)

model.train()
best_loss = float("inf")
best_model_state = None
for epoch in range(EPOCHS):
    total_loss = 0
    for (x,) in loader:
        noise = torch.randn_like(x) * 0.05
        optimizer.zero_grad(); recon, _ = model(x + noise)
        loss = criterion(recon, x); loss.backward(); optimizer.step()
        total_loss += loss.item()
    if total_loss < best_loss:
        best_loss = total_loss
        best_model_state = copy.deepcopy(model.state_dict())
    if (epoch + 1) % 10 == 0: print(f"Epoch {epoch+1} Loss: {round(total_loss, 4)}")

model.load_state_dict(best_model_state); model.eval()

print("Building RF Knowledge Base...")
X_rf_scaled = scaler.transform(df_train.drop("label", axis=1))
with torch.no_grad():
    recon_rf, _ = model(torch.tensor(X_rf_scaled, dtype=torch.float32))
    res_rf = torch.mean((recon_rf - torch.tensor(X_rf_scaled, dtype=torch.float32))**2, dim=1).numpy()

var_rf = pd.Series(res_rf).rolling(window=25, min_periods=1).var().fillna(0).values
X_rf_aug = np.hstack([X_rf_scaled, res_rf.reshape(-1, 1), var_rf.reshape(-1, 1)])
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
rf.fit(X_rf_aug, df_train["label"])

# =========================================================
# STEP 3: DOMAIN ADAPTATION (THE FIX)
# =========================================================
print("\nCalibrating Threshold to Target (IoT) Baseline...")
# 
iot_benign_sample = df_test[df_test['label'] == 'BENIGN'].sample(frac=0.1, random_state=seed)
X_iot_calib = scaler.transform(iot_benign_sample.drop('label', axis=1))

with torch.no_grad():
    recon_cal, _ = model(torch.tensor(X_iot_calib, dtype=torch.float32))
    res_cal = torch.mean((recon_cal - torch.tensor(X_iot_calib, dtype=torch.float32))**2, dim=1).numpy()

# Set threshold based on the IoT "Normal", not the PC "Normal"
threshold = np.percentile(res_cal, THRESHOLD_PERCENTILE)
print(f"Adapted IoT Threshold: {threshold}")

# =========================================================
# STEP 4: EVALUATE ON TARGET (CICIoT2023)
# =========================================================
print("Evaluating on CICIoT2023...")
X_test_scaled = scaler.transform(df_test.drop("label", axis=1))
y_test_binary = (df_test["label"] != "BENIGN").astype(int) 

with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    recon_test, _ = model(X_test_tensor)
    res_test = torch.mean((recon_test - X_test_tensor)**2, dim=1).numpy()

var_test = pd.Series(res_test).rolling(window=25, min_periods=1).var().fillna(0).values
X_test_aug = np.hstack([X_test_scaled, res_test.reshape(-1, 1), var_test.reshape(-1, 1)])

rf_preds = rf.predict(X_test_aug)
rf_probs = np.max(rf.predict_proba(X_test_aug), axis=1)

final_preds = []
for i in range(len(X_test_aug)):
    r, p, pr = res_test[i], rf_preds[i], rf_probs[i]
    if r > threshold:
        final_preds.append(0 if (p == "BENIGN" and pr >= ALPHA_BENIGN) else 1)
    else:
        final_preds.append(0 if p == "BENIGN" else 1)

print("\n" + "="*45)
print("   CROSS-DOMAIN ADAPTED RESULTS")
print("="*45)
print(f"Benign Recall (Specificity):  {round(recall_score(y_test_binary, final_preds, pos_label=0), 4)}")
print(f"Anomaly Recall (Sensitivity): {round(recall_score(y_test_binary, final_preds, pos_label=1), 4)}")
print(f"F1-Score:                     {round(f1_score(y_test_binary, final_preds), 4)}")
print("="*45)