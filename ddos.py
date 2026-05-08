import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from preprocessing.preprocess_cicids import load_clean_cicids
from collections import deque
import random
import matplotlib.pyplot as plt

# =========================================================
# PARAMETERS
# =========================================================
WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 90
ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85
EPOCHS = 35
BATCH_SIZE = 4096

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# LOAD AND PREPROCESS DATA
# =========================================================
df = load_clean_cicids()

# Optimize data types
for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")
for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# FILTER ONLY DDOS ATTACKS
ZERO_DAY_LIST = [label for label in df["Label"].unique() if "DDoS" in label]
print(f"Targeting Zero-Day Attacks: {ZERO_DAY_LIST}")

# =========================================================
# TRAIN DAE (ONCE ON BENIGN DATA)
# =========================================================
benign_full = df[df["Label"] == "BENIGN"]
scaler = StandardScaler()
scaler.fit(benign_full.drop("Label", axis=1))
X_benign = scaler.transform(benign_full.drop("Label", axis=1))
X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model = DAE(X_benign.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True)

print("Training DAE...")
model.train()
prev_loss = float("inf")
for epoch in range(EPOCHS):
    total_loss = 0
    for (x,) in loader:
        x = x.to(device)
        noise = torch.randn_like(x) * 0.05
        optimizer.zero_grad()
        recon, _ = model(x + noise)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch > 5 and abs(prev_loss - total_loss) < 1e-4: break
    prev_loss = total_loss
model.eval()

# =========================================================
# DDoS EVALUATION & PLOTTING LOOP
# =========================================================
for ZERO_DAY in ZERO_DAY_LIST:
    print(f"\nEvaluating: {ZERO_DAY}")
    train_df = df[df["Label"] != ZERO_DAY]
    zero_df = df[df["Label"] == ZERO_DAY]
    benign_df = train_df[train_df["Label"] == "BENIGN"]

    # Sliding Window Init
    residual_memory = deque(maxlen=WINDOW_SIZE)
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            recon, _ = model(x)
            res = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
            residual_memory.extend(res)

    # Train RF
    X_rf = scaler.transform(train_df.drop("Label", axis=1))
    y_rf = train_df["Label"]
    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)
    
    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_rf_tensor), BATCH_SIZE):
            batch = X_rf_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())
    
    residual_rf = torch.cat(residual_list).numpy()
    variance_rf = pd.Series(residual_rf).rolling(window=15, min_periods=1).var().fillna(0).values
    X_rf_aug = np.hstack([X_rf, residual_rf.reshape(-1, 1), variance_rf.reshape(-1, 1)])
    
    rf = RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    rf.fit(X_rf_aug, y_rf)

    # Evaluation
    eval_df = pd.concat([benign_df.sample(min(300000, len(benign_df)), random_state=seed), zero_df])
    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values
    X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    residual_list = []
    with torch.no_grad():
        for i in range(0, len(X_eval_tensor), BATCH_SIZE):
            batch = X_eval_tensor[i:i+BATCH_SIZE].to(device)
            recon, _ = model(batch)
            residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())
    
    residual_eval = torch.cat(residual_list).numpy()
    variance_eval = pd.Series(residual_eval).rolling(window=15, min_periods=1).var().fillna(0).values
    threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)

    # --- PROFESSIONAL PLOT GENERATION ---
    attack_indices = np.where(y_eval == ZERO_DAY)[0]
    if len(attack_indices) > 0:
        start_idx = attack_indices[0]
        
        # ZOOM FIX: 100 flows of normal traffic, 400 flows of attack traffic
        plot_start = max(0, start_idx - 100)
        plot_end = min(len(residual_eval), start_idx + 400)
        
        plt.rcParams.update({"font.family": "serif", "font.size": 12})
        plt.figure(figsize=(10, 5), dpi=300)
        
        x_range = np.arange(plot_end - plot_start)
        
        # Plotting the lines
        plt.plot(x_range, residual_eval[plot_start:plot_end], label=r'Reconstruction Residual ($r_i$)', 
                 color='#1f77b4', linewidth=1.2, alpha=0.85)
        plt.plot(x_range, variance_eval[plot_start:plot_end], label=r'15-Step Rolling Variance ($V_r$)', 
                 color='#ff7f0e', linewidth=1.6)
        
        # Adaptive Threshold
        plt.axhline(y=threshold, color='#2c3e50', linestyle='--', linewidth=2, label=r'Adaptive Threshold ($\tau_{90}$)')
        
        # OVERLAP FIX 1: Add 25% extra headroom to the Y-axis so the legend doesn't crush the text
        plt.ylim(0, plt.gca().get_ylim()[1] * 1.25)
        
        # Attack Start Line (Will appear exactly at x=100 on the graph)
        rel_idx = start_idx - plot_start
        plt.axvline(x=rel_idx, color='#d62728', linestyle=':', linewidth=2.5)
        
        # OVERLAP FIX 2: Moved text down to 70% of the graph's height (was 85%)
        plt.text(rel_idx + 5, plt.gca().get_ylim()[1] * 0.70, 'DDoS Attack Start', color='#d62728', fontweight='bold', fontsize=13,
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=3))

        # Formal Formatting
        plt.xlabel('Sequential Flow Index')
        plt.ylabel(r'Residual ($r_i$) / Variance ($V_r$)')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc='upper left', frameon=True, shadow=False, fancybox=True)
        
        plt.tight_layout()
        
        # Save as individual files for each DDoS variant
        filename = f"ddos_plot_{ZERO_DAY.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Graph saved as {filename}")
        plt.show()

print("\nDDoS Specific Evaluation Completed.")