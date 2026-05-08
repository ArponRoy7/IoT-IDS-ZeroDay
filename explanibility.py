# =========================================================
# FEATURE IMPORTANCE GENERATOR (EXPLAINABILITY)
# Heavily optimized to only train DAE + Hybrid RF to 
# extract and average Gini importances across all attacks.
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocess_cicids import load_clean_cicids
import random
import gc
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PARAMETERS & SETUP
# =========================================================
EPOCHS = 35
BATCH_SIZE = 8192  # Increased for faster DAE extraction
EXCLUDE_ATTACKS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Initializing Explainability Benchmark on {device}")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# MODEL DEFINITIONS
# =========================================================
class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# =========================================================
# PLOTTING FUNCTION
# =========================================================
def generate_feature_importance_plot(avg_importances, base_columns):
    print("\n[System] Generating IEEE-Standard Feature Importance Chart...")
    
    # Combine base features with engineered features (must match the order of np.hstack)
    feature_names = list(base_columns) + ["DAE_Residual", "Rolling_Variance"]
    
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': avg_importances
    })
    
    # Sort ascending so the largest bar is at the top of the horizontal chart
    top10_df = feat_imp_df.sort_values(by='Importance', ascending=True).tail(10)
    
    # Academic Plot Styling
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    # Color Logic: Red for Engineered, Blue for Standard
    colors = ['#d62728' if feat in ['DAE_Residual', 'Rolling_Variance'] else '#1f77b4' 
              for feat in top10_df['Feature']]
    
    bars = ax.barh(top10_df['Feature'], top10_df['Importance'], 
                   color=colors, edgecolor='black', linewidth=0.8, height=0.6)
    
    # Formatting
    ax.set_xlabel('Mean Decrease in Impurity (Gini Importance)', fontsize=13, fontweight='bold', labelpad=10)
    ax.set_title('Top 10 Features Driving Hybrid IDS Decisions', fontsize=14, fontweight='bold', pad=15)
    
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True) # Puts grid behind bars
    
    # Custom Legend
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label='Engineered Dual-Residuals (Proposed)'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='Standard Network Telemetry (Baseline)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=True, shadow=False)
    
    plt.tight_layout()
    filename = "feature_importance_explainability.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"[System] Feature Importance chart saved as: {filename}")
    plt.show()

# =========================================================
# MAIN EXECUTION BLOCK
# =========================================================
if __name__ == '__main__':
    print("\n[System] Loading and preparing dataset...")
    df = load_clean_cicids()
    for col in df.select_dtypes(include=["float64"]).columns: df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns: df[col] = df[col].astype("int32")
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    ZERO_DAY_LIST = [lbl for lbl in df["Label"].unique() if lbl != "BENIGN" and lbl not in EXCLUDE_ATTACKS]
    
    # Save base feature columns for the graph labels
    base_feature_columns = df.drop("Label", axis=1).columns

    benign_full = df[df["Label"] == "BENIGN"]
    print("[System] Fitting Global StandardScaler on Benign data...")
    scaler = StandardScaler()
    scaler.fit(benign_full.drop("Label", axis=1))
    X_benign = scaler.transform(benign_full.drop("Label", axis=1))
    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)
    
    loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True)

    # --- Train DAE ---
    print("\n[Global] Training Proposed DAE...")
    model = DAE(X_benign.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
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
            
        print(f"  -> DAE Epoch {epoch + 1}/{EPOCHS} Loss: {total_loss:.4f}")
        if epoch > 5 and abs(prev_loss - total_loss) < 1e-4: 
            print("  -> Early stopping triggered")
            break
        prev_loss = total_loss
        
    model.eval()
    
    del X_benign_tensor, loader, benign_full
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # FEATURE IMPORTANCE EXTRACTION LOOP
    # =========================================================
    all_rf_importances = []
    
    for ZERO_DAY in ZERO_DAY_LIST:
        print(f"\nExtracting RF Feature Importances (Fold: {ZERO_DAY})...")
        
        train_df = df[df["Label"] != ZERO_DAY]
        y_train_raw = train_df["Label"]
        X_train_raw = scaler.transform(train_df.drop("Label", axis=1))

        # Extract DAE Residuals
        X_train_tensor = torch.tensor(X_train_raw, dtype=torch.float32)
        residual_list = []
        with torch.no_grad():
            for i in range(0, len(X_train_tensor), BATCH_SIZE):
                batch = X_train_tensor[i:i+BATCH_SIZE].to(device)
                recon, _ = model(batch)
                residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())
                del batch, recon
        residual_train = torch.cat(residual_list).numpy()
        
        # Calculate Rolling Variance
        var_train = pd.Series(residual_train).rolling(window=15, min_periods=1).var().fillna(0).values
        
        # Augment Features (Order here MUST match the labels in the plotting function)
        X_train_aug = np.hstack([X_train_raw, residual_train.reshape(-1, 1), var_train.reshape(-1, 1)])
        
        # Train Hybrid RF and Save Importances
        print("  -> Fitting Random Forest...")
        hybrid_rf = RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        hybrid_rf.fit(X_train_aug, y_train_raw)
        
        # >> CRITICAL STEP: Grab the Gini importances <<
        all_rf_importances.append(hybrid_rf.feature_importances_)

        # Flush memory to prevent OOM across folds
        del train_df, X_train_raw, X_train_aug, X_train_tensor
        del hybrid_rf, residual_train, var_train
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[System] Extraction loops complete.")
    
    # Calculate global feature importances across all LOAO folds
    global_avg_importances = np.mean(all_rf_importances, axis=0)
    
    # Generate the final graph
    generate_feature_importance_plot(global_avg_importances, base_feature_columns)