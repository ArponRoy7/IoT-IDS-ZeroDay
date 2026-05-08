# =========================================================
# PRECISION-RECALL (PR) CURVE GENERATOR
# Evaluates Hybrid vs. Std RF vs. KitNET vs. MLP vs. Isolation Forest
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_recall_curve, auc
from preprocessing.preprocess_cicids import load_clean_cicids
import matplotlib.pyplot as plt
import gc
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PARAMETERS & SETUP
# =========================================================
EPOCHS = 35
MLP_EPOCHS = 5
BATCH_SIZE = 8192
EXCLUDE_ATTACKS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Initializing PR Curve Generator on {device}")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# =========================================================
# MODEL DEFINITIONS
# =========================================================
class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x): return self.decoder(self.encoder(x)[0]), self.encoder(x)

class DeepMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.network(x)

class SmallAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, dim))
    def forward(self, x): return self.decoder(self.encoder(x))

class CombinerAE(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(k, 8), nn.ReLU(), nn.Linear(8, 2))
        self.decoder = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, k))
    def forward(self, x): return self.decoder(self.encoder(x))

# =========================================================
# PLOTTING FUNCTION
# =========================================================
def plot_pr_curve(y_true, hybrid_probs, rf_probs, kitnet_scores, mlp_probs, if_scores):
    print("\n[System] Generating IEEE-Standard PR Curve...")
    
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 5 Models mapped to specific colors and line styles
    models = {
        'Proposed Hybrid IDS': (hybrid_probs, '#1f77b4', '-'),       # Solid Blue
        'Deep MLP Baseline': (mlp_probs, '#2ca02c', '-.'),           # Dot-Dash Green
        'KitNET (Unsupervised)': (kitnet_scores, '#ff7f0e', '--'),   # Dashed Orange
        'Isolation Forest Baseline': (if_scores, '#9467bd', '--'),   # Dashed Purple
        'Std RF (Supervised)': (rf_probs, '#555555', ':')            # Dotted Gray
    }

    for name, (scores, color, linestyle) in models.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall, precision)
        
        # Plot line
        ax.plot(recall, precision, color=color, linestyle=linestyle, linewidth=2.5 if name == 'Proposed Hybrid IDS' else 2, 
                label=f'{name} (AUC = {pr_auc:.3f})')
        
        # Add subtle fill for the proposed model to make it stand out
        if name == 'Proposed Hybrid IDS':
            ax.fill_between(recall, precision, alpha=0.1, color=color)

    ax.set_xlabel('Recall (True Positive Rate)', fontweight='bold')
    ax.set_ylabel('Precision (Positive Predictive Value)', fontweight='bold')
    ax.set_title('Precision-Recall Curve on Zero-Day Attacks', fontweight='bold', pad=15)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.legend(loc='lower left', frameon=True, fancybox=True, shadow=False)
    
    plt.tight_layout()
    filename = "pr_curve_comparison.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"[System] PR Curve saved successfully as: {filename}")
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
    benign_full = df[df["Label"] == "BENIGN"]
    
    print("[System] Fitting Global StandardScaler...")
    scaler = StandardScaler()
    X_benign = scaler.fit_transform(benign_full.drop("Label", axis=1))
    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True)

    # 1. Train DAE
    print("\n[Global] Training Proposed DAE...")
    hybrid_dae = DAE(X_benign.shape[1]).to(device)
    opt_dae = torch.optim.AdamW(hybrid_dae.parameters(), lr=1e-3)
    hybrid_dae.train()
    for epoch in range(EPOCHS):
        for (x,) in loader:
            x = x.to(device)
            opt_dae.zero_grad()
            recon, _ = hybrid_dae(x + (torch.randn_like(x) * 0.05))
            loss = nn.MSELoss()(recon, x)
            loss.backward()
            opt_dae.step()
    hybrid_dae.eval()

    # 2. Train KitNET
    print("\n[Global] Training KitNET Ensembles...")
    NUM_GROUPS = 5
    num_features = X_benign.shape[1]
    indices = np.arange(num_features)
    np.random.shuffle(indices)
    split_size = num_features // NUM_GROUPS
    FEATURE_SPLITS = [indices[i*split_size:(i+1)*split_size] if i < NUM_GROUPS-1 else indices[i*split_size:] for i in range(NUM_GROUPS)]

    ensemble = [SmallAE(len(split)).to(device) for split in FEATURE_SPLITS]
    opt_ens = [torch.optim.Adam(ae.parameters(), lr=1e-3) for ae in ensemble]
    combiner = CombinerAE(NUM_GROUPS).to(device)
    opt_comb = torch.optim.Adam(combiner.parameters(), lr=1e-3)

    for idx, ae in enumerate(ensemble):
        for epoch in range(3):
            for (x,) in loader:
                x_sub = x[:, FEATURE_SPLITS[idx]].to(device)
                opt_ens[idx].zero_grad()
                loss = nn.MSELoss()(ae(x_sub), x_sub)
                loss.backward()
                opt_ens[idx].step()

    for epoch in range(3):
        for (x,) in loader:
            x = x.to(device)
            errs = torch.stack([torch.mean((ae(x[:, split]) - x[:, split])**2, dim=1) for ae, split in zip(ensemble, FEATURE_SPLITS)], dim=1)
            opt_comb.zero_grad()
            loss = nn.MSELoss()(combiner(errs), errs)
            loss.backward()
            opt_comb.step()

    def kitnet_score(X_tensor):
        scores = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), BATCH_SIZE):
                batch = X_tensor[i:i+BATCH_SIZE].to(device)
                errs = torch.stack([torch.mean((ae(batch[:, split]) - batch[:, split])**2, dim=1) for ae, split in zip(ensemble, FEATURE_SPLITS)], dim=1)
                scores.append(torch.mean((combiner(errs) - errs)**2, dim=1).cpu())
        return torch.cat(scores).numpy()

    del X_benign_tensor, loader
    gc.collect()

    # Data Collectors for PR Curve
    y_true_all = []
    hybrid_probs_all = []
    rf_probs_all = []
    kitnet_scores_all = []
    mlp_probs_all = []
    if_scores_all = [] # Added for Isolation Forest

    # =========================================================
    # SINGLE AGGREGATED FOLD FOR PR CURVE
    # We will pick the top 3 hardest attacks to create a challenging test set
    # =========================================================
    TARGET_ATTACKS = ["Heartbleed", "DDoS", "DoS slowloris"]
    print(f"\n[System] Extracting probabilities on challenging targets: {TARGET_ATTACKS}")
    
    train_df = df[~df["Label"].isin(TARGET_ATTACKS)]
    test_df = df[df["Label"].isin(TARGET_ATTACKS + ["BENIGN"])].sample(frac=0.1, random_state=seed) # Subsample for speed
    
    y_train_raw = train_df["Label"]
    X_train_raw = scaler.transform(train_df.drop("Label", axis=1))
    
    y_test_raw = test_df["Label"]
    X_test_raw = scaler.transform(test_df.drop("Label", axis=1))
    
    y_test_binary = np.where(y_test_raw == "BENIGN", 0, 1)
    y_true_all.extend(y_test_binary)

    # 1. Std RF Baseline
    print("  -> Training Std RF...")
    std_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    std_rf.fit(X_train_raw, y_train_raw)
    
    benign_idx = list(std_rf.classes_).index("BENIGN")
    rf_probs = 1.0 - std_rf.predict_proba(X_test_raw)[:, benign_idx]
    rf_probs_all.extend(rf_probs)

    # 2. Isolation Forest Baseline (NEW)
    print("  -> Training Isolation Forest...")
    iso_f = IsolationForest(n_estimators=100, n_jobs=-1, random_state=seed)
    iso_f.fit(X_train_raw)
    
    # Invert scores so higher = more anomalous (needed for PR Curve math)
    if_scores = -iso_f.decision_function(X_test_raw)
    if_scores_all.extend(if_scores)

    # 3. Deep MLP Baseline
    print("  -> Training Deep MLP...")
    y_train_mlp = np.where(y_train_raw == "BENIGN", 0, 1).astype(np.float32)
    mlp_loader = DataLoader(TensorDataset(torch.tensor(X_train_raw, dtype=torch.float32), torch.tensor(y_train_mlp).unsqueeze(1)), batch_size=BATCH_SIZE, shuffle=True)
    mlp_model = DeepMLP(X_train_raw.shape[1]).to(device)
    mlp_optim = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
    mlp_model.train()
    for e in range(MLP_EPOCHS):
        for bx, by in mlp_loader:
            bx, by = bx.to(device), by.to(device)
            mlp_optim.zero_grad()
            loss = nn.BCELoss()(mlp_model(bx), by)
            loss.backward()
            mlp_optim.step()
    mlp_model.eval()
    
    with torch.no_grad():
        mlp_probs = torch.cat([mlp_model(torch.tensor(X_test_raw[i:i+BATCH_SIZE], dtype=torch.float32).to(device)).cpu() for i in range(0, len(X_test_raw), BATCH_SIZE)]).numpy().flatten()
    mlp_probs_all.extend(mlp_probs)

    # 4. KitNET
    print("  -> Extracting KitNET Scores...")
    kitnet_scores = kitnet_score(torch.tensor(X_test_raw, dtype=torch.float32))
    kitnet_scores_all.extend(kitnet_scores)

    # 5. Proposed Hybrid
    print("  -> Training Proposed Hybrid...")
    X_train_tensor = torch.tensor(X_train_raw, dtype=torch.float32)
    res_list = []
    with torch.no_grad():
        for i in range(0, len(X_train_tensor), BATCH_SIZE):
            res_list.append(torch.mean((hybrid_dae(X_train_tensor[i:i+BATCH_SIZE].to(device))[0] - X_train_tensor[i:i+BATCH_SIZE].to(device)) ** 2, dim=1).cpu())
    res_train = torch.cat(res_list).numpy()
    var_train = pd.Series(res_train).rolling(window=15, min_periods=1).var().fillna(0).values
    X_train_aug = np.hstack([X_train_raw, res_train.reshape(-1, 1), var_train.reshape(-1, 1)])
    
    hybrid_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)
    hybrid_rf.fit(X_train_aug, y_train_raw)

    print("  -> Extracting Hybrid Probabilities...")
    X_test_tensor = torch.tensor(X_test_raw, dtype=torch.float32)
    res_test_list = []
    with torch.no_grad():
        for i in range(0, len(X_test_tensor), BATCH_SIZE):
            res_test_list.append(torch.mean((hybrid_dae(X_test_tensor[i:i+BATCH_SIZE].to(device))[0] - X_test_tensor[i:i+BATCH_SIZE].to(device)) ** 2, dim=1).cpu())
    res_test = torch.cat(res_test_list).numpy()
    var_test = pd.Series(res_test).rolling(window=15, min_periods=1).var().fillna(0).values
    X_test_aug = np.hstack([X_test_raw, res_test.reshape(-1, 1), var_test.reshape(-1, 1)])
    
    benign_idx_h = list(hybrid_rf.classes_).index("BENIGN")
    hybrid_probs = 1.0 - hybrid_rf.predict_proba(X_test_aug)[:, benign_idx_h]
    
    res_norm = (res_test - res_test.min()) / (res_test.max() - res_test.min() + 1e-8)
    hybrid_final_scores = (hybrid_probs * 0.7) + (res_norm * 0.3)
    hybrid_probs_all.extend(hybrid_final_scores)

    # Generate the Graph (Added if_scores_all)
    plot_pr_curve(y_true_all, hybrid_probs_all, rf_probs_all, kitnet_scores_all, mlp_probs_all, if_scores_all)