# =========================================================
# MASTER EVALUATION SUITE: IEEE ZERO-DAY IDS
# Evaluates Table II Attacks ONLY.
# Generates Text Table and ONE Graph (Radar Chart).
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import recall_score
from preprocessing.preprocess_cicids import load_clean_cicids
from collections import deque
import random
import time
import gc
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# PARAMETERS & SETUP
# =========================================================
WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 90
ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85

EPOCHS = 35
MLP_EPOCHS = 5      
BATCH_SIZE = 8192   

# STRICT FILTER FOR TABLE II ATTACKS ONLY
TABLE_II_ATTACKS = [
    "DDoS", "DoS slowloris", "DoS Slowhttptest", "DoS GoldenEye",
    "Heartbleed", "Infiltration",
    "PortScan", "Bot"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# OUTPUT GENERATOR FUNCTIONS
# =========================================================
def generate_text_table(results_list):
    print("\n" + "="*98)
    print("FINAL ZERO-DAY RECALL COMPARISON VS BASELINES")
    print("="*98)
    
    categories = {
        "Volumetric DoS": ["DDoS", "DoS slowloris", "DoS Slowhttptest", "DoS GoldenEye"],
        "Complex Exploitation": ["Heartbleed", "Infiltration"],
        "Reconnaissance": ["PortScan", "Bot"]
    }

    avg_b_hybrid = np.mean([r['b_Hybrid'] for r in results_list]) * 100
    avg_b_kitnet = np.mean([r['b_KitNET'] for r in results_list]) * 100
    avg_b_mlp = np.mean([r['b_MLP'] for r in results_list]) * 100
    avg_b_if = np.mean([r['b_IF'] for r in results_list]) * 100
    avg_b_rf = np.mean([r['b_RF'] for r in results_list]) * 100

    print(f"{'Isolated Zero-Day':<30} | {'Hybrid':<10} | {'KitNET':<10} | {'MLP':<10} | {'IF':<10} | {'Std RF':<10}")
    print("-" * 98)

    for cat_name, attacks in categories.items():
        valid_attacks = [res for res in results_list if res["attack"] in attacks or res["attack"].replace('\uFFFD', '-') in attacks]
        if not valid_attacks: continue
        
        print(f"[{cat_name}]")
        for res in valid_attacks:
            atk_display = res["attack"].replace('\uFFFD', '-')
            print(f"{atk_display:<30} | {res['Hybrid']*100:>8.2f}% | {res['KitNET']*100:>8.2f}% | {res['MLP']*100:>8.2f}% | {res['IF']*100:>8.2f}% | {res['RF']*100:>8.2f}%")
        print("-" * 98)

    print(f"{'Benign Average':<30} | {avg_b_hybrid:>8.2f}% | {avg_b_kitnet:>8.2f}% | {avg_b_mlp:>8.2f}% | {avg_b_if:>8.2f}% | {avg_b_rf:>8.2f}%")
    print("=" * 98)

def generate_radar_chart(results_list):
    print("\n[System] Generating IEEE-Standard Radar Chart...")
    categories_map = {
        "Volumetric DoS": ["DDoS", "DoS slowloris", "DoS Slowhttptest", "DoS GoldenEye"],
        "Complex Exploitation": ["Heartbleed", "Infiltration"],
        "Reconnaissance": ["PortScan", "Bot"]
    }
    cat_names = ["Volumetric DoS", "Complex Exploitation", "Reconnaissance", "Benign Recall"]
    
    hybrid_scores, kitnet_scores, rf_scores = [], [], []
    for cat, attacks in categories_map.items():
        valid_res = [r for r in results_list if r["attack"] in attacks or r["attack"].replace('\uFFFD', '-') in attacks]
        if valid_res:
            hybrid_scores.append(np.mean([r['Hybrid'] for r in valid_res]))
            kitnet_scores.append(np.mean([r['KitNET'] for r in valid_res]))
            rf_scores.append(np.mean([r['RF'] for r in valid_res]))
        else:
            hybrid_scores.append(0); kitnet_scores.append(0); rf_scores.append(0)
            
    hybrid_scores.append(np.mean([r['b_Hybrid'] for r in results_list]))
    kitnet_scores.append(np.mean([r['b_KitNET'] for r in results_list]))
    rf_scores.append(np.mean([r['b_RF'] for r in results_list]))

    angles = np.linspace(0, 2 * np.pi, len(cat_names), endpoint=False).tolist()
    hybrid_scores += hybrid_scores[:1]; kitnet_scores += kitnet_scores[:1]; rf_scores += rf_scores[:1]; angles += angles[:1]
    
    plt.rcParams.update({"font.family": "serif", "font.size": 12})
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300, subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], cat_names, color='black', size=13, fontweight='bold')
    
    # Text Alignment Fix to prevent overlapping borders
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle == 0: 
            label.set_horizontalalignment('center')
            label.set_verticalalignment('bottom')
        elif angle == np.pi / 2: 
            label.set_horizontalalignment('left')
            label.set_verticalalignment('center')
        elif angle == np.pi: 
            label.set_horizontalalignment('center')
            label.set_verticalalignment('top')
        else: 
            label.set_horizontalalignment('right')
            label.set_verticalalignment('center')
            
    ax.tick_params(axis='x', pad=25) 
    
    # 45-DEGREE LABEL FIX: Puts the 0.2, 0.4 numbers neatly in empty space
    ax.set_rlabel_position(45) 
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="#7f8c8d", size=9.5, fontstyle='italic', fontweight='bold')
    plt.ylim(0, 1.05)
    
    ax.plot(angles, hybrid_scores, color='#1f77b4', linewidth=2.5, linestyle='solid', label='Proposed Hybrid IDS')
    ax.fill(angles, hybrid_scores, color='#1f77b4', alpha=0.15)
    ax.plot(angles, kitnet_scores, color='#ff7f0e', linewidth=2, linestyle='--', label='KitNET (Unsupervised)')
    ax.plot(angles, rf_scores, color='#555555', linewidth=2, linestyle=':', label='Std RF (Supervised)')

    plt.grid(color='#E8E8E8', linestyle='-', linewidth=1)
    ax.spines['polar'].set_color('#CCCCCC')
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), frameon=True, fancybox=True, shadow=False)
    plt.tight_layout()
    plt.savefig("radar_chart_comparison.png", bbox_inches='tight')
    plt.close()

# =========================================================
# MAIN EXECUTION BLOCK
# =========================================================
if __name__ == '__main__':
    print(f"🚀 Initializing High-Speed SOTA Benchmark on {device}")
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("\n[System] Loading and preparing dataset...")
    df = load_clean_cicids()
    for col in df.select_dtypes(include=["float64"]).columns: df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns: df[col] = df[col].astype("int32")
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    available_attacks = df["Label"].unique()
    ZERO_DAY_LIST = [lbl for lbl in available_attacks if lbl in TABLE_II_ATTACKS or lbl.replace('\uFFFD', '-') in TABLE_II_ATTACKS]
    print(f"[System] Found {len(ZERO_DAY_LIST)} exact Table II attacks to evaluate.")

    benign_full = df[df["Label"] == "BENIGN"]
    print("[System] Fitting Global StandardScaler on Benign data...")
    scaler = StandardScaler()
    X_benign = scaler.fit_transform(benign_full.drop("Label", axis=1))
    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_benign_tensor), batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    print("\n[Global] Training Proposed DAE...")
    hybrid_dae = DAE(X_benign.shape[1]).to(device)
    opt_dae = torch.optim.AdamW(hybrid_dae.parameters(), lr=1e-3)
    hybrid_dae.train()
    prev_loss = float("inf")
    for epoch in range(EPOCHS):
        total_loss = 0
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
            opt_dae.zero_grad()
            recon, _ = hybrid_dae(x + (torch.randn_like(x) * 0.05))
            loss = nn.MSELoss()(recon, x)
            loss.backward()
            opt_dae.step()
            total_loss += loss.item()
        if epoch > 5 and abs(prev_loss - total_loss) < 1e-4: break
        prev_loss = total_loss
    hybrid_dae.eval()

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
        for epoch in range(5):
            for (x,) in loader:
                x_sub = x[:, FEATURE_SPLITS[idx]].to(device, non_blocking=True)
                opt_ens[idx].zero_grad()
                loss = nn.MSELoss()(ae(x_sub), x_sub)
                loss.backward()
                opt_ens[idx].step()

    for epoch in range(5):
        for (x,) in loader:
            x = x.to(device, non_blocking=True)
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

    KITNET_THRESHOLD = np.percentile(kitnet_score(X_benign_tensor[:50000]), THRESHOLD_PERCENTILE)
    del X_benign_tensor, loader
    gc.collect(); torch.cuda.empty_cache()

    all_results = []
    
    for ZERO_DAY in ZERO_DAY_LIST:
        print("\n" + "="*80)
        print(f"Evaluating Isolated Zero-Day: {ZERO_DAY}")
        
        train_df = df[df["Label"] != ZERO_DAY]
        zero_df = df[df["Label"] == ZERO_DAY]
        benign_df = train_df[train_df["Label"] == "BENIGN"]

        X_train_raw = scaler.transform(train_df.drop("Label", axis=1))
        y_train_raw = train_df["Label"]
        
        std_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        std_rf.fit(X_train_raw, y_train_raw)

        iso_f = IsolationForest(n_estimators=100, contamination=0.01, n_jobs=-1, random_state=seed)
        iso_f.fit(X_train_raw)

        y_train_mlp = np.where(y_train_raw == "BENIGN", 0, 1).astype(np.float32)
        mlp_loader = DataLoader(TensorDataset(torch.tensor(X_train_raw, dtype=torch.float32), torch.tensor(y_train_mlp).unsqueeze(1)), batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
        mlp_model = DeepMLP(X_train_raw.shape[1]).to(device)
        mlp_optim = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
        mlp_model.train()
        for e in range(MLP_EPOCHS):
            for bx, by in mlp_loader:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                mlp_optim.zero_grad()
                loss = nn.BCELoss()(mlp_model(bx), by)
                loss.backward()
                mlp_optim.step()
        mlp_model.eval()

        X_train_tensor = torch.tensor(X_train_raw, dtype=torch.float32)
        residual_list = []
        with torch.no_grad():
            for i in range(0, len(X_train_tensor), BATCH_SIZE):
                batch = X_train_tensor[i:i+BATCH_SIZE].to(device)
                recon, _ = hybrid_dae(batch)
                residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())
        residual_train = torch.cat(residual_list).numpy()
        var_train = pd.Series(residual_train).rolling(window=15, min_periods=1).var().fillna(0).values
        X_train_aug = np.hstack([X_train_raw, residual_train.reshape(-1, 1), var_train.reshape(-1, 1)])
        
        hybrid_rf = RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        hybrid_rf.fit(X_train_aug, y_train_raw)
        
        residual_memory = deque(residual_train[y_train_raw == "BENIGN"][-WINDOW_SIZE:], maxlen=WINDOW_SIZE)

        eval_df = pd.concat([benign_df.sample(min(300000, len(benign_df)), random_state=seed), zero_df])
        X_eval = scaler.transform(eval_df.drop("Label", axis=1))
        y_eval = eval_df["Label"].values
        y_eval_binary = np.where(y_eval == "BENIGN", 0, 1)
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

        rf_preds_raw = std_rf.predict(X_eval)
        rf_preds = np.where(rf_preds_raw == "BENIGN", "BENIGN", "ZERO_DAY")
        
        if_preds_raw = iso_f.predict(X_eval)
        if_preds = np.where(if_preds_raw == -1, "ZERO_DAY", "BENIGN")

        with torch.no_grad():
            mlp_probs = torch.cat([mlp_model(X_eval_tensor[i:i+BATCH_SIZE].to(device)).cpu() for i in range(0, len(X_eval_tensor), BATCH_SIZE)]).numpy().flatten()
        mlp_preds = np.where(mlp_probs > 0.5, 1, 0)
        
        kitnet_res = kitnet_score(X_eval_tensor)
        kitnet_preds = np.where(kitnet_res > KITNET_THRESHOLD, "ZERO_DAY", "BENIGN")

        res_eval_list = []
        with torch.no_grad():
            for i in range(0, len(X_eval_tensor), BATCH_SIZE):
                batch = X_eval_tensor[i:i+BATCH_SIZE].to(device)
                recon, _ = hybrid_dae(batch)
                res_eval_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())
        res_eval = torch.cat(res_eval_list).numpy()
        var_eval = pd.Series(res_eval).rolling(window=15, min_periods=1).var().fillna(0).values
        X_eval_aug = np.hstack([X_eval, res_eval.reshape(-1, 1), var_eval.reshape(-1, 1)])
        
        hrf_preds = hybrid_rf.predict(X_eval_aug)
        hrf_probs = np.max(hybrid_rf.predict_proba(X_eval_aug), axis=1)

        hybrid_final_preds = []
        threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)
        for i in range(len(X_eval)):
            if i > 0 and i % 1000 == 0: threshold = np.percentile(residual_memory, THRESHOLD_PERCENTILE)
            r, p, pr = res_eval[i], hrf_preds[i], hrf_probs[i]
            
            if r > threshold:
                final = "BENIGN" if (p == "BENIGN" and pr >= ALPHA_BENIGN) else "ZERO_DAY"
            else: final = p 
                
            hybrid_final_preds.append(final)
            if final == "BENIGN": residual_memory.append(r)
        
        hybrid_final_preds = np.array(hybrid_final_preds)

        all_results.append({
            "attack": ZERO_DAY,
            "Hybrid": recall_score(y_eval == ZERO_DAY, hybrid_final_preds == "ZERO_DAY"),
            "KitNET": recall_score(y_eval == ZERO_DAY, kitnet_preds == "ZERO_DAY"),
            "MLP": recall_score(y_eval_binary == 1, mlp_preds == 1),
            "IF": recall_score(y_eval == ZERO_DAY, if_preds == "ZERO_DAY"),
            "RF": recall_score(y_eval == ZERO_DAY, rf_preds == "ZERO_DAY"),
            "b_Hybrid": recall_score(y_eval == "BENIGN", hybrid_final_preds == "BENIGN"),
            "b_KitNET": recall_score(y_eval == "BENIGN", kitnet_preds == "BENIGN"),
            "b_MLP": recall_score(y_eval_binary == 0, mlp_preds == 0),
            "b_IF": recall_score(y_eval == "BENIGN", if_preds == "BENIGN"),
            "b_RF": recall_score(y_eval == "BENIGN", rf_preds == "BENIGN")
        })

        del train_df, zero_df, benign_df, X_train_raw, X_train_aug, X_eval, X_eval_aug, X_eval_tensor, X_train_tensor
        del mlp_model, mlp_optim, mlp_loader, hybrid_rf, std_rf, iso_f, residual_train, res_eval, var_train, var_eval
        gc.collect(); torch.cuda.empty_cache()

    print("\n[System] All LOAO evaluations complete. Generating outputs...")
    
    generate_text_table(all_results)
    generate_radar_chart(all_results)
    
    print("\n[System] Master Script Completed. Radar chart saved to directory.")