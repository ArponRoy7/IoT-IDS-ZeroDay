# =========================================================
# UNIFIED SOTA EVALUATION (HIGH-SPEED & OOM-PROOF)
# Evaluates Hybrid, KitNET, MLP, IF, and Std_RF in one pass.
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

# =========================================================
# PARAMETERS & SETUP
# =========================================================
WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 90
ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85

EPOCHS = 35
MLP_EPOCHS = 5      # Optimized: 5 epochs is plenty for baseline benchmark
BATCH_SIZE = 8192   # Optimized: Massive batch size for GPU acceleration
EXCLUDE_ATTACKS = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# MODEL DEFINITIONS
# =========================================================

class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, dim))
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

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
# TEXT TABLE GENERATOR FUNCTION
# =========================================================
def generate_text_table(results_list):
    print("\n" + "="*98)
    print("FINAL ZERO-DAY RECALL COMPARISON VS BASELINES")
    print("="*98)
    
    categories = {
        "Volumetric & App-Layer DoS": ["DDoS", "DoS Hulk", "DoS Slowhttptest", "DoS slowloris", "DoS GoldenEye"],
        "Complex Exploitation": ["Heartbleed", "Infiltration", "FTP-Patator", "SSH-Patator", "Web Attack - Brute Force", "Web Attack - XSS", "Web Attack - Sql Injection", "Web Attack  Brute Force", "Web Attack  XSS", "Web Attack  Sql Injection"],
        "Reconnaissance": ["PortScan", "Bot"]
    }

    avg_b_hybrid = np.mean([r['b_Hybrid'] for r in results_list]) * 100
    avg_b_kitnet = np.mean([r['b_KitNET'] for r in results_list]) * 100
    avg_b_mlp = np.mean([r['b_MLP'] for r in results_list]) * 100
    avg_b_if = np.mean([r['b_IF'] for r in results_list]) * 100
    avg_b_rf = np.mean([r['b_RF'] for r in results_list]) * 100

    print(f"{'Isolated Zero-Day':<30} | {'Hybrid':<10} | {'KitNET':<10} | {'MLP':<10} | {'IF':<10} | {'Std RF':<10}")
    print("-" * 98)

    processed = set()
    for cat_name, attacks in categories.items():
        valid_attacks = [res for res in results_list if res["attack"] in attacks or res["attack"].replace('\uFFFD', '-') in attacks]
        if not valid_attacks: continue
        
        print(f"[{cat_name}]")
        for res in valid_attacks:
            atk_display = res["attack"].replace('\uFFFD', '-')
            processed.add(res["attack"])
            print(f"{atk_display:<30} | {res['Hybrid']*100:>8.2f}% | {res['KitNET']*100:>8.2f}% | {res['MLP']*100:>8.2f}% | {res['IF']*100:>8.2f}% | {res['RF']*100:>8.2f}%")
        print("-" * 98)

    unmapped = [r for r in results_list if r["attack"] not in processed]
    if unmapped:
        print("[Other Attacks]")
        for res in unmapped:
            atk_display = res["attack"].replace('\uFFFD', '-')
            print(f"{atk_display:<30} | {res['Hybrid']*100:>8.2f}% | {res['KitNET']*100:>8.2f}% | {res['MLP']*100:>8.2f}% | {res['IF']*100:>8.2f}% | {res['RF']*100:>8.2f}%")
        print("-" * 98)

    print(f"{'Benign Average':<30} | {avg_b_hybrid:>8.2f}% | {avg_b_kitnet:>8.2f}% | {avg_b_mlp:>8.2f}% | {avg_b_if:>8.2f}% | {avg_b_rf:>8.2f}%")
    print("=" * 98)

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

    ZERO_DAY_LIST = [lbl for lbl in df["Label"].unique() if lbl != "BENIGN" and lbl not in EXCLUDE_ATTACKS]
    print(f"[System] Found {len(ZERO_DAY_LIST)} attacks to evaluate as Zero-Days.")

    benign_full = df[df["Label"] == "BENIGN"]
    print("[System] Fitting Global StandardScaler on Benign data...")
    scaler = StandardScaler()
    X_benign = scaler.fit_transform(benign_full.drop("Label", axis=1))
    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)
    
    # Optimized DataLoader
    loader = DataLoader(
        TensorDataset(X_benign_tensor), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True
    )

    # --- Train DAE ---
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
            del x, recon, loss
        print(f"  -> DAE Epoch {epoch + 1}/{EPOCHS} Loss: {total_loss:.4f}")
        if epoch > 5 and abs(prev_loss - total_loss) < 1e-4: break
        prev_loss = total_loss
    hybrid_dae.eval()

    # --- Train KitNET ---
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
        print(f"  -> Training KitNET SmallAE {idx + 1}/{NUM_GROUPS}...")
        for epoch in range(5):
            for (x,) in loader:
                x_sub = x[:, FEATURE_SPLITS[idx]].to(device, non_blocking=True)
                opt_ens[idx].zero_grad()
                loss = nn.MSELoss()(ae(x_sub), x_sub)
                loss.backward()
                opt_ens[idx].step()

    print("  -> Training KitNET Combiner...")
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

    print("[Global] Calculating Base KitNET Threshold...")
    KITNET_THRESHOLD = np.percentile(kitnet_score(X_benign_tensor[:50000]), THRESHOLD_PERCENTILE)
    
    del X_benign_tensor, loader
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================
    # ZERO DAY EVALUATION LOOP
    # =========================================================
    all_results = []
    
    for ZERO_DAY in ZERO_DAY_LIST:
        print("\n" + "="*80)
        print(f"Evaluating Isolated Zero-Day: {ZERO_DAY}")
        print("="*80)
        
        train_df = df[df["Label"] != ZERO_DAY]
        zero_df = df[df["Label"] == ZERO_DAY]
        benign_df = train_df[train_df["Label"] == "BENIGN"]

        X_train_raw = scaler.transform(train_df.drop("Label", axis=1))
        y_train_raw = train_df["Label"]
        
        # 1. Train Supervised RF
        print("  -> Training Supervised Std RF Baseline...")
        std_rf = RandomForestClassifier(n_estimators=100, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        std_rf.fit(X_train_raw, y_train_raw)

        # 1b. Train Isolation Forest (NEW)
        print("  -> Training Isolation Forest Baseline...")
        iso_f = IsolationForest(n_estimators=100, contamination=0.01, n_jobs=-1, random_state=seed)
        iso_f.fit(X_train_raw)

        # 2. Train Deep MLP
        print("  -> Training Deep MLP Baseline...")
        y_train_mlp = np.where(y_train_raw == "BENIGN", 0, 1).astype(np.float32)
        # Optimized MLP DataLoader
        mlp_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_raw, dtype=torch.float32), torch.tensor(y_train_mlp).unsqueeze(1)), 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=8, 
            pin_memory=True
        )
        
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

        # 3. Train Hybrid RF
        print("  -> Extracting DAE Residuals for Proposed Hybrid...")
        X_train_tensor = torch.tensor(X_train_raw, dtype=torch.float32)
        residual_list = []
        with torch.no_grad():
            for i in range(0, len(X_train_tensor), BATCH_SIZE):
                batch = X_train_tensor[i:i+BATCH_SIZE].to(device)
                recon, _ = hybrid_dae(batch)
                residual_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())
                del batch, recon
        residual_train = torch.cat(residual_list).numpy()
        
        print("  -> Calculating Rolling Variance for Proposed Hybrid...")
        var_train = pd.Series(residual_train).rolling(window=15, min_periods=1).var().fillna(0).values
        X_train_aug = np.hstack([X_train_raw, residual_train.reshape(-1, 1), var_train.reshape(-1, 1)])
        
        print("  -> Training Proposed Hybrid RF (350 estimators)...")
        hybrid_rf = RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        hybrid_rf.fit(X_train_aug, y_train_raw)
        
        residual_memory = deque(residual_train[y_train_raw == "BENIGN"][-WINDOW_SIZE:], maxlen=WINDOW_SIZE)

        # --- EVALUATION ---
        print("\n  -> Preparing Unified Evaluation Stream...")
        eval_df = pd.concat([benign_df.sample(min(300000, len(benign_df)), random_state=seed), zero_df])
        X_eval = scaler.transform(eval_df.drop("Label", axis=1))
        y_eval = eval_df["Label"].values
        y_eval_binary = np.where(y_eval == "BENIGN", 0, 1)
        X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

        print("  -> Evaluating: Std RF Baseline...")
        rf_preds_raw = std_rf.predict(X_eval)
        # UPDATED RF LOGIC: Binary Anomaly mapping
        rf_preds = np.where(rf_preds_raw == "BENIGN", "BENIGN", "ZERO_DAY")
        
        print("  -> Evaluating: Isolation Forest Baseline...")
        if_preds_raw = iso_f.predict(X_eval)
        # IF returns -1 for outliers (attacks), 1 for inliers (benign)
        if_preds = np.where(if_preds_raw == -1, "ZERO_DAY", "BENIGN")

        print("  -> Evaluating: Deep MLP Baseline...")
        with torch.no_grad():
            mlp_probs = torch.cat([mlp_model(X_eval_tensor[i:i+BATCH_SIZE].to(device)).cpu() for i in range(0, len(X_eval_tensor), BATCH_SIZE)]).numpy().flatten()
        mlp_preds = np.where(mlp_probs > 0.5, 1, 0)
        
        print("  -> Evaluating: KitNET Baseline...")
        kitnet_res = kitnet_score(X_eval_tensor)
        kitnet_preds = np.where(kitnet_res > KITNET_THRESHOLD, "ZERO_DAY", "BENIGN")

        print("  -> Evaluating: Proposed Hybrid...")
        res_eval_list = []
        with torch.no_grad():
            for i in range(0, len(X_eval_tensor), BATCH_SIZE):
                batch = X_eval_tensor[i:i+BATCH_SIZE].to(device)
                recon, _ = hybrid_dae(batch)
                res_eval_list.append(torch.mean((recon - batch) ** 2, dim=1).cpu())
                del batch, recon
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
            else:
                final = p 
                
            hybrid_final_preds.append(final)
            if final == "BENIGN": residual_memory.append(r)
        
        hybrid_final_preds = np.array(hybrid_final_preds)

        # Store metrics
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
        
        print(f"\n     >>>>>>>> RESULTS FOR {ZERO_DAY} <<<<<<<<")
        print(f"     [Hybrid] Zero-Day Recall: {all_results[-1]['Hybrid']:.4f}")
        print(f"     [KitNET] Zero-Day Recall: {all_results[-1]['KitNET']:.4f}")
        print(f"     [MLP]    Zero-Day Recall: {all_results[-1]['MLP']:.4f}")
        print(f"     [IF]     Zero-Day Recall: {all_results[-1]['IF']:.4f}")
        print(f"     [Std RF] Zero-Day Recall: {all_results[-1]['RF']:.4f}")
        print("     >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<")

        print("  -> Flushing GPU & RAM before next attack...")
        del train_df, zero_df, benign_df
        del X_train_raw, X_train_aug, X_eval, X_eval_aug, X_eval_tensor, X_train_tensor
        del mlp_model, mlp_optim, mlp_loader, hybrid_rf, std_rf, iso_f
        del residual_train, res_eval, var_train, var_eval
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[System] Evaluation loops complete. Generating final table...")
    generate_text_table(all_results)