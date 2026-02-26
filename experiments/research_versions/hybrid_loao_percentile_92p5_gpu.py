# =========================================================
# HYPER-TUNED HYBRID LOAO – ADAPTIVE SLIDING WINDOW
# ALL ATTACKS VERSION + FINAL SUMMARY TABLE
# Window=30000 | Percentile=92.5 | Alpha=0.999
# =========================================================

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

# =========================================================
# HYPERPARAMETERS
# =========================================================

WINDOW_SIZE = 30000
THRESHOLD_PERCENTILE = 92.5   # ✅ ONLY UPDATED VALUE
ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85

# =========================================================

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

print("\nHyperparameters:")
print("Window Size:", WINDOW_SIZE)
print("Threshold Percentile:", THRESHOLD_PERCENTILE)
print("Alpha Benign:", ALPHA_BENIGN)
print("Alpha Attack:", ALPHA_ATTACK)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# AUTO-DETECT ALL ATTACKS
# =========================================================

ZERO_DAY_LIST = [
    label for label in df["Label"].unique()
    if label != "BENIGN"
]

print("\nAll detected attacks:")
print(ZERO_DAY_LIST)

# =========================================================

seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# =========================================================
# STORE RESULTS
# =========================================================

results = []

# =========================================================
# LOOP OVER ALL ATTACKS
# =========================================================

for ZERO_DAY in ZERO_DAY_LIST:

    print("\n" + "="*80)
    print("TEST:", ZERO_DAY)
    print("="*80)

    train_df = df[df["Label"] != ZERO_DAY]

    zero_df  = df[df["Label"] == ZERO_DAY]

    benign_df = train_df[train_df["Label"] == "BENIGN"]

    scaler = StandardScaler()

    scaler.fit(benign_df.drop("Label", axis=1))

    # =====================================================
    # TRAIN DAE
    # =====================================================

    X_benign = scaler.transform(benign_df.drop("Label", axis=1))

    X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

    class DAE(nn.Module):

        def __init__(self, dim):

            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(dim,256),
                nn.ReLU(),
                nn.Linear(256,64),
                nn.ReLU(),
                nn.Linear(64,8)
            )

            self.decoder = nn.Sequential(
                nn.Linear(8,64),
                nn.ReLU(),
                nn.Linear(64,256),
                nn.ReLU(),
                nn.Linear(256,dim)
            )

        def forward(self,x):

            z = self.encoder(x)
            recon = self.decoder(z)

            return recon,z

    model = DAE(X_benign.shape[1]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_benign_tensor),
        batch_size=4096,
        shuffle=True
    )

    model.train()

    for epoch in range(20):

        for (x,) in loader:

            x = x.to(device)

            noise = torch.randn_like(x)*0.05

            recon,_ = model(x+noise)

            loss = criterion(recon,x)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

    model.eval()

    print("DAE trained")

    # =====================================================
    # SLIDING WINDOW INIT
    # =====================================================

    residual_memory = deque(maxlen=WINDOW_SIZE)

    with torch.no_grad():

        for (x,) in loader:

            x = x.to(device)

            recon,_ = model(x)

            residual = torch.mean((recon-x)**2,dim=1).cpu().numpy()

            for r in residual:

                residual_memory.append(r)

    print("Sliding Window Initialized")

    # =====================================================
    # RF TRAIN
    # =====================================================

    rf_sample = train_df.sample(250000, random_state=seed)

    X_rf = scaler.transform(rf_sample.drop("Label", axis=1))

    y_rf = rf_sample["Label"]

    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)

    residual_list = []

    with torch.no_grad():

        for i in range(0,len(X_rf_tensor),4096):

            x = X_rf_tensor[i:i+4096].to(device)

            recon,_ = model(x)

            residual_list.append(
                torch.mean((recon-x)**2,dim=1).cpu()
            )

    residual_rf = torch.cat(residual_list).numpy()

    X_rf_aug = np.hstack([X_rf,residual_rf.reshape(-1,1)])

    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_rf_aug,y_rf)

    print("RF trained")

    # =====================================================
    # EVALUATION
    # =====================================================

    eval_df = pd.concat([
        train_df.sample(60000, random_state=seed),
        zero_df.sample(min(10000,len(zero_df)), random_state=seed)
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))

    y_eval = eval_df["Label"].values

    X_eval_tensor = torch.tensor(X_eval,dtype=torch.float32)

    residual_list=[]

    with torch.no_grad():

        for i in range(0,len(X_eval_tensor),4096):

            batch = X_eval_tensor[i:i+4096].to(device)

            recon,_ = model(batch)

            residual_list.append(
                torch.mean((recon-batch)**2,dim=1).cpu()
            )

    residual_eval = torch.cat(residual_list).numpy()

    X_eval_aug = np.hstack([X_eval,residual_eval.reshape(-1,1)])

    rf_preds = rf.predict(X_eval_aug)

    rf_probs = rf.predict_proba(X_eval_aug)

    hybrid_preds=[]

    for i in range(len(X_eval)):

        residual = residual_eval[i]

        threshold = np.percentile(residual_memory,THRESHOLD_PERCENTILE)

        rf_pred = rf_preds[i]

        rf_prob = np.max(rf_probs[i])

        if residual>threshold:

            if rf_pred=="BENIGN":

                final_pred = "BENIGN" if rf_prob>=ALPHA_BENIGN else "ZERO_DAY"

            else:

                final_pred = rf_pred if rf_prob>=ALPHA_ATTACK else "ZERO_DAY"

        else:

            final_pred = rf_pred

        hybrid_preds.append(final_pred)

        if final_pred=="BENIGN":

            residual_memory.append(residual)

    hybrid_preds = np.array(hybrid_preds)

    benign_recall = recall_score(
        y_eval=="BENIGN",
        hybrid_preds=="BENIGN"
    )

    zero_recall = recall_score(
        y_eval==ZERO_DAY,
        hybrid_preds=="ZERO_DAY"
    )

    print("Benign Recall:", round(benign_recall,4))
    print("Zero-Day Recall:", round(zero_recall,4))

    results.append([ZERO_DAY, benign_recall, zero_recall])

# =========================================================
# FINAL SUMMARY TABLE
# =========================================================

results_df = pd.DataFrame(
    results,
    columns=["Attack","Benign Recall","Zero-Day Recall"]
)

print("\n" + "="*80)
print("FINAL SUMMARY TABLE")
print("="*80)

print(results_df.to_string(index=False))

print("\nMEAN VALUES")
print(results_df.mean(numeric_only=True))