# =========================================================
# HYPER-TUNED HYBRID LOAO – ADAPTIVE SLIDING WINDOW (FULL)
# FULL DATASET VERSION — MAX UTILIZATION
# Window=100000 | Percentile=90 | Alpha=0.999
# LOGIC SAME — ONLY DATA LIMITS REMOVED
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from preprocessing.preprocess_ciciot2023 import load_clean_ciciot2023
from collections import deque
import random

# =========================================================
# HYPERPARAMETERS
# =========================================================

WINDOW_SIZE = 100000
THRESHOLD_PERCENTILE = 90
ALPHA_BENIGN = 0.999
ALPHA_ATTACK = 0.85

# =========================================================

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

print("\nHyper-parameters:")
print("Window Size:", WINDOW_SIZE)
print("Percentile:", THRESHOLD_PERCENTILE)
print("Alpha Benign:", ALPHA_BENIGN)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_ciciot2023()

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

ZERO_DAY_LIST = [
    label for label in df["Label"].unique()
    if label != "BENIGN"
]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

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
    # DAE
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
        batch_size=8192,
        shuffle=True
    )

    model.train()

    print("Training DAE FULL...")

    for epoch in range(50):

        total_loss=0

        for (x,) in loader:

            x = x.to(device)

            noise = torch.randn_like(x)*0.05

            recon,_ = model(x+noise)

            loss = criterion(recon,x)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss+=loss.item()

        print("Epoch",epoch+1,"Loss:",round(total_loss,4))

    model.eval()

    # =====================================================
    # SLIDING WINDOW FULL
    # =====================================================

    residual_memory = deque(maxlen=WINDOW_SIZE)

    with torch.no_grad():

        for (x,) in loader:

            x = x.to(device)

            recon,_ = model(x)

            residual = torch.mean((recon-x)**2,dim=1).cpu().numpy()

            residual_memory.extend(residual)

    print("Sliding Window Initialized")

    # =====================================================
    # RF TRAIN FULL DATASET
    # =====================================================

    print("Training RF FULL dataset")

    rf_sample = train_df

    X_rf = scaler.transform(rf_sample.drop("Label", axis=1))
    y_rf = rf_sample["Label"]

    X_rf_tensor = torch.tensor(X_rf, dtype=torch.float32)

    residual_list = []

    with torch.no_grad():

        for i in range(0,len(X_rf_tensor),8192):

            x = X_rf_tensor[i:i+8192].to(device)

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

    print("RF trained FULL")

    # =====================================================
    # FULL EVALUATION
    # =====================================================

    print("Running FULL evaluation")

    eval_df = pd.concat([
        benign_df.sample(500000, random_state=seed),
        zero_df
    ])

    X_eval = scaler.transform(eval_df.drop("Label", axis=1))
    y_eval = eval_df["Label"].values

    X_eval_tensor = torch.tensor(X_eval,dtype=torch.float32)

    residual_list=[]

    with torch.no_grad():

        for i in range(0,len(X_eval_tensor),8192):

            batch = X_eval_tensor[i:i+8192].to(device)

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

                final_pred="BENIGN" if rf_prob>=ALPHA_BENIGN else "ZERO_DAY"

            else:

                final_pred=rf_pred if rf_prob>=ALPHA_ATTACK else "ZERO_DAY"

        else:

            final_pred=rf_pred

        hybrid_preds.append(final_pred)

        if final_pred=="BENIGN":

            residual_memory.append(residual)

    hybrid_preds=np.array(hybrid_preds)

    print("Benign Recall:",
        round(recall_score(y_eval=="BENIGN", hybrid_preds=="BENIGN"),4))

    print("Zero-Day Recall:",
        round(recall_score(y_eval==ZERO_DAY, hybrid_preds=="ZERO_DAY"),4))

print("\nFULL UTILIZATION HYPER-TUNED COMPLETED")