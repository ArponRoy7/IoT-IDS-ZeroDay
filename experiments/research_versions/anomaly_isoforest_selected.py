import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from preprocessing.preprocess_cicids import load_clean_cicids

print("Loading dataset...")
df = load_clean_cicids()

# --------------------------------------------------
# Load Selected Features
# --------------------------------------------------

with open("models/top_features.pkl", "rb") as f:
    top_features = pickle.load(f)

print("Using features:", len(top_features))

df = df[top_features + ["Label"]]

# --------------------------------------------------
# Prepare BENIGN training data
# --------------------------------------------------

benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

attack_df = df[df["Label"] != "BENIGN"]

X_train = benign_df.drop("Label", axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# --------------------------------------------------
# Train Isolation Forest
# --------------------------------------------------

print("Training Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train)

# --------------------------------------------------
# Evaluate AUC
# --------------------------------------------------

test_df = pd.concat([benign_df, attack_df])

X_test = scaler.transform(test_df.drop("Label", axis=1))
y_binary = (test_df["Label"] != "BENIGN").astype(int)

scores = model.decision_function(X_test)

auc = roc_auc_score(y_binary, -scores)

print("\nIsolation Forest AUC with selected features:", auc)
