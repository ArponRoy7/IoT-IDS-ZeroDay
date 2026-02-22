# =========================================================
# STAGE-2 RANDOM FOREST SUPERVISED EVALUATION
# FULL MULTI-CLASS PERFORMANCE
# =========================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from preprocessing.preprocess_cicids import load_clean_cicids

print("="*70)
print("STAGE-2 RANDOM FOREST SUPERVISED EVALUATION")
print("="*70)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()
df.columns = df.columns.str.strip()

eps = 1e-6

if "Flow Duration" in df.columns:
    df["Packets_Per_Second"] = df["Total Fwd Packets"] / (df["Flow Duration"].abs() + eps)
    df["Bytes_Per_Second"] = df["Total Length of Fwd Packets"] / (df["Flow Duration"].abs() + eps)

if "Total Backward Packets" in df.columns:
    df["Fwd_Bwd_Ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"].abs() + eps)

numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].clip(-1e9, 1e9)

log_dict = {
    f"log_{col}": np.log1p(df[col])
    for col in numeric_cols
    if col != "Label" and df[col].min() >= 0
}

df = pd.concat([df, pd.DataFrame(log_dict)], axis=1)
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# FEATURES & LABELS
# =========================================================

X = df.drop("Label", axis=1)
y = df["Label"]

# =========================================================
# STRATIFIED TRAIN-TEST SPLIT (NO ZERO-DAY REMOVAL)
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# =========================================================
# FEATURE SCALING
# =========================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================================================
# RANDOM FOREST TRAINING
# =========================================================

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)

# =========================================================
# TRAIN PERFORMANCE
# =========================================================

y_train_pred = rf.predict(X_train)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_macro_f1 = f1_score(y_train, y_train_pred, average="macro")

print("\n================ TRAIN PERFORMANCE ================")
print("Train Accuracy:", round(train_accuracy, 4))
print("Train Macro F1:", round(train_macro_f1, 4))

# =========================================================
# TEST PERFORMANCE
# =========================================================

y_test_pred = rf.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_macro_f1 = f1_score(y_test, y_test_pred, average="macro")
test_weighted_f1 = f1_score(y_test, y_test_pred, average="weighted")

print("\n================ TEST PERFORMANCE ================")
print("Test Accuracy:", round(test_accuracy, 4))
print("Test Macro F1:", round(test_macro_f1, 4))
print("Test Weighted F1:", round(test_weighted_f1, 4))

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\n================ CLASSIFICATION REPORT ================")
print(classification_report(y_test, y_test_pred))

# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n================ CONFUSION MATRIX ================")
cm = confusion_matrix(y_test, y_test_pred)

labels = sorted(y.unique())

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)