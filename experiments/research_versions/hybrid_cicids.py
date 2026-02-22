import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from preprocessing.preprocess_cicids import load_clean_cicids

print("Loading dataset...")
df = load_clean_cicids()

# -------------------------------------------------
# STEP 1: Downsample BENIGN for memory safety
# -------------------------------------------------

benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

attack_df = df[df["Label"] != "BENIGN"]

print("Benign:", benign_df.shape)
print("Attack:", attack_df.shape)

# -------------------------------------------------
# STEP 2: Train Isolation Forest (Stage 1)
# -------------------------------------------------

print("\nTraining Isolation Forest...")

X_benign = benign_df.drop("Label", axis=1)

scaler = StandardScaler()
X_benign_scaled = scaler.fit_transform(X_benign)

iso_model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

iso_model.fit(X_benign_scaled)

# Define operating FPR
TARGET_FPR = 0.01
train_scores = iso_model.decision_function(X_benign_scaled)
threshold = np.percentile(train_scores, TARGET_FPR * 100)

print(f"Stage 1 Threshold (FPR {TARGET_FPR*100}%): {threshold}")

# -------------------------------------------------
# STEP 3: Train Random Forest (Stage 2)
# -------------------------------------------------

print("\nTraining Random Forest on known classes...")

rf_train_df = df[df["Label"] != "BENIGN"]

X_rf = rf_train_df.drop("Label", axis=1)
y_rf = rf_train_df["Label"]

X_rf_scaled = scaler.transform(X_rf)

rf_model = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_rf_scaled, y_rf)

print("Stage 2 training complete.")

# -------------------------------------------------
# STEP 4: Hybrid Evaluation
# -------------------------------------------------

print("\nRunning Hybrid Evaluation...")

test_df = df.sample(n=100000, random_state=42)

X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

X_test_scaled = scaler.transform(X_test)

# Stage 1
scores = iso_model.decision_function(X_test_scaled)
anomaly_mask = scores < threshold

final_predictions = []

for i in range(len(X_test_scaled)):

    if anomaly_mask[i]:
        # Stage 2 classify
        pred = rf_model.predict(
            X_test_scaled[i].reshape(1, -1)
        )[0]
    else:
        pred = "BENIGN"

    final_predictions.append(pred)

final_predictions = np.array(final_predictions)

print("\nHybrid Classification Report:")
print(classification_report(y_test, final_predictions))
