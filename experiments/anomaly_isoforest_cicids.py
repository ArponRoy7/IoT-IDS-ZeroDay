import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from preprocessing.preprocess_cicids import load_clean_cicids

print("Loading dataset...")
df = load_clean_cicids()

# -------------------------------------------------
# Downsample BENIGN (for memory safety)
# -------------------------------------------------

print("\nDownsampling BENIGN traffic...")

benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

print("Benign shape used for training:", benign_df.shape)

# -------------------------------------------------
# Train Isolation Forest on BENIGN only
# -------------------------------------------------

X_train = benign_df.drop("Label", axis=1)

print("\nTraining Isolation Forest on BENIGN only...")

model = IsolationForest(
    n_estimators=100,
    contamination=0.02,   # expected anomaly proportion
    random_state=42,
    n_jobs=-1
)

model.fit(X_train)

print("Training complete.")

# -------------------------------------------------
# Evaluate on selected attack types
# -------------------------------------------------

test_attacks = [
    "DDoS",
    "PortScan",
    "Web Attack � Sql Injection",
    "Infiltration"
]

results = []

for attack in test_attacks:

    print("\n=======================================")
    print(f"Testing anomaly detection for: {attack}")
    print("=======================================")

    attack_df = df[df["Label"] == attack]

    if len(attack_df) == 0:
        print("No samples found.")
        continue

    X_test = attack_df.drop("Label", axis=1)

    # IsolationForest outputs:
    #  1  = normal
    # -1  = anomaly

    preds = model.predict(X_test)

    anomaly_detected = (preds == -1).sum()
    total = len(preds)

    detection_rate = anomaly_detected / total

    print(f"Total samples: {total}")
    print(f"Detected as anomaly: {anomaly_detected}")
    print(f"Anomaly Detection Rate: {detection_rate:.4f}")

    results.append({
        "Attack": attack,
        "Samples": total,
        "Detected as Anomaly": anomaly_detected,
        "Detection Rate": detection_rate
    })

# -------------------------------------------------
# Final Summary
# -------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\n=========== ANOMALY DETECTION SUMMARY ===========")
print(results_df.sort_values("Detection Rate"))
