import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from preprocessing.preprocess_cicids import load_clean_cicids

print("Loading dataset...")
df = load_clean_cicids()

# -------------------------------------------------
# Downsample BENIGN (controlled training size)
# -------------------------------------------------

print("\nPreparing BENIGN training data...")
benign_train = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

X_train = benign_train.drop("Label", axis=1)

# Separate test sets
benign_test = df[df["Label"] == "BENIGN"].sample(
    n=50000,
    random_state=24
)

X_benign_test = benign_test.drop("Label", axis=1)

# Choose major attacks to evaluate
attack_list = ["DDoS", "PortScan"]

attack_tests = {
    attack: df[df["Label"] == attack].drop("Label", axis=1)
    for attack in attack_list
}

# -------------------------------------------------
# Contamination values to test
# -------------------------------------------------

contamination_values = [0.005, 0.01, 0.02, 0.05, 0.1]

results = []

# -------------------------------------------------
# Run experiments
# -------------------------------------------------

for contamination in contamination_values:

    print("\n===========================================")
    print(f"Testing contamination = {contamination}")
    print("===========================================")

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train)

    # ---- False Positive Rate ----
    benign_preds = model.predict(X_benign_test)
    benign_anomalies = (benign_preds == -1).sum()
    fpr = benign_anomalies / len(benign_preds)

    print(f"False Positive Rate: {fpr:.4f}")

    attack_detection = {}

    for attack_name, X_attack in attack_tests.items():

        attack_preds = model.predict(X_attack)
        detected = (attack_preds == -1).sum()
        detection_rate = detected / len(attack_preds)

        attack_detection[attack_name] = detection_rate

        print(f"{attack_name} Detection Rate: {detection_rate:.4f}")

    results.append({
        "contamination": contamination,
        "FPR": fpr,
        **attack_detection
    })

# -------------------------------------------------
# Convert to DataFrame
# -------------------------------------------------

results_df = pd.DataFrame(results)

print("\n\n=========== TUNING SUMMARY ===========")
print(results_df)

# -------------------------------------------------
# Plot Detection vs FPR
# -------------------------------------------------

plt.figure(figsize=(8,6))

for attack in attack_list:
    plt.plot(results_df["FPR"], results_df[attack], marker='o', label=attack)

plt.xlabel("False Positive Rate")
plt.ylabel("Detection Rate")
plt.title("Detection vs False Positive Rate")
plt.legend()
plt.grid(True)

plt.show()
