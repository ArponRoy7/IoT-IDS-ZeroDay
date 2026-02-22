import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from preprocessing.preprocess_cicids import load_clean_cicids


print("Loading dataset...")
df = load_clean_cicids()

# -------------------------------------------------
# Downsample BENIGN (for memory safety)
# -------------------------------------------------

print("\nPreparing BENIGN training data...")

benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

X_train = benign_df.drop("Label", axis=1)

# -------------------------------------------------
# Scale features (important for anomaly stability)
# -------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -------------------------------------------------
# Train Isolation Forest
# -------------------------------------------------

print("\nTraining Isolation Forest (contamination='auto')...")

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled)

print("Training complete.")

# -------------------------------------------------
# Attack-wise ROC analysis
# -------------------------------------------------

test_attacks = [
    "DDoS",
    "PortScan",
    "Infiltration"
]

plt.figure(figsize=(8,6))

for attack in test_attacks:

    print(f"\nProcessing ROC for: {attack}")

    attack_df = df[df["Label"] == attack]
    benign_test = df[df["Label"] == "BENIGN"].sample(
        n=len(attack_df),
        random_state=42
    )

    combined = pd.concat([benign_test, attack_df])

    X_test = combined.drop("Label", axis=1)
    y_true = np.array(
        [0]*len(benign_test) + [1]*len(attack_df)
    )

    X_test_scaled = scaler.transform(X_test)

    scores = -model.decision_function(X_test_scaled)

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    print(f"AUC for {attack}: {roc_auc:.4f}")

    plt.plot(fpr, tpr, label=f"{attack} (AUC={roc_auc:.3f})")

# Random baseline
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Attack-wise ROC Curves - Isolation Forest")
plt.legend()
plt.grid(True)
plt.show()
