import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from preprocessing.preprocess_cicids import load_clean_cicids

print("Loading dataset...")
df = load_clean_cicids()

# =====================================================
# STEP 1: Downsample BENIGN for memory safety
# =====================================================

print("\nDownsampling BENIGN traffic...")

benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

attack_df = df[df["Label"] != "BENIGN"]

print("Benign used for training:", benign_df.shape)
print("Attack samples available:", attack_df.shape)

# =====================================================
# STEP 2: Prepare training data (BENIGN only)
# =====================================================

X_train = benign_df.drop("Label", axis=1)

USE_SCALING = True

if USE_SCALING:
    print("\nApplying StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

# =====================================================
# STEP 3: Train Isolation Forest (NO contamination)
# =====================================================

print("\nTraining Isolation Forest (contamination='auto')...")

model = IsolationForest(
    n_estimators=100,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train)

print("Training complete.")

# =====================================================
# STEP 4: Compute Scores
# =====================================================

print("\nComputing anomaly scores...")

train_scores = model.decision_function(X_train)

test_df = pd.concat([benign_df, attack_df])
X_test = test_df.drop("Label", axis=1)

if USE_SCALING:
    X_test = scaler.transform(X_test)

y_true_binary = (test_df["Label"] != "BENIGN").astype(int)

test_scores = model.decision_function(X_test)

# =====================================================
# STEP 5: Threshold Tuning
# =====================================================

print("\nEvaluating different FPR targets...")

fpr_targets = [0.005, 0.01, 0.02, 0.05]

results = []

for fpr_target in fpr_targets:

    threshold = np.percentile(train_scores, fpr_target * 100)

    y_pred_binary = (test_scores < threshold).astype(int)

    benign_mask = (y_true_binary == 0)
    attack_mask = (y_true_binary == 1)

    false_positives = (y_pred_binary[benign_mask] == 1).sum()
    total_benign = benign_mask.sum()

    true_positives = (y_pred_binary[attack_mask] == 1).sum()
    total_attack = attack_mask.sum()

    fpr = false_positives / total_benign
    tpr = true_positives / total_attack

    print("\n----------------------------------")
    print(f"Target FPR: {fpr_target}")
    print(f"Actual FPR: {fpr:.4f}")
    print(f"Attack Detection Rate (TPR): {tpr:.4f}")

    results.append({
        "Target_FPR": fpr_target,
        "Actual_FPR": fpr,
        "Detection_Rate": tpr
    })

# =====================================================
# STEP 6: SELECT PRIMARY OPERATING POINT
# =====================================================

PRIMARY_FPR = 0.01  # 🔥 Your chosen operating point

print("\n\n==============================")
print(f"PRIMARY OPERATING POINT = {PRIMARY_FPR*100:.1f}% FPR")
print("==============================")

final_threshold = np.percentile(train_scores, PRIMARY_FPR * 100)

y_final = (test_scores < final_threshold).astype(int)

benign_mask = (y_true_binary == 0)
attack_mask = (y_true_binary == 1)

final_fp = (y_final[benign_mask] == 1).sum()
final_tp = (y_final[attack_mask] == 1).sum()

final_fpr = final_fp / benign_mask.sum()
final_tpr = final_tp / attack_mask.sum()

print(f"Final Threshold: {final_threshold:.6f}")
print(f"Final FPR: {final_fpr:.4f}")
print(f"Final Detection Rate (TPR): {final_tpr:.4f}")

# =====================================================
# STEP 7: ROC Curve
# =====================================================

print("\nGenerating ROC curve...")

fpr_vals, tpr_vals, thresholds = roc_curve(y_true_binary, -test_scores)
roc_auc = auc(fpr_vals, tpr_vals)

plt.figure(figsize=(8,6))
plt.plot(fpr_vals, tpr_vals, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Isolation Forest")
plt.legend()
plt.grid(True)
plt.show()

# =====================================================
# STEP 8: Final Summary Table
# =====================================================

results_df = pd.DataFrame(results)

print("\n\n=========== THRESHOLD TUNING SUMMARY ===========")
print(results_df)
