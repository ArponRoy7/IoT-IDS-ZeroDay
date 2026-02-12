import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocessing.preprocess_cicids import load_clean_cicids

print("Loading dataset...")
df = load_clean_cicids()

# -------------------------------------------------
# Downsample BENIGN class (for memory efficiency)
# -------------------------------------------------

print("\nDownsampling BENIGN traffic...")

benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000, random_state=42
)

attack_df = df[df["Label"] != "BENIGN"]

df = pd.concat([benign_df, attack_df])

print("After downsampling shape:", df.shape)

# -------------------------------------------------
# Define zero-day attack
# -------------------------------------------------

zero_day_attack = " Web Attack � Brute Force"

print(f"\nRunning LOAO for attack: {zero_day_attack}")

# -------------------------------------------------
# Manual LOAO split
# -------------------------------------------------

train_df = df[df["Label"] != zero_day_attack]
test_df = df[df["Label"] == zero_day_attack]

print("\nTrain shape:", train_df.shape)
print("Test shape (zero-day):", test_df.shape)

# Separate features and labels
X_train = train_df.drop("Label", axis=1)
y_train = train_df["Label"]

X_test = test_df.drop("Label", axis=1)
y_test = test_df["Label"]

# -------------------------------------------------
# Train Random Forest
# -------------------------------------------------

print("\nTraining Random Forest (no scaling)...")

model = RandomForestClassifier(
    n_estimators=50,      # reduced for faster experiment
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------------------------
# Evaluate on zero-day attack
# -------------------------------------------------

print("\nEvaluating on unseen attack...")

y_pred = model.predict(X_test)

print("\nPredicted class distribution:")
print(pd.Series(y_pred).value_counts())

print("\nClassification Report (Zero-Day Only):")
print(classification_report(y_test, y_pred, zero_division=0))

# -------------------------------------------------
# Zero-day detection rate
# -------------------------------------------------

correct = (y_pred == zero_day_attack).sum()
total = len(y_test)

print("\nZero-Day Detection Rate:")
print(f"{correct} / {total} correctly detected as {zero_day_attack}")
print(f"Detection Rate: {correct/total:.4f}")
