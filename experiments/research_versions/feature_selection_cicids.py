import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from preprocessing.preprocess_cicids import load_clean_cicids

print("Loading dataset...")
df = load_clean_cicids()

# --------------------------------------------------
# Downsample BENIGN for memory safety
# --------------------------------------------------

print("Downsampling BENIGN traffic...")

benign_df = df[df["Label"] == "BENIGN"].sample(
    n=200000,
    random_state=42
)

attack_df = df[df["Label"] != "BENIGN"]

df_small = pd.concat([benign_df, attack_df])

print("Dataset used:", df_small.shape)

# --------------------------------------------------
# Prepare data
# --------------------------------------------------

X = df_small.drop("Label", axis=1)
y = df_small["Label"]

print("Training Random Forest for feature importance...")

rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)

rf.fit(X, y)

importances = rf.feature_importances_
feature_names = X.columns

# --------------------------------------------------
# Select Top N Features
# --------------------------------------------------

TOP_N = 25

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

top_features = feature_importance_df.head(TOP_N)["Feature"].tolist()

print("\nTop Features Selected:")
print(top_features)

# --------------------------------------------------
# Save Selected Features
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

with open("models/top_features.pkl", "wb") as f:
    pickle.dump(top_features, f)

print("\nTop features saved to models/top_features.pkl")
