import os
import pandas as pd
import numpy as np

folder = "../data/raw/cicids"
files = [f for f in os.listdir(folder) if f.endswith(".csv")]

dfs = []

for file in files:
    print(f"Loading {file}...")
    df = pd.read_csv(os.path.join(folder, file), low_memory=False)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

print("Total Shape Before Cleaning:", combined_df.shape)

# ==============================
# CLEANING STEP (FREEZE STAGE)
# ==============================

# Strip column names (VERY IMPORTANT for CICIDS)
combined_df.columns = combined_df.columns.str.strip()

# Replace infinity values
combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop NaN rows
combined_df.dropna(inplace=True)

print("After Cleaning Shape:", combined_df.shape)

print("\nOverall Label Distribution:")
print(combined_df["Label"].value_counts())
combined_df.to_csv("../data/processed/cicids2017_clean.csv", index=False)
