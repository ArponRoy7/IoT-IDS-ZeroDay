import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_dir = os.path.join(BASE_DIR, "data", "raw", "cicids")
output_path = os.path.join(BASE_DIR, "data", "processed", "cicids2017_enhanced.csv")

print("Building enhanced CICIDS dataset...")

files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

df_list = []

for file in files:
    print("Reading:", file)
    path = os.path.join(raw_dir, file)
    temp_df = pd.read_csv(path)
    df_list.append(temp_df)

df = pd.concat(df_list, ignore_index=True)
df.columns = df.columns.str.strip()

eps = 1e-6

# =========================================================
# DDoS Features
# =========================================================
df["Packets_Per_Second"] = df["Total Fwd Packets"] / (df["Flow Duration"] + eps)
df["Bytes_Per_Second"] = df["Total Length of Fwd Packets"] / (df["Flow Duration"] + eps)
df["Fwd_Bwd_Ratio"] = df["Total Fwd Packets"] / (df["Total Backward Packets"] + eps)

# =========================================================
# Infiltration Features
# =========================================================
df["Activity_Ratio"] = df["Active Mean"] / (df["Idle Mean"] + eps)
df["IAT_Variability"] = df["Flow IAT Std"] / (df["Flow IAT Mean"] + eps)

# =========================================================
# PortScan Approximation Features
# =========================================================
df["Small_Flow_Flag"] = (df["Total Fwd Packets"] <= 3).astype(int)
df["Short_Duration_Flag"] = (df["Flow Duration"] < 50000).astype(int)
df["Low_Bytes_Flag"] = (df["Total Length of Fwd Packets"] < 100).astype(int)

# Combined scan score proxy
df["Scan_Score"] = (
    df["Small_Flow_Flag"] +
    df["Short_Duration_Flag"] +
    df["Low_Bytes_Flag"]
)

# =========================================================
# Log transform positive features
# =========================================================
numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    if col != "Label":
        if df[col].min() >= 0:
            df[f"log_{col}"] = np.log1p(df[col])

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

print("Final shape:", df.shape)

df.to_csv(output_path, index=False)
print("Enhanced dataset saved to:", output_path)
