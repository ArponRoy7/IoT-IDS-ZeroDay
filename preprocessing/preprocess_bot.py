import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir = os.path.join(BASE_DIR, "data", "raw", "bot_iot")
output_path = os.path.join(BASE_DIR, "data", "processed", "bot_iot_clean.csv")

files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

df_list = []

for file in files:
    print("Reading:", file)
    df_list.append(pd.read_csv(
        os.path.join(raw_dir, file),
        low_memory=False
    ))

df = pd.concat(df_list, ignore_index=True)
df.columns = df.columns.str.strip()

print("Original shape:", df.shape)

# =========================================================
# USE CATEGORY AS LABEL
# =========================================================

df.rename(columns={"category": "Label"}, inplace=True)

# Convert BENIGN correctly
df["Label"] = df["Label"].replace({
    "Normal": "BENIGN",
    "normal": "BENIGN"
})

# Keep only BENIGN and DDoS for Level 1
df = df[df["Label"].isin(["BENIGN", "DDoS"])]

print("After filtering:", df["Label"].value_counts())

# =========================================================
# REMOVE IDENTIFIERS
# =========================================================

drop_cols = [
    "pkSeqID", "stime", "ltime", "seq",
    "saddr", "daddr", "sport", "dport",
    "attack", "subcategory"
]

df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Keep numeric only + Label
numeric_cols = df.select_dtypes(include=np.number).columns
df = df[numeric_cols.tolist() + ["Label"]]

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

print("Final shape:", df.shape)

df.to_csv(output_path, index=False)
print("Saved to:", output_path)