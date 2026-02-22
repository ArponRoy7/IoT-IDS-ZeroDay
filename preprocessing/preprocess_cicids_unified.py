import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(BASE_DIR, "data", "processed", "cicids2017_enhanced.csv")
output_path = os.path.join(BASE_DIR, "data", "processed", "cicids_unified.csv")

df = pd.read_csv(input_path)
eps = 1e-6

df_u = pd.DataFrame()

df_u["duration"] = df["Flow Duration"].abs()

df_u["total_packets"] = (
    df["Total Fwd Packets"] +
    df["Total Backward Packets"]
)

df_u["total_bytes"] = (
    df["Total Length of Fwd Packets"] +
    df["Total Length of Bwd Packets"]
)

df_u["packets_per_second"] = (
    df_u["total_packets"] / (df_u["duration"] + eps)
)

df_u["bytes_per_second"] = (
    df_u["total_bytes"] / (df_u["duration"] + eps)
)

df_u["fwd_bwd_ratio"] = (
    df["Total Fwd Packets"] /
    (df["Total Backward Packets"] + eps)
)

df_u["Label"] = df["Label"]

df_u.replace([np.inf, -np.inf], 0, inplace=True)
df_u.fillna(0, inplace=True)

print("Unified CICIDS shape:", df_u.shape)
df_u.to_csv(output_path, index=False)
print("Saved:", output_path)