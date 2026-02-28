import os
import pandas as pd

def load_clean_ciciot2023():

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    data_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "ciciot2023_combined.csv"
    )

    print("Loading CICIoT2023 from:", data_path)

    df = pd.read_csv(data_path)

    # Rename label column
    df.rename(columns={"label": "Label"}, inplace=True)

    # VERY IMPORTANT: rename benign label
    df["Label"] = df["Label"].replace("BenignTraffic", "BENIGN")

    df.replace([float("inf"), float("-inf")], 0, inplace=True)
    df.fillna(0, inplace=True)

    print("Dataset shape:", df.shape)
    print("Label distribution:\n", df["Label"].value_counts())

    return df