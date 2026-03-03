import pandas as pd

df = pd.read_csv(
"/scratch/sm/Desktop/Arpon/IoT-IDS-ZeroDay/data/processed/ciciot2023_combined.csv"
)

df.rename(columns={"label":"Label"}, inplace=True)

df["Label"] = df["Label"].astype(str).str.upper()

print(df["Label"].value_counts())