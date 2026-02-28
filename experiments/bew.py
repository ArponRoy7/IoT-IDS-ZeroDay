import pandas as pd

df = pd.read_csv("data/processed/cicids2017_clean.csv")

print("\nShape:", df.shape)

print("\nFeature Count:", len(df.columns))

print("\nFeature Names:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

print("\nAttack Distribution:")
print(df.iloc[:,-1].value_counts())