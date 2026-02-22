import os
import pandas as pd

# Adjust this path if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_dir = os.path.join(BASE_DIR, "data", "raw", "cicids")

print("Reading one sample file...\n")

files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

if len(files) == 0:
    raise FileNotFoundError("No CSV files found in cicids raw directory.")

sample_file = os.path.join(raw_dir, files[0])
print("Sample file:", files[0])

df = pd.read_csv(sample_file, nrows=5)

print("\n================ COLUMN NAMES ================\n")
for col in df.columns:
    print(col)

print("\n================ CHECK IMPORTANT FIELDS ================\n")

check_cols = [
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
    "Flow Duration",
    "Label"
]

for col in check_cols:
    print(f"{col} present? ->", col in df.columns)

print("\n================ DATA SAMPLE ================\n")
print(df.head())
