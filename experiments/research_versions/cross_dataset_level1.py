from preprocessing.preprocess_cicids import load_clean_cicids
import pandas as pd

cicids = load_clean_cicids()
bot = pd.read_csv("data/processed/bot_iot_clean.csv")

cicids.columns = cicids.columns.str.strip()
bot.columns = bot.columns.str.strip()

# 🔥 Align features
common_cols = list(set(cicids.columns) & set(bot.columns))
common_cols.remove("Label")

cicids = cicids[common_cols + ["Label"]]
bot = bot[common_cols + ["Label"]]