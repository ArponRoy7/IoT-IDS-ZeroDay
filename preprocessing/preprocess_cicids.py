import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_clean_cicids():
    """
    Load frozen cleaned CICIDS2017 dataset.
    Path is resolved dynamically relative to project root.
    """

    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct dataset path
    data_path = os.path.join(
        BASE_DIR,
        "data",
        "processed",
        "cicids2017_clean.csv"
    )

    print("Loading from:", data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    return df


def split_features_labels(df):
    X = df.drop("Label", axis=1)
    y = df["Label"]
    return X, y


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
