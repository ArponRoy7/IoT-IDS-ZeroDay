from preprocessing.preprocess_cicids import (
    load_clean_cicids,
    split_features_labels,
    train_test_split_stratified,
    scale_features
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("Loading dataset...")
df = load_clean_cicids()

print("Splitting features and labels...")
X, y = split_features_labels(df)

print("Train-test split...")
X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

print("Scaling features...")
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

print("Training Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("Evaluating...")
y_pred = model.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
