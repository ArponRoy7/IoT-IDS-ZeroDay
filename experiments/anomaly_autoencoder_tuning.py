import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from preprocessing.preprocess_cicids import load_clean_cicids

# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# Load Data
# ==========================================================

df = load_clean_cicids()

benign_df = df[df["Label"] == "BENIGN"].sample(n=200000, random_state=42)
attack_df = df[df["Label"] != "BENIGN"]

X_train = benign_df.drop("Label", axis=1).values
X_test = pd.concat([benign_df, attack_df]).drop("Label", axis=1).values
y_test = (pd.concat([benign_df, attack_df])["Label"] != "BENIGN").astype(int).values

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

input_dim = X_train.shape[1]

# ==========================================================
# Early Stopping Function
# ==========================================================

def train_autoencoder(bottleneck_dim):

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, bottleneck_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck_dim, 64),
                nn.ReLU(),
                nn.Linear(64, input_dim)
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = AE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience = 5
    trigger = 0

    epochs = 50
    batch_size = 1024

    print(f"\nTraining AE with bottleneck = {bottleneck_dim}")

    for epoch in range(epochs):

        model.train()
        perm = torch.randperm(X_train.size(0))
        epoch_loss = 0

        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch = X_train[idx]

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        # Early stopping check
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            trigger = 0
        else:
            trigger += 1

        if trigger >= patience:
            print("Early stopping triggered.")
            break

    # Evaluate
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_test)
        mse = torch.mean((X_test - reconstructed)**2, dim=1)

    scores = mse.cpu().numpy()

    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    print(f"AUC (bottleneck {bottleneck_dim}): {roc_auc:.4f}")

    return roc_auc


# ==========================================================
# Bottleneck Grid Search
# ==========================================================

bottleneck_list = [16, 12, 8, 4]

results = {}

for b in bottleneck_list:
    auc_score = train_autoencoder(b)
    results[b] = auc_score

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

for k, v in results.items():
    print(f"Bottleneck {k}: AUC = {v:.4f}")

best_dim = max(results, key=results.get)

print("\nBest Bottleneck Dimension:", best_dim)
print("Best AUC:", results[best_dim])
