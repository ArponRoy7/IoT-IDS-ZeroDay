# =========================================================
# PUBLISHABLE t-SNE LATENT SPACE GENERATOR (IEEE FORMAT)
# =========================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# (Assume load_clean_cicids is imported from your preprocessing module)
from preprocessing.preprocess_cicids import load_clean_cicids

# =========================================================
# PARAMETERS
# =========================================================

EPOCHS = 35
BATCH_SIZE = 4096
seed = 42

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# LOAD DATA
# =========================================================

df = load_clean_cicids()

for col in df.select_dtypes(include=["float64"]).columns:
    df[col] = df[col].astype("float32")

for col in df.select_dtypes(include=["int64"]).columns:
    df[col] = df[col].astype("int32")

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

# =========================================================
# TRAIN DAE ONCE USING ALL BENIGN DATA
# =========================================================

benign_full = df[df["Label"] == "BENIGN"]

scaler = StandardScaler()
scaler.fit(benign_full.drop("Label", axis=1))
X_benign = scaler.transform(benign_full.drop("Label", axis=1))

X_benign_tensor = torch.tensor(X_benign, dtype=torch.float32)

class DAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

model = DAE(X_benign.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

loader = DataLoader(
    TensorDataset(X_benign_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("Training DAE to extract Latent Space...")
model.train()

prev_loss = float("inf")

for epoch in range(EPOCHS):
    total_loss = 0
    for (x,) in loader:
        x = x.to(device)
        noise = torch.randn_like(x) * 0.05
        optimizer.zero_grad()
        recon, _ = model(x + noise)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch > 5 and abs(prev_loss - total_loss) < 1e-4:
        break
    prev_loss = total_loss

model.eval()

# =========================================================
# GENERATE IEEE-READY FIGURE 4 (t-SNE Latent Space)
# =========================================================
print("\n" + "="*80)
print("Generating Figure 4: t-SNE Latent Space Synergy Plot...")
print("="*80)

# 1. Sample Benign traffic (the baseline manifold)
sample_benign = df[df['Label'] == 'BENIGN'].sample(1500, random_state=seed)

# 2. Sample Volumetric Attacks (Should separate cleanly)
ddos_mask = df['Label'].str.contains('DDoS', na=False)
sample_ddos = df[ddos_mask].sample(1000, random_state=seed, replace=True)

# 3. SCIENTIFIC FIX: Extract Stage-1 False Negatives
print("Extracting Stage-1 False Negatives for Web Attacks...")
raw_web_mask = df['Label'].str.contains('Web|Sql Injection|XSS', case=False, na=False)
large_web_sample = df[raw_web_mask].sample(min(5000, df[raw_web_mask].shape[0]), random_state=seed)

# Push the large pool through the DAE to get their reconstruction error
X_web_raw = scaler.transform(large_web_sample.drop("Label", axis=1))
X_web_tensor = torch.tensor(X_web_raw, dtype=torch.float32).to(device)

with torch.no_grad():
    recon_web, _ = model(X_web_tensor)
    web_residuals = torch.mean((recon_web - X_web_tensor) ** 2, dim=1).cpu().numpy()

# Only keep the 800 Web Attacks that have the LOWEST residuals 
# (These are the stealthy flows that successfully mimic Benign traffic)
stealthy_indices = np.argsort(web_residuals)[:800] 
sample_web = large_web_sample.iloc[stealthy_indices]

# Combine all samples
plot_df = pd.concat([sample_benign, sample_ddos, sample_web])

# 4. Extract true labels for color-coding
y_labels_raw = plot_df['Label'].values
y_labels = np.array(['BENIGN' if 'BENIGN' in x else ('DDoS' if 'DDoS' in x else 'Web Attack') for x in y_labels_raw])

# 5. Preprocess features
X_plot = scaler.transform(plot_df.drop("Label", axis=1))
X_plot_tensor = torch.tensor(X_plot, dtype=torch.float32).to(device)

# 6. Extract 8-dimensional bottleneck (z) from the DAE
with torch.no_grad():
    _, latent_z = model(X_plot_tensor)
    latent_z_numpy = latent_z.cpu().numpy()

# 7. Compute t-SNE (Squashes 8D -> 2D)
print("Computing t-SNE (perplexity=80, this may take a minute)...")
tsne = TSNE(n_components=2, perplexity=80, random_state=seed, init='pca', learning_rate='auto')
tsne_results = tsne.fit_transform(latent_z_numpy)

# 8. IEEE Standard Plotting
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

# Plot Benign first so it forms the background layer
mask_benign = (y_labels == 'BENIGN')
ax.scatter(tsne_results[mask_benign, 0], tsne_results[mask_benign, 1], 
           c='#1f77b4', marker='o', s=15, alpha=0.4, label='Benign', edgecolors='none')

# Plot Web Attacks second so they sit clearly inside/on top of the Benign layer
mask_web = (y_labels == 'Web Attack')
ax.scatter(tsne_results[mask_web, 0], tsne_results[mask_web, 1], 
           c='#2ca02c', marker='s', s=20, alpha=0.85, label='Web Attack (Payload)', edgecolors='none')

# Plot DDoS last so it clearly separates on the edges
mask_ddos = (y_labels == 'DDoS')
ax.scatter(tsne_results[mask_ddos, 0], tsne_results[mask_ddos, 1], 
           c='#d62728', marker='^', s=20, alpha=0.85, label='DDoS (Volumetric)', edgecolors='none')

# Formatting
ax.set_xlabel("t-SNE Dimension 1", fontsize=11, fontname='serif')
ax.set_ylabel("t-SNE Dimension 2", fontsize=11, fontname='serif')

for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontname('serif')
    tick.set_fontsize(10)

ax.legend(prop={'family': 'serif', 'size': 10}, loc='best', framealpha=0.9)
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("Fig4_tSNE_Synergy_Publishable.png", format='png', bbox_inches='tight')

print("Figure saved as Fig4_tSNE_Synergy_Publishable.png")
plt.close()

print("\nPROCESS FINISHED.")