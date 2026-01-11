import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)

# -----------------------------
# Load data
# -----------------------------
X = pd.read_csv(r"D:\S3\data_mining_unsupervised\X_kmeans.csv").values
y_fire = pd.read_csv(r"D:\S3\data_mining_unsupervised\y_labels.csv")["is_fire"].values

# -----------------------------
# Load trained model
# -----------------------------
import joblib
model = joblib.load("clara_model.pkl")

# -----------------------------
# Get cluster labels
# -----------------------------
labels_cluster = model.labels_  # DBSCAN / CLARA

# -----------------------------
# Remove noise (DBSCAN)
# -----------------------------
mask = labels_cluster != -1
X_clean = X[mask]
labels_clean = labels_cluster[mask]
y_clean = y_fire[mask]

# -----------------------------
# Internal metrics (UNSUPERVISED)
# -----------------------------
sil = silhouette_score(X_clean, labels_clean)
db  = davies_bouldin_score(X_clean, labels_clean)
ch  = calinski_harabasz_score(X_clean, labels_clean)

# -----------------------------
# External metrics (A POSTERIORI)
# -----------------------------
ari = adjusted_rand_score(y_clean, labels_clean)
nmi = normalized_mutual_info_score(y_clean, labels_clean)

# -----------------------------
# Noise handling
# -----------------------------
noise_ratio = np.mean(labels_cluster == -1)

# -----------------------------
# Print results
# -----------------------------
print("📊 CLUSTERING EVALUATION")
print("-" * 40)
print(f"Silhouette Score       : {sil:.3f}")
print(f"Davies-Bouldin Index   : {db:.3f}")
print(f"Calinski-Harabasz     : {ch:.2f}")
print(f"ARI                  : {ari:.3f}")
print(f"NMI                  : {nmi:.3f}")
print(f"Noise Ratio           : {noise_ratio*100:.1f}%")
