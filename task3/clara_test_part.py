import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn_extra.cluster import CLARA
from sklearn.metrics import silhouette_score
from bayes_opt import BayesianOptimization
from scipy.spatial.distance import cdist
import joblib


print("="*80)
print("CLARA CLUSTERING AVEC BAYESIAN OPTIMIZATION (SILHOUETTE)")
print("="*80)

X = pd.read_csv(r"D:\S3\data_mining_unsupervised\X_kmeans.csv").values
position_data = pd.read_csv("position_data.csv")

try:
    y_true = pd.read_csv("y_labels.csv").values.ravel()
    has_labels = True
except:
    y_true = None
    has_labels = False

print(f"✅ X shape: {X.shape}")

def clara_silhouette(n_clusters, n_sampling):
    k = int(round(n_clusters))
    s = int(round(n_sampling))

    if k < 2:
        return -1

    try:
        model = CLARA(
            n_clusters=k,
            n_sampling=min(s, len(X)),
            n_sampling_iter=5,
            random_state=42
        )

        labels = model.fit_predict(X)

        sil = silhouette_score(X, labels)

        return sil

    except Exception:
        return -1

optimizer = BayesianOptimization(
    f=clara_silhouette,
    pbounds={
        "n_clusters": (2, 10),
        "n_sampling": (100, 800)
    },
    random_state=42
)

print("\n🔍 Bayesian Optimization started...")
optimizer.maximize(init_points=5, n_iter=25)

best_params = optimizer.max["params"]
best_k = int(round(best_params["n_clusters"]))
best_sampling = int(round(best_params["n_sampling"]))

print(f"\n🏆 Best K (Silhouette): {best_k}")
print(f"🏆 Best sampling size: {best_sampling}")


final_model = CLARA(
    n_clusters=best_k,
    n_sampling=min(best_sampling, len(X)),
    n_sampling_iter=10,
    random_state=42
)

labels = final_model.fit_predict(X)
medoids = final_model.cluster_centers_

sil = silhouette_score(X, labels)

print("\n📈 FINAL CLUSTERING METRICS")
print(f"Silhouette Score: {sil:.4f}")

plt.figure(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, best_k))

for c in range(best_k):
    mask = labels == c
    plt.scatter(
        position_data.loc[mask, "longitude"],
        position_data.loc[mask, "latitude"],
        s=10,
        alpha=0.7,
        color=colors[c],
        label=f"Cluster {c}"
    )


plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(f"CLARA Geographic Clusters (K={best_k})")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("clara_geographic_clusters.png", dpi=300)
plt.close()


if has_labels:
    print("\n🔥 FIRE DISTRIBUTION PER CLUSTER")
    for c in range(best_k):
        mask = labels == c
        fire = (y_true[mask] == 1).sum()
        nofire = (y_true[mask] == 0).sum()
        purity = max(fire, nofire) / (fire + nofire)

        print(f"Cluster {c}: Fire={fire:,}, NoFire={nofire:,}, Purity={purity:.2f}")


results_df = position_data.copy()
results_df["cluster"] = labels

if has_labels:
    results_df["is_fire"] = y_true

results_df.to_csv("clara_results.csv", index=False)
joblib.dump(final_model, "clara_model.pkl")

print("\n💾 Files saved:")
print(" - clara_results.csv")
print(" - clara_model.pkl")
print(" - clara_geographic_clusters.png")

