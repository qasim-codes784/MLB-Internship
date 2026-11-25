import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
names = [str(n) for n in iris.target_names]

X_scaled = StandardScaler().fit_transform(X)

k_values = list(range(1, 11))
inertias = []
silhouettes = {}
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    if k > 1:
        silhouettes[k] = silhouette_score(X_scaled, labels)

print("Elbow Method (inertia) and Silhouette Score:")
for k in k_values:
    sil = f"{silhouettes[k]:.4f}" if k in silhouettes else "-"
    print(f"  K={k:<2} inertia={inertias[k - 1]:8.2f}  silhouette={sil}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(k_values, inertias, marker="o", color="#4C78A8")
axes[0].axvline(3, linestyle="--", color="#E45756", label="K = 3")
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_xticks(k_values)
axes[0].legend()
axes[1].plot(list(silhouettes.keys()), list(silhouettes.values()), marker="o", color="#54A24B")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of clusters (K)")
axes[1].set_ylabel("Score")
axes[1].set_xticks(list(silhouettes.keys()))
plt.tight_layout()
elbow_path = os.path.join(OUT_DIR, "elbow_method.png")
plt.savefig(elbow_path, dpi=150)

K = 3
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

print(f"\nK-Means with K={K}")
print("Cluster sizes:", pd.Series(clusters).value_counts().sort_index().to_dict())
print("Silhouette score:", round(silhouette_score(X_scaled, clusters), 4))
print("Adjusted Rand Index vs real species:", round(adjusted_rand_score(y, clusters), 4))

print("\nClusters vs real species:")
print(pd.crosstab(pd.Series([names[i] for i in y], name="Species"), pd.Series(clusters, name="Cluster")))

centers = pd.DataFrame(
    StandardScaler().fit(X).inverse_transform(kmeans.cluster_centers_),
    columns=iris.feature_names,
).round(2)
centers.index.name = "Cluster"
print("\nCluster centres (original units, cm):")
print(centers)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ["#4C78A8", "#F58518", "#54A24B"]
for i, n in enumerate(names):
    m = y == i
    axes[0].scatter(X.loc[m, iris.feature_names[2]], X.loc[m, iris.feature_names[3]], label=n, color=colors[i], edgecolor="white", s=60)
axes[0].set_title("Actual Species")
for c in range(K):
    m = clusters == c
    axes[1].scatter(X.loc[m, iris.feature_names[2]], X.loc[m, iris.feature_names[3]], label=f"Cluster {c}", color=colors[c], edgecolor="white", s=60)
axes[1].scatter(centers[iris.feature_names[2]], centers[iris.feature_names[3]], marker="X", s=250, color="black", label="Centroids")
axes[1].set_title(f"K-Means Clusters (K={K})")
for ax in axes:
    ax.set_xlabel(iris.feature_names[2])
    ax.set_ylabel(iris.feature_names[3])
    ax.legend()
plt.tight_layout()
cluster_path = os.path.join(OUT_DIR, "kmeans_clusters.png")
plt.savefig(cluster_path, dpi=150)

print("\nCharts saved to:")
print(" ", elbow_path)
print(" ", cluster_path)
plt.show()
