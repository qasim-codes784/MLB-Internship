import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("IRIS FLOWER CLUSTERING AND VISUALIZATION")
print("=" * 60)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [str(iris.target_names[i]) for i in iris.target]
features = list(iris.feature_names)
names = [str(n) for n in iris.target_names]
colors = ["#4C78A8", "#F58518", "#54A24B"]

print("\nDataset shape:", df.shape)
print("Missing values:", int(df.isnull().sum().sum()))
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary statistics:")
print(df[features].describe().round(2))
print("\nSamples per species:")
print(df["species"].value_counts())

X_scaled = StandardScaler().fit_transform(df[features])

print("\n" + "=" * 60)
print("ELBOW METHOD")
print("=" * 60)
k_values = list(range(1, 11))
inertias = []
silhouettes = {}
for k in k_values:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    if k > 1:
        silhouettes[k] = silhouette_score(X_scaled, labels)
    print(f"K={k:<2} inertia={inertias[-1]:8.2f}")

best_silhouette_k = max(silhouettes, key=silhouettes.get)
K = 3
print("\nBest silhouette score at K =", best_silhouette_k, f"({silhouettes[best_silhouette_k]:.4f})")
print("Chosen K =", K, "(elbow bend, and Iris has 3 known species)")

kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

print("\n" + "=" * 60)
print("K-MEANS RESULTS")
print("=" * 60)
print("Clusters formed:", df["cluster"].nunique())
print("Silhouette score:", round(silhouette_score(X_scaled, df["cluster"]), 4))
ari = adjusted_rand_score(iris.target, df["cluster"])
print("Adjusted Rand Index vs species:", round(ari, 4))
print("\nClusters vs real species:")
crosstab = pd.crosstab(df["species"], df["cluster"])
print(crosstab)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
kept = pca.explained_variance_ratio_.sum()

print("\n" + "=" * 60)
print("PCA")
print("=" * 60)
print("Dimensions: 4 ->", X_pca.shape[1])
print(f"PC1 variance: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2 variance: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Total variance kept: {kept:.2%}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
for i, n in enumerate(names):
    m = df["species"] == n
    axes[0].scatter(df.loc[m, features[2]], df.loc[m, features[3]], label=n, color=colors[i], edgecolor="white", s=60)
axes[0].set_title("1. Original Data (Petal Length vs Petal Width)")
axes[0].set_xlabel(features[2])
axes[0].set_ylabel(features[3])
axes[0].legend()

for c in range(K):
    m = df["cluster"] == c
    axes[1].scatter(df.loc[m, features[2]], df.loc[m, features[3]], label=f"Cluster {c}", color=colors[c], edgecolor="white", s=60)
centers = pd.DataFrame(
    StandardScaler().fit(df[features]).inverse_transform(kmeans.cluster_centers_),
    columns=features,
)
axes[1].scatter(centers[features[2]], centers[features[3]], marker="X", s=250, color="black", label="Centroids")
axes[1].set_title(f"2. K-Means Clusters (K={K})")
axes[1].set_xlabel(features[2])
axes[1].set_ylabel(features[3])
axes[1].legend()

for c in range(K):
    m = df["cluster"] == c
    axes[2].scatter(df.loc[m, "PC1"], df.loc[m, "PC2"], label=f"Cluster {c}", color=colors[c], edgecolor="white", s=60)
axes[2].set_title("3. PCA-Transformed Data (2D)")
axes[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axes[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axes[2].legend()
plt.tight_layout()
main_path = os.path.join(OUT_DIR, "iris_clustering_project.png")
plt.savefig(main_path, dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(k_values, inertias, marker="o", color="#4C78A8")
axes[0].axvline(K, linestyle="--", color="#E45756", label=f"K = {K}")
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
elbow_path = os.path.join(OUT_DIR, "project_elbow_method.png")
plt.savefig(elbow_path, dpi=150)

mixed = int(crosstab.values.sum() - crosstab.max(axis=1).sum())
print("\n" + "=" * 60)
print("OBSERVATIONS")
print("=" * 60)
print(f"1. Clusters formed: {K}.")
print(f"2. Setosa forms its own clean cluster. Versicolor and virginica overlap,")
print(f"   so {mixed} flowers landed in a cluster that does not match their species.")
print(f"3. PCA keeps {kept:.1%} of the variance in 2 dimensions, so the 4D data")
print("   can be plotted in 2D with almost no loss, and the groups are easy to see.")

print("\nCharts saved to:")
print(" ", main_path)
print(" ", elbow_path)
plt.show()
