import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
names = [str(n) for n in iris.target_names]
colors = ["#4C78A8", "#F58518", "#54A24B"]

X_scaled = StandardScaler().fit_transform(X)

pca_full = PCA()
pca_full.fit(X_scaled)
ratios = pca_full.explained_variance_ratio_

print("Explained variance by each component:")
for i, r in enumerate(ratios, start=1):
    print(f"  PC{i}: {r:.4f} ({r:.2%})  cumulative: {ratios[:i].sum():.2%}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\nShape before PCA:", X_scaled.shape)
print("Shape after PCA :", X_pca.shape)
print("Variance kept by 2 components:", f"{pca.explained_variance_ratio_.sum():.2%}")

loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=iris.feature_names).round(3)
print("\nFeature loadings:")
print(loadings)

clusters = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, n in enumerate(names):
    m = y == i
    axes[0].scatter(X_pca[m, 0], X_pca[m, 1], label=n, color=colors[i], edgecolor="white", s=60)
axes[0].set_title("PCA - Colored by Actual Species")
for c in range(3):
    m = clusters == c
    axes[1].scatter(X_pca[m, 0], X_pca[m, 1], label=f"Cluster {c}", color=colors[c], edgecolor="white", s=60)
axes[1].set_title("PCA - Colored by K-Means Clusters")
for ax in axes:
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend()
plt.tight_layout()
pca_path = os.path.join(OUT_DIR, "pca_species_vs_clusters.png")
plt.savefig(pca_path, dpi=150)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(range(1, 5), ratios, color="#4C78A8", label="Individual")
ax.plot(range(1, 5), np.cumsum(ratios), marker="o", color="#E45756", label="Cumulative")
ax.set_xticks(range(1, 5))
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")
ax.set_title("PCA - Explained Variance")
ax.legend()
plt.tight_layout()
var_path = os.path.join(OUT_DIR, "pca_explained_variance.png")
plt.savefig(var_path, dpi=150)

print("\nCharts saved to:")
print(" ", pca_path)
print(" ", var_path)
plt.show()
