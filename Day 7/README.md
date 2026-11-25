# Day 7 – Clustering (K-Means) and PCA

Iris dataset (built into scikit-learn): 150 flowers, 4 features, 3 species. The species labels are used only to check the results, never to train K-Means or PCA.

## Files

| File | Purpose |
|---|---|
| `01_dataset_exploration.py` | DataFrame, `head()`, `info()`, `describe()`, class counts, pair plot, correlation heatmap |
| `02_kmeans_clustering.py` | Elbow Method, Silhouette Score, K-Means with K=3, cluster scatter plot |
| `03_pca.py` | PCA to 2 components, explained variance, PCA plot colored by species and by cluster |
| `04_iris_clustering_visualization.py` | Mini project: full pipeline with the three required visualizations and observations |
| `*.png` | Generated graphs |

## How to Run

```
pip install scikit-learn pandas matplotlib seaborn
python 01_dataset_exploration.py
python 02_kmeans_clustering.py
python 03_pca.py
python 04_iris_clustering_visualization.py
```

## What is Clustering?

Clustering is an unsupervised learning technique that groups similar data points together without using labels. K-Means picks K centres, assigns each point to its nearest centre, moves each centre to the mean of its points, and repeats until the groups stop changing. It is used for customer segmentation, recommendation systems and anomaly detection.

## What is PCA?

Principal Component Analysis reduces the number of features while keeping as much of the variation (information) as possible. It creates new features called principal components, ordered by how much variance they explain. Here it turns the 4 Iris features into 2, which can be plotted. Features are scaled first, because both K-Means and PCA are sensitive to feature size.

## How I Determined the Best Value of K

| K | Inertia | Silhouette |
|---|---|---|
| 1 | 600.00 | - |
| 2 | 222.36 | 0.5818 |
| 3 | 139.82 | 0.4599 |
| 4 | 114.09 | 0.3869 |
| 5 | 90.93 | 0.3459 |

- The Elbow Method plots inertia against K. The curve bends sharply at K=2 and K=3, and flattens after that.
- The Silhouette Score is highest at K=2 (0.58) and lower at K=3 (0.46).
- I chose **K=3**. The elbow shows a clear bend there, and Iris has 3 known species. The silhouette favours K=2 because versicolor and virginica overlap, so the data looks like two blobs: setosa, and everything else.

## Results

| Measure | Value |
|---|---|
| Clusters formed | 3 (sizes 53, 50, 47) |
| Silhouette score (K=3) | 0.4599 |
| Adjusted Rand Index vs real species | 0.6201 |
| Variance kept by 2 PCA components | 95.81% (PC1 72.96%, PC2 22.85%) |

Clusters vs real species:

| | Cluster 0 | Cluster 1 | Cluster 2 |
|---|---|---|---|
| setosa | 0 | 50 | 0 |
| versicolor | 39 | 0 | 11 |
| virginica | 14 | 0 | 36 |

Cluster numbers are arbitrary labels. Cluster 1 is the setosa group in this run.

## Insights from the Visualizations

- **How many clusters formed?** Three. One is clean and the other two overlap.
- **Did the clusters match the species?** Setosa was separated perfectly (all 50 in one cluster). Versicolor and virginica were partly mixed, with 25 of 150 flowers in a cluster that does not match their species. The Adjusted Rand Index of 0.62 shows a good but not perfect match. K-Means works only from distances, so it cannot separate groups that overlap in the measurements.
- **How did PCA help?** Two components keep 95.8% of the variance, so 4-dimensional data is shown in 2D with almost no loss. In the PCA plot setosa is far from the rest, and the versicolor and virginica overlap is easy to see. PC1 alone carries about 73% of the variance, mostly from petal length and petal width.
- **Supervised vs unsupervised:** Logistic Regression on Iris (Day 5) reached about 97% accuracy using the labels. K-Means, which never sees labels, gets roughly 83% of flowers in the matching cluster (125 of 150). The labels make a real difference when groups overlap.
