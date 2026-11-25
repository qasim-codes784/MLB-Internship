import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [str(iris.target_names[i]) for i in iris.target]

print("Shape:", df.shape)
print("Features:", list(iris.feature_names))
print("Species:", [str(n) for n in iris.target_names])

print("\nhead():")
print(df.head())

print("\ninfo():")
df.info()

print("\ndescribe():")
print(df.describe().T)

print("\nMissing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

print("\nSamples per species:")
print(df["species"].value_counts())

print("\nMean of each feature per species:")
print(df.groupby("species").mean().round(2))

pair = sns.pairplot(df, hue="species", corner=False)
pair.figure.suptitle("Iris - Pair Plot", y=1.02)
pair_path = os.path.join(OUT_DIR, "pair_plot.png")
pair.savefig(pair_path, dpi=150)

plt.figure(figsize=(7, 6))
sns.heatmap(df[iris.feature_names].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Iris - Feature Correlation")
plt.tight_layout()
corr_path = os.path.join(OUT_DIR, "correlation_heatmap.png")
plt.savefig(corr_path, dpi=150)

print("\nCharts saved to:")
print(" ", pair_path)
print(" ", corr_path)
plt.show()
