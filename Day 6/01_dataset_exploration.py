import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print("Shape:", df.shape)
print("Target classes:", {int(i): str(n) for i, n in enumerate(data.target_names)})

print("\nhead():")
print(df.head())

print("\ninfo():")
df.info()

print("\ndescribe():")
print(df.describe().T)

print("\nMissing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

counts = df["target"].value_counts().sort_index()
print("\nTarget class distribution:")
for i, n in enumerate(data.target_names):
    print(f"  {i} ({n}): {counts[i]} ({counts[i] / len(df):.1%})")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar([str(n) for n in data.target_names], counts.values, color=["#E45756", "#54A24B"])
for bar, value in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
ax.set_title("Breast Cancer - Target Class Distribution")
ax.set_ylabel("Samples")
plt.tight_layout()
path = os.path.join(OUT_DIR, "class_distribution.png")
plt.savefig(path, dpi=150)
print("\nChart saved to:", path)
plt.show()
