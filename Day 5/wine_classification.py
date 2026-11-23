import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts", "wine")
os.makedirs(CHART_DIR, exist_ok=True)


def save_chart(name):
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, name), dpi=150)


data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
names = list(data.target_names)
X_plot = X

print("Shape of features:", X.shape)
print("Feature names:", list(data.feature_names))
print("Target classes:", names)
print("\nFirst 5 rows:")
print(X.head())
print("\nSummary statistics:")
print(X.describe())
print("\nClass distribution:")
print(y.value_counts().sort_index())

plt.figure(figsize=(6, 4))
counts = y.value_counts().sort_index()
bars = plt.bar(names, counts.values, color=["#4C78A8", "#F58518", "#54A24B"][: len(names)])
for bar, value in zip(bars, counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
plt.title("Class Distribution - Wine")
plt.xlabel("Class")
plt.ylabel("Samples")
save_chart("01_class_distribution.png")

X_plot.hist(bins=20, figsize=(12, 8), color="#4C78A8", edgecolor="white")
plt.suptitle("Feature Histograms - Wine")
save_chart("02_feature_histograms.png")

plt.figure(figsize=(9, 7))
sns.heatmap(X_plot.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Feature Correlation Heatmap - Wine")
save_chart("03_correlation_heatmap.png")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

scaler = StandardScaler()
X_train_m = scaler.fit_transform(X_train)
X_test_m = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_m, y_train)

y_train_pred = model.predict(X_train_m)
y_pred = model.predict(X_test_m)

AVG = "macro"
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average=AVG)
recall = recall_score(y_test, y_pred, average=AVG)
f1 = f1_score(y_test, y_pred, average=AVG)

print("\nLogistic Regression")
print("Training accuracy:", round(train_acc, 4))
print("Testing accuracy :", round(test_acc, 4))
print("Precision        :", round(precision, 4))
print("Recall           :", round(recall, 4))
print("F1-Score         :", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
tree_train_acc = accuracy_score(y_train, tree.predict(X_train))
tree_pred = tree.predict(X_test)
tree_test_acc = accuracy_score(y_test, tree_pred)

print("\nDecision Tree")
print("Training accuracy:", round(tree_train_acc, 4))
print("Testing accuracy :", round(tree_test_acc, 4))
print("F1-Score         :", round(f1_score(y_test, tree_pred, average=AVG), 4))

plt.figure(figsize=(7, 5))
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
metric_values = [test_acc, precision, recall, f1]
bars = plt.bar(metric_names, metric_values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
for bar, value in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
plt.ylim(0, 1.1)
plt.title("Logistic Regression Metrics - Wine")
plt.ylabel("Score")
save_chart("05_evaluation_metrics.png")

plt.figure(figsize=(7, 5))
positions = range(2)
width = 0.35
train_values = [train_acc, tree_train_acc]
test_values = [test_acc, tree_test_acc]
bars_train = plt.bar([p - width / 2 for p in positions], train_values, width, label="Training", color="#4C78A8")
bars_test = plt.bar([p + width / 2 for p in positions], test_values, width, label="Testing", color="#F58518")
for bar in list(bars_train) + list(bars_test):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.3f}", ha="center", va="bottom")
plt.xticks(list(positions), ["Logistic Regression", "Decision Tree"])
plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy - Wine")
plt.legend()
save_chart("06_train_vs_test_accuracy.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Logistic Regression")
ConfusionMatrixDisplay(confusion_matrix(y_test, tree_pred), display_labels=names).plot(ax=axes[1], cmap="Greens", colorbar=False)
axes[1].set_title("Decision Tree")
fig.suptitle("Confusion Matrices - Wine")
save_chart("07_confusion_matrices.png")

small_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
small_tree.fit(X_train, y_train)
plt.figure(figsize=(18, 9))
plot_tree(small_tree, feature_names=list(X.columns), class_names=names, filled=True, fontsize=8)
plt.title("Decision Tree (max_depth=3) - Wine")
save_chart("08_decision_tree.png")

depths = list(range(1, 11))
depth_train = []
depth_test = []
for depth in depths:
    t = DecisionTreeClassifier(max_depth=depth, random_state=42)
    t.fit(X_train, y_train)
    depth_train.append(accuracy_score(y_train, t.predict(X_train)))
    depth_test.append(accuracy_score(y_test, t.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(depths, depth_train, marker="o", label="Training", color="#4C78A8")
plt.plot(depths, depth_test, marker="o", label="Testing", color="#F58518")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Underfitting vs Overfitting - Wine")
plt.xticks(depths)
plt.legend()
plt.grid(alpha=0.3)
save_chart("09_underfitting_vs_overfitting.png")

print("\nCharts saved in:", CHART_DIR)
plt.show()
