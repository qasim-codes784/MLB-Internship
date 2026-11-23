from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CHART_DIR = Path(__file__).resolve().parent / "charts"
CHART_DIR.mkdir(exist_ok=True)
SIGNATURE = "Written by: Syed M Qasim Bukhari"

data = load_breast_cancer()
X = data.data
y = (data.target == 0).astype(int)

print("PRACTICE 1: Binary classification (is the tumour malignant?)")
print("Samples:", X.shape[0], "| Features:", X.shape[1])
print("Malignant (1):", int(y.sum()), "| Benign (0):", int((y == 0).sum()))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)

cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()

print("\nPRACTICE 2: Confusion matrix (positive class = malignant)")
print(cm)
print(f"True Negatives  (benign, predicted benign):       {tn}")
print(f"False Positives (benign, predicted malignant):    {fp}")
print(f"False Negatives (malignant, predicted benign):    {fn}")
print(f"True Positives  (malignant, predicted malignant): {tp}")

print("\nPRACTICE 3: Metrics calculated by hand and with scikit-learn")
manual = {
    "Accuracy": (tp + tn) / (tp + tn + fp + fn),
    "Precision": tp / (tp + fp),
    "Recall": tp / (tp + fn),
}
manual["F1"] = 2 * manual["Precision"] * manual["Recall"] / (manual["Precision"] + manual["Recall"])
library = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1": f1_score(y_test, pred),
}
for name in manual:
    print(f"{name:<10} by hand: {manual[name]:.4f} | scikit-learn: {library[name]:.4f}")
    assert np.isclose(manual[name], library[name])

print("\nPRACTICE 4: Why accuracy alone can mislead")
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
print(f"Dummy model that always says 'benign': accuracy = {accuracy_score(y_test, baseline_pred):.4f}, "
      f"recall = {recall_score(y_test, baseline_pred):.4f}")
print(f"Logistic Regression:                   accuracy = {library['Accuracy']:.4f}, "
      f"recall = {library['Recall']:.4f}")

print("\nPRACTICE 5: Changing the decision threshold (precision vs recall)")
probabilities = model.predict_proba(X_test_s)[:, 1]
print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<10}{'Missed cancers (FN)':<22}")
for threshold in (0.2, 0.5, 0.8):
    t_pred = (probabilities >= threshold).astype(int)
    t_cm = confusion_matrix(y_test, t_pred)
    print(f"{threshold:<12}{precision_score(y_test, t_pred):<12.3f}"
          f"{recall_score(y_test, t_pred):<10.3f}{t_cm[1, 0]:<22}")

plt.figure(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Breast Cancer Practice")
plt.tight_layout()
plt.savefig(CHART_DIR / "confusion_matrix_practice.png", dpi=150)
plt.close()

print("\nChart saved -> charts/confusion_matrix_practice.png")
print("-----------------------\n" + SIGNATURE)
