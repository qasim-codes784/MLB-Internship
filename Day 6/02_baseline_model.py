import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
names = [str(n) for n in data.target_names]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

baseline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)
baseline.fit(X_train, y_train)

y_train_pred = baseline.predict(X_train)
y_pred = baseline.predict(X_test)

print("\nBASELINE LOGISTIC REGRESSION")
print("Training accuracy:", round(accuracy_score(y_train, y_train_pred), 4))
print("Testing accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision        :", round(precision_score(y_test, y_pred), 4))
print("Recall           :", round(recall_score(y_test, y_pred), 4))
print("F1-Score         :", round(f1_score(y_test, y_pred), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="f1")
print("\n5-Fold Cross Validation (F1) on training data:")
print("Scores:", [round(float(s), 4) for s in cv_scores])
print("Mean  :", round(cv_scores.mean(), 4))
print("Std   :", round(cv_scores.std(), 4))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Baseline Model - Confusion Matrix")
plt.tight_layout()
path = os.path.join(OUT_DIR, "baseline_confusion_matrix.png")
plt.savefig(path, dpi=150)
print("\nChart saved to:", path)
plt.show()
