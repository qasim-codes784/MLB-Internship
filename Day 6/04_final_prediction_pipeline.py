import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
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

print("=" * 60)
print("BREAST CANCER PREDICTION SYSTEM")
print("=" * 60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
names = [str(n) for n in data.target_names]

print("\nDataset shape:", X.shape)
print("Classes:", names)
print("Missing values:", int(X.isnull().sum().sum()))
for i, n in enumerate(names):
    print(f"  {n}: {int((y == i).sum())}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


def make_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000)),
        ]
    )


def evaluate(model):
    pred = model.predict(X_test)
    return pred, {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
    }


baseline = make_pipeline()
baseline.fit(X_train, y_train)
base_pred, base_scores = evaluate(baseline)

print("\n" + "=" * 60)
print("BEFORE TUNING (BASELINE)")
print("=" * 60)
for k, v in base_scores.items():
    print(f"{k:<10}: {v:.4f}")

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "clf__solver": ["lbfgs", "liblinear"],
    "clf__class_weight": [None, "balanced"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(make_pipeline(), param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

print("\n" + "=" * 60)
print("GRIDSEARCHCV")
print("=" * 60)
print("Combinations tried:", len(grid.cv_results_["params"]))
print("Best parameters   :")
for k, v in grid.best_params_.items():
    print(f"  {k.replace('clf__', '')}: {v}")
baseline_cv = cross_val_score(make_pipeline(), X_train, y_train, cv=cv, scoring="f1").mean()
print("Baseline CV F1    :", round(baseline_cv, 4))
print("Tuned CV F1       :", round(grid.best_score_, 4))

tuned = grid.best_estimator_
tuned_pred, tuned_scores = evaluate(tuned)

print("\n" + "=" * 60)
print("AFTER TUNING")
print("=" * 60)
for k, v in tuned_scores.items():
    print(f"{k:<10}: {v:.4f}")

comparison = pd.DataFrame({"Baseline": base_scores, "Tuned": tuned_scores})
comparison["Change"] = comparison["Tuned"] - comparison["Baseline"]
print("\n" + "=" * 60)
print("COMPARISON (TEST SET)")
print("=" * 60)
print(comparison.round(4))

print("\nTuned model classification report:")
print(classification_report(y_test, tuned_pred, target_names=names))

base_cm = confusion_matrix(y_test, base_pred)
tuned_cm = confusion_matrix(y_test, tuned_pred)
print("Baseline confusion matrix:")
print(base_cm)
print("Tuned confusion matrix:")
print(tuned_cm)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
ConfusionMatrixDisplay(base_cm, display_labels=names).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Baseline")
ConfusionMatrixDisplay(tuned_cm, display_labels=names).plot(ax=axes[1], cmap="Greens", colorbar=False)
axes[1].set_title("Tuned (GridSearchCV)")
fig.suptitle("Breast Cancer - Confusion Matrix Before vs After Tuning")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)

ax = comparison[["Baseline", "Tuned"]].plot(kind="bar", figsize=(8, 5), ylim=(0.9, 1.0), rot=0, color=["#4C78A8", "#54A24B"])
ax.set_title("Baseline vs Tuned - Test Metrics")
ax.set_ylabel("Score")
plt.tight_layout()
cmp_path = os.path.join(OUT_DIR, "baseline_vs_tuned.png")
plt.savefig(cmp_path, dpi=150)

print("\n" + "=" * 60)
print("PREDICT A NEW PATIENT (first test sample)")
print("=" * 60)
sample = X_test.iloc[[0]]
proba = tuned.predict_proba(sample)[0]
print("Actual    :", names[int(y_test.iloc[0])])
print("Predicted :", names[int(tuned.predict(sample)[0])])
for n, p in zip(names, proba):
    print(f"  {n}: {p:.2%}")

print("\nImages saved to:")
print(" ", cm_path)
print(" ", cmp_path)
plt.show()
