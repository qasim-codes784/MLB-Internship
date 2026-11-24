import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def make_pipeline():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000)),
        ]
    )


def score_model(model):
    pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
    }


baseline = make_pipeline()
baseline.fit(X_train, y_train)
baseline_scores = score_model(baseline)

param_grid = {
    "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "clf__solver": ["lbfgs", "liblinear"],
    "clf__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(make_pipeline(), param_grid, cv=cv, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)
grid_scores = score_model(grid.best_estimator_)

baseline_cv = cross_val_score(make_pipeline(), X_train, y_train, cv=cv, scoring="f1").mean()
print("Baseline CV F1-Score:", round(baseline_cv, 4))

print("\nGRIDSEARCHCV")
print("Combinations tried:", len(grid.cv_results_["params"]))
print("Best parameters   :", grid.best_params_)
print("Best CV F1-Score  :", round(grid.best_score_, 4))

random_search = RandomizedSearchCV(
    make_pipeline(),
    {
        "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "clf__solver": ["lbfgs", "liblinear"],
        "clf__class_weight": [None, "balanced"],
    },
    n_iter=10,
    cv=cv,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
)
random_search.fit(X_train, y_train)
random_scores = score_model(random_search.best_estimator_)

print("\nRANDOMIZEDSEARCHCV")
print("Combinations tried:", 10)
print("Best parameters   :", random_search.best_params_)
print("Best CV F1-Score  :", round(random_search.best_score_, 4))

comparison = pd.DataFrame(
    {
        "Baseline": baseline_scores,
        "GridSearchCV": grid_scores,
        "RandomizedSearchCV": random_scores,
    }
).round(4)

print("\nTEST SET COMPARISON")
print(comparison)

top = (
    pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    .sort_values("rank_test_score")
    .head(5)
)
print("\nTop 5 GridSearchCV combinations:")
print(top.to_string(index=False))

ax = comparison.T.plot(kind="bar", figsize=(9, 5), ylim=(0.9, 1.0), rot=0, colormap="tab10")
ax.set_title("Baseline vs Tuned Models - Test Set")
ax.set_ylabel("Score")
plt.tight_layout()
path = os.path.join(OUT_DIR, "tuning_comparison.png")
plt.savefig(path, dpi=150)
print("\nChart saved to:", path)
plt.show()
