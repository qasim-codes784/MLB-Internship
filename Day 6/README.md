# Day 6 – Model Evaluation and Hyperparameter Tuning

Breast Cancer Wisconsin Diagnostic dataset (built into scikit-learn, 569 samples, 30 features, 2 classes).

## Files

| File | Purpose |
|---|---|
| `01_dataset_exploration.py` | Loads the data into a DataFrame, runs `head()`, `info()`, `describe()`, checks class distribution |
| `02_baseline_model.py` | Baseline Logistic Regression with metrics, confusion matrix and 5-fold cross validation |
| `03_hyperparameter_tuning.py` | GridSearchCV and RandomizedSearchCV compared with the baseline |
| `04_final_prediction_pipeline.py` | Mini project: full pipeline, before and after tuning, confusion matrix heatmaps |
| `confusion_matrix.png` | Confusion matrices, baseline vs tuned |
| `baseline_vs_tuned.png` | Metric comparison chart |
| `class_distribution.png`, `baseline_confusion_matrix.png`, `tuning_comparison.png` | Supporting charts |

## How to Run

```
pip install scikit-learn pandas matplotlib
python 01_dataset_exploration.py
python 02_baseline_model.py
python 03_hyperparameter_tuning.py
python 04_final_prediction_pipeline.py
```

## What I Learned About Model Evaluation

- A single train/test split can be misleading, especially on small data. Cross validation (5-fold) gives a more stable estimate by testing on several different splits.
- Train vs test scores show overfitting (high train, lower test) and underfitting (low on both). The baseline scored 0.989 on training and 0.9825 on testing, so it fits well.
- The metric must match the problem. For cancer detection, recall (finding every real case) matters most, with precision and F1 as support.
- Scaling is done inside a `Pipeline`, so the scaler is fitted only on training folds. This prevents data leakage during cross validation.

## What is Hyperparameter Tuning and Why It Matters

Hyperparameters are settings chosen before training (for Logistic Regression: `C`, `solver`, `class_weight`), unlike parameters such as the coefficients, which the model learns. Tuning tries different settings and picks the best one using cross validation, so the choice does not depend on the test set.

- **GridSearchCV** tries every combination (24 here).
- **RandomizedSearchCV** tries a random sample (10 here), which is faster for large search spaces.

## Best Parameters Found by GridSearchCV

| Parameter | Value |
|---|---|
| `C` | 0.1 |
| `class_weight` | None |
| `solver` | lbfgs |

## Baseline vs Tuned

| Metric | Baseline | Tuned | Change |
|---|---|---|---|
| Cross validation F1 (training data) | 0.9825 | 0.9861 | +0.0036 |
| Accuracy (test) | 0.9825 | 0.9737 | -0.0088 |
| Precision (test) | 0.9861 | 0.9726 | -0.0135 |
| Recall (test) | 0.9861 | 0.9861 | 0.0000 |
| F1-Score (test) | 0.9861 | 0.9793 | -0.0068 |

Confusion matrices on the 114 test samples:

| | Baseline | Tuned |
|---|---|---|
| Malignant predicted correctly | 41 of 42 | 40 of 42 |
| Benign predicted correctly | 71 of 72 | 71 of 72 |

## Key Observations

- Tuning improved the cross validation F1 slightly (0.9825 to 0.9861), which is the score the search optimises.
- On the test set the tuned model did not improve. It made one more mistake (a malignant case predicted as benign), which lowers accuracy and precision by about one percent. Recall for benign stayed the same.
- The difference is one sample out of 114, so it is within the normal noise of a small test set. The baseline was already very strong (0.9825), leaving little room to improve.
- Tuning does not guarantee better test results. It helps most when the default settings are a poor fit. Judging a model on cross validation and a single test split together gives a more honest picture than either alone.
- For a medical problem, the missed malignant case is the important error. A next step would be to tune with `scoring="recall"` to reduce missed cancers.
