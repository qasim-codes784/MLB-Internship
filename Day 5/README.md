# Day 5 – Model Evaluation and Classification

## Contents

| File | Dataset | Task |
|---|---|---|
| `iris_classification.py` | Iris (3 classes) | Logistic Regression, plus a Decision Tree comparison |
| `breast_cancer_classification.py` | Breast Cancer (2 classes) | Logistic Regression with feature scaling, plus a Decision Tree comparison |
| `wine_classification.py` | Wine (3 classes) | Logistic Regression with feature scaling, plus a Decision Tree comparison |

All datasets are loaded from scikit-learn's built-in datasets, so no CSV files are needed.

## Requirements

```
pip install scikit-learn pandas matplotlib seaborn
```

## How to Run

```
python iris_classification.py
python breast_cancer_classification.py
python wine_classification.py
```

Each script prints the dataset details and the metrics, saves all charts as PNG files, then shows them in windows.

## Charts

Saved in `charts/iris`, `charts/breast_cancer` and `charts/wine`:

| File | Shows |
|---|---|
| `01_class_distribution.png` | Number of samples per class |
| `02_feature_histograms.png` | Distribution of each feature |
| `03_correlation_heatmap.png` | Correlation between features |
| `04_pair_plot.png` | Feature pairs coloured by class (Iris only) |
| `05_evaluation_metrics.png` | Accuracy, Precision, Recall and F1-Score |
| `06_train_vs_test_accuracy.png` | Training vs testing accuracy for both models |
| `07_confusion_matrices.png` | Confusion matrices for Logistic Regression and Decision Tree |
| `08_decision_tree.png` | The Decision Tree (max_depth=3) |
| `09_underfitting_vs_overfitting.png` | Accuracy across tree depths, showing underfitting and overfitting |

## What Each Script Does

1. Loads the dataset
2. Explores features, target classes and class distribution
3. Splits the data into 80% training and 20% testing (stratified, `random_state=42`)
4. Trains a Logistic Regression model
5. Predicts on the training and testing sets
6. Evaluates with Accuracy, Precision, Recall, F1-Score and a Confusion Matrix
7. Trains a Decision Tree for comparison and saves all charts

## Results

| Dataset | Train Acc. | Test Acc. | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Iris | 0.975 | 0.967 | 0.970 | 0.967 | 0.967 |
| Breast Cancer | 0.989 | 0.983 | 0.986 | 0.986 | 0.986 |
| Wine | 1.000 | 0.972 | 0.978 | 0.967 | 0.971 |

Iris and Wine use macro averages across classes. Breast Cancer is binary.

## Key Concepts

**Training vs Testing Performance**
Training accuracy shows how well the model fits data it has seen. Testing accuracy shows how well it generalises to new data.

**Overfitting and Underfitting**
- Overfitting: very high training score, noticeably lower testing score. The Decision Tree on Iris scores 1.0 on training and 0.933 on testing.
- Underfitting: low scores on both training and testing, meaning the model is too simple. On Iris, a tree with `max_depth=1` scores about 0.67 on both.

**Metrics**
- Accuracy: correct predictions out of all predictions
- Precision: of the predicted positives, how many were truly positive
- Recall: of the actual positives, how many were found
- F1-Score: the harmonic mean of precision and recall

**Confusion Matrix**
Rows are actual classes and columns are predicted classes. The diagonal holds correct predictions and everything off the diagonal is an error.

**Classification vs Regression**
Classification predicts a category. Regression predicts a continuous number.

**Logistic Regression**
A linear model that outputs class probabilities and is used for classification.

**Decision Trees**
A model that splits the data with a sequence of feature-based rules. It is easy to interpret but prone to overfitting.

**Real-world Classification Examples**
Spam detection, disease diagnosis, fraud detection, image recognition and customer churn prediction.
