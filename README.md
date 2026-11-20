# Day 4 – Data Preprocessing and First Machine Learning Model

Dataset: `students.csv` (12 sample students: maths, science, english, course).
> The file is demo data, so the results describe this sample only.

## How to run
```bash
pip install pandas matplotlib seaborn scikit-learn
python data_preprocessing.py
python linear_regression_model.py
python student_score_prediction.py
```

## Files
| File | Purpose |
|---|---|
| `data_preprocessing.py` | Loads data, splits, fills missing values, scales and encodes |
| `linear_regression_model.py` | Trains Linear Regression and calculates MAE, MSE, R² |
| `student_score_prediction.py` | Mini project: full pipeline, comparison table, graph |
| `charts/actual_vs_predicted.png` | Actual vs Predicted scatter plot |

## What I learned about data preprocessing
- **Features and target:** X = `maths`, `science`, `course`. y = `Average_Score` (mean of maths, science, english).
- **Categorical data:** `course` is text with no order (AI, Computer Vision, Data Science), so I used One-Hot Encoding. Label Encoding would wrongly suggest one course is "bigger" than another.
- **Feature scaling:** I used Standardization (`StandardScaler`) on marks, so each column has mean 0 and standard deviation 1.
- **Missing values:** 1 science and 1 english mark were blank. I filled them with the median of the **training** data only.
- **Data leakage:** every step that learns from data (median, scaler, encoder) is fitted on the training set only and then applied to the test set. I also left `english` out of the features, because `Average_Score` is built from it. With all three subjects as features, the model gets R² = 1.0 and MAE = 0.0, because it just rediscovers the averaging formula. That is leakage, not real learning.

## Why train-test splitting is important
A model can memorize its training data. Testing on students it has never seen shows whether it really learned a pattern. I used an 80/20 split (`random_state=42`): 9 students for training and 3 for testing.

## Evaluation metrics used
- **MAE** (Mean Absolute Error): average size of the prediction error, in score points.
- **MSE** (Mean Squared Error): average squared error, which punishes big mistakes more.
- **R² Score:** how much of the variation in scores the model explains (1.0 is perfect, 0 is no better than guessing the average).

## Model performance and observations
| Metric | Value |
|---|---|
| MAE | 1.921 |
| MSE | 4.303 |
| R² | 0.896 |

| Student | Actual | Predicted |
|---|---|---|
| Sample Student K | 82.00 | 79.90 |
| Sample Student J | 69.67 | 70.56 |
| Sample Student A | 84.33 | 81.55 |

1. Predictions are on average about 1.9 points away from the real average, so the model fits well.
2. Maths has the largest coefficient (14.6), so it influences the predicted average most. The course adds almost nothing (coefficients between -0.5 and 0.7).
3. **Limitation:** the test set has only 3 students, so these numbers are not reliable. A different split or a larger dataset could change R² a lot.

## Screen recording
_(add link / file here)_

- Day 4: Data preprocessing and Linear Regression (see Day-4 folder)

- Day 4: Data preprocessing and Linear Regression (see Day-4 folder)
