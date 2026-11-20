import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import SIGNATURE, preprocess


def train_and_evaluate():
    data = preprocess()
    model = LinearRegression()
    model.fit(data["X_train"], data["y_train"])

    predictions = model.predict(data["X_test"])
    metrics = {
        "MAE": mean_absolute_error(data["y_test"], predictions),
        "MSE": mean_squared_error(data["y_test"], predictions),
        "R2": r2_score(data["y_test"], predictions),
    }

    comparison = pd.DataFrame({
        "Student": data["test_df"]["name"].values,
        "Actual": data["y_test"].values,
        "Predicted": predictions.round(2),
    })
    comparison["Error"] = (comparison["Actual"] - comparison["Predicted"]).round(2)
    return model, data, comparison, metrics


if __name__ == "__main__":
    model, data, comparison, metrics = train_and_evaluate()
    print("Model: Linear Regression")
    print("Training rows:", len(data["y_train"]), "| Testing rows:", len(data["y_test"]))

    print("\nLearned coefficients:")
    for name, coef in zip(data["feature_names"], model.coef_):
        print(f"  {name:<28}{coef:>8.3f}")
    print(f"  {'intercept':<28}{model.intercept_:>8.3f}")

    print("\nActual vs Predicted (test set):")
    print(comparison.to_string(index=False))

    print("\nEvaluation metrics:")
    print(f"  Mean Absolute Error (MAE): {metrics['MAE']:.3f}")
    print(f"  Mean Squared Error (MSE):  {metrics['MSE']:.3f}")
    print(f"  R2 Score:                  {metrics['R2']:.3f}")
    print("-----------------------\n" + SIGNATURE)
