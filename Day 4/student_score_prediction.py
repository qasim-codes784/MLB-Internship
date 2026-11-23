from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import SIGNATURE
from linear_regression_model import train_and_evaluate

CHART_DIR = Path(__file__).resolve().parent / "charts"
CHART_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

model, data, comparison, metrics = train_and_evaluate()

print("=" * 50)
print(" STUDENT SCORE PREDICTION SYSTEM")
print("=" * 50)
print(f"\nStudents in dataset: {data['total_rows']}")
print(f"Training set: {len(data['y_train'])} students | Test set: {len(data['y_test'])} students")
print("Features used: maths, science, course")
print("Target: Average_Score")

print("\nActual vs Predicted:")
print(comparison.to_string(index=False))

print("\nModel evaluation:")
print(f"  MAE: {metrics['MAE']:.3f}")
print(f"  MSE: {metrics['MSE']:.3f}")
print(f"  R2:  {metrics['R2']:.3f}")

actual = comparison["Actual"]
predicted = comparison["Predicted"]
low = min(actual.min(), predicted.min()) - 3
high = max(actual.max(), predicted.max()) + 3

plt.figure(figsize=(7, 6))
plt.scatter(actual, predicted, s=100, color="steelblue", edgecolor="black", label="Test students")
plt.plot([low, high], [low, high], "r--", label="Perfect prediction")
plt.xlim(low, high)
plt.ylim(low, high)
plt.xlabel("Actual Average Score")
plt.ylabel("Predicted Average Score")
plt.title(f"Actual vs Predicted (R2 = {metrics['R2']:.2f})")
plt.legend()
plt.tight_layout()
target = CHART_DIR / "actual_vs_predicted.png"
try:
    plt.savefig(target, dpi=150)
    saved = target
except OSError:
    saved = CHART_DIR / "actual_vs_predicted_new.png"
    plt.savefig(saved, dpi=150)
    print("\nWarning: actual_vs_predicted.png is open in another program. Close it and run again.")
plt.close()

print(f"\nGraph saved -> charts/{saved.name}")
print("-----------------------\n" + SIGNATURE)
