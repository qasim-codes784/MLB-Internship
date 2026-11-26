import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(42)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential(
    [
        layers.Input(shape=(4,)),
        layers.Dense(16, activation="relu", name="hidden_layer_1"),
        layers.Dense(8, activation="relu", name="hidden_layer_2"),
        layers.Dense(3, activation="softmax", name="output_layer"),
    ],
    name="iris_ann",
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=0)

train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\nFinal training accuracy  :", round(train_acc, 4))
print("Final validation accuracy:", round(val_acc, 4))
print("Test accuracy            :", round(test_acc, 4))
print("Test loss                :", round(test_loss, 4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history.history["accuracy"], label="Training", color="#4C78A8")
axes[0].plot(history.history["val_accuracy"], label="Validation", color="#F58518")
axes[0].set_title("ANN on Iris - Accuracy")
axes[0].set_ylabel("Accuracy")
axes[1].plot(history.history["loss"], label="Training", color="#4C78A8")
axes[1].plot(history.history["val_loss"], label="Validation", color="#F58518")
axes[1].set_title("ANN on Iris - Loss")
axes[1].set_ylabel("Loss")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
path = os.path.join(OUT_DIR, "iris_ann_training_curves.png")
plt.savefig(path, dpi=150)

print("\nChart saved to:", path)
plt.show()
