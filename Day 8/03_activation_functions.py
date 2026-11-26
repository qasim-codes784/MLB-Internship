import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(42)


def build_model(activation):
    return keras.Sequential(
        [
            layers.Input(shape=(4,)),
            layers.Dense(8, activation=activation, name="hidden_layer"),
            layers.Dense(3, activation="softmax", name="output_layer"),
        ],
        name=f"model_{activation}",
    )


activations = ["relu", "sigmoid", "tanh"]
models = {a: build_model(a) for a in activations}

print("MODEL STRUCTURE FOR EACH ACTIVATION")
print(f"{'Activation':<12}{'Hidden output':<16}{'Output shape':<16}{'Parameters'}")
for a, m in models.items():
    print(f"{a:<12}{str(m.layers[0].output.shape):<16}{str(m.layers[1].output.shape):<16}{m.count_params()}")
print("\nObservation: the structure (layers, shapes, parameter count) is identical.")
print("The activation function changes the values flowing through the network, not its shape.")

reference_weights = models["relu"].get_weights()
for m in models.values():
    m.set_weights(reference_weights)

sample = np.array([[-1.0, 0.5, 2.0, -0.5]])
print("\nHIDDEN LAYER OUTPUT FOR THE SAME INPUT AND SAME WEIGHTS")
print("Input:", sample[0].tolist())
for a, m in models.items():
    hidden = keras.Model(m.inputs, m.layers[0].output)
    values = hidden.predict(sample, verbose=0)[0]
    print(f"{a:<8}: {[round(float(v), 3) for v in values]}")

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTRAINING EACH MODEL ON IRIS (200 epochs)")
histories = {}
for a in activations:
    keras.utils.set_random_seed(42)
    m = build_model(a)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    histories[a] = m.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=0)
    loss, acc = m.evaluate(X_test, y_test, verbose=0)
    print(f"{a:<8} train acc={histories[a].history['accuracy'][-1]:.4f}  val acc={histories[a].history['val_accuracy'][-1]:.4f}  test acc={acc:.4f}")

x = np.linspace(-6, 6, 400)
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
axes[0, 0].plot(x, np.maximum(0, x), color="#4C78A8")
axes[0, 0].set_title("ReLU: max(0, x)")
axes[0, 1].plot(x, 1 / (1 + np.exp(-x)), color="#F58518")
axes[0, 1].set_title("Sigmoid: 1 / (1 + e^-x)")
axes[1, 0].plot(x, np.tanh(x), color="#54A24B")
axes[1, 0].set_title("Tanh")
scores = np.array([2.0, 1.0, 0.1])
soft = np.exp(scores) / np.exp(scores).sum()
bars = axes[1, 1].bar(["Class 0", "Class 1", "Class 2"], soft, color="#E45756")
for bar, v in zip(bars, soft):
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
axes[1, 1].set_title("Softmax of scores [2.0, 1.0, 0.1]")
axes[1, 1].set_ylim(0, 1)
for ax in axes.flat[:3]:
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.grid(alpha=0.3)
fig.suptitle("Activation Functions")
plt.tight_layout()
curves_path = os.path.join(OUT_DIR, "activation_functions.png")
plt.savefig(curves_path, dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for a, color in zip(activations, ["#4C78A8", "#F58518", "#54A24B"]):
    axes[0].plot(histories[a].history["accuracy"], label=a, color=color)
    axes[1].plot(histories[a].history["val_accuracy"], label=a, color=color)
axes[0].set_title("Training Accuracy")
axes[1].set_title("Validation Accuracy")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
train_path = os.path.join(OUT_DIR, "activation_training_comparison.png")
plt.savefig(train_path, dpi=150)

print("\nCharts saved to:")
print(" ", curves_path)
print(" ", train_path)
plt.show()
