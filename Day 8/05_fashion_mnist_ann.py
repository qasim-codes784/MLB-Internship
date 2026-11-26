import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import fashion_mnist

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(42)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

print("=" * 60)
print("FASHION MNIST - ARTIFICIAL NEURAL NETWORK")
print("=" * 60)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("\nDATASET EXPLORATION")
print("Training images shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape :", X_test.shape)
print("Testing labels shape :", y_test.shape)
print("Pixel value range    :", X_train.min(), "to", X_train.max())
print("Image size           : 28 x 28 pixels, grayscale")
print("\nClasses and training samples per class:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name:<12} {int((y_train == i).sum())}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap="gray")
    ax.set_title(f"{i}: {class_names[i]}")
    ax.axis("off")
fig.suptitle("Fashion MNIST - One Sample per Class")
plt.tight_layout()
samples_path = os.path.join(OUT_DIR, "dataset_samples.png")
plt.savefig(samples_path, dpi=150)

X_train = X_train / 255.0
X_test = X_test / 255.0
print("\nAfter normalization, pixel range:", X_train.min(), "to", X_train.max())

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28), name="input_layer"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="hidden_layer_1"),
        layers.Dense(64, activation="relu", name="hidden_layer_2"),
        layers.Dense(10, activation="softmax", name="output_layer"),
    ],
    name="fashion_ann",
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("\nMODEL SUMMARY")
model.summary()

print("\nTRAINING")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=2)

train_acc = history.history["accuracy"][-1]
val_acc = history.history["val_accuracy"][-1]
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\nRESULTS")
print("Final training accuracy  :", round(train_acc, 4))
print("Final validation accuracy:", round(val_acc, 4))
print("Test accuracy            :", round(test_acc, 4))
print("Test loss                :", round(test_loss, 4))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history.history["accuracy"], marker="o", label="Training", color="#4C78A8")
axes[0].plot(history.history["val_accuracy"], marker="o", label="Validation", color="#F58518")
axes[0].set_title("Training and Validation Accuracy")
axes[0].set_ylabel("Accuracy")
axes[1].plot(history.history["loss"], marker="o", label="Training", color="#4C78A8")
axes[1].plot(history.history["val_loss"], marker="o", label="Validation", color="#F58518")
axes[1].set_title("Training and Validation Loss")
axes[1].set_ylabel("Loss")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
curves_path = os.path.join(OUT_DIR, "fashion_training_curves.png")
plt.savefig(curves_path, dpi=150)

probabilities = model.predict(X_test, verbose=0)
predicted = probabilities.argmax(axis=1)
wrong_total = int((predicted != y_test).sum())
print(f"\nCorrect predictions on test set: {len(y_test) - wrong_total} of {len(y_test)}")

print("\nSAMPLE PREDICTIONS (first 10 test images)")
print(f"{'#':<4}{'Actual':<14}{'Predicted':<14}{'Confidence':<12}{'Correct'}")
for i in range(10):
    conf = probabilities[i][predicted[i]]
    ok = "Yes" if predicted[i] == y_test[i] else "No"
    print(f"{i:<4}{class_names[y_test[i]]:<14}{class_names[predicted[i]]:<14}{conf:<12.2%}{ok}")

fig, axes = plt.subplots(3, 5, figsize=(13, 8.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i], cmap="gray")
    ok = predicted[i] == y_test[i]
    ax.set_title(
        f"Pred: {class_names[predicted[i]]}\nActual: {class_names[y_test[i]]}",
        color="green" if ok else "red",
        fontsize=10,
    )
    ax.axis("off")
fig.suptitle("Sample Predictions (green = correct, red = wrong)")
plt.tight_layout()
pred_path = os.path.join(OUT_DIR, "sample_predictions.png")
plt.savefig(pred_path, dpi=150)

per_class = []
for i, name in enumerate(class_names):
    m = y_test == i
    per_class.append((name, float((predicted[m] == i).mean())))
print("\nACCURACY PER CLASS")
for name, acc in sorted(per_class, key=lambda t: t[1], reverse=True):
    print(f"  {name:<12} {acc:.2%}")

print("\nCharts saved to:")
print(" ", samples_path)
print(" ", curves_path)
print(" ", pred_path)
plt.show()
