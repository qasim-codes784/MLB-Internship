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

print("PRACTICE 1: LOAD, VISUALIZE AND NORMALIZE")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print("Training images:", X_train.shape)
print("Testing images :", X_test.shape)
print("Pixel range before normalization:", X_train.min(), "to", X_train.max())

fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(class_names[y_train[i]])
    ax.axis("off")
fig.suptitle("Fashion MNIST - First 10 Training Images")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_images.png"), dpi=150)

X_train = (X_train / 255.0).reshape(-1, 28, 28, 1)
X_test = (X_test / 255.0).reshape(-1, 28, 28, 1)
print("Pixel range after normalization :", X_train.min(), "to", X_train.max())
print("Shape after adding channel      :", X_train.shape)

print("\nPRACTICE 2: BUILD AND TRAIN A SIMPLE CNN")
model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", name="conv_layer"),
        layers.MaxPooling2D((2, 2), name="max_pooling"),
        layers.Flatten(name="flatten"),
        layers.Dense(64, activation="relu", name="dense_layer"),
        layers.Dense(10, activation="softmax", name="output_layer"),
    ],
    name="simple_cnn",
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)

print("\nPRACTICE 3: EVALUATION")
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Training accuracy:", round(train_acc, 4))
print("Training loss    :", round(train_loss, 4))
print("Test accuracy    :", round(test_acc, 4))
print("Test loss        :", round(test_loss, 4))

probabilities = model.predict(X_test[:10], verbose=0)
predicted = probabilities.argmax(axis=1)
print("\nPredictions on 10 sample test images:")
print(f"{'#':<4}{'Actual':<14}{'Predicted':<14}{'Confidence':<12}{'Correct'}")
for i in range(10):
    ok = "Yes" if predicted[i] == y_test[i] else "No"
    print(f"{i:<4}{class_names[y_test[i]]:<14}{class_names[predicted[i]]:<14}{probabilities[i][predicted[i]]:<12.2%}{ok}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history.history["accuracy"], marker="o", label="Training", color="#4C78A8")
axes[0].plot(history.history["val_accuracy"], marker="o", label="Validation", color="#F58518")
axes[0].set_title("Accuracy")
axes[1].plot(history.history["loss"], marker="o", label="Training", color="#4C78A8")
axes[1].plot(history.history["val_loss"], marker="o", label="Validation", color="#F58518")
axes[1].set_title("Loss")
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "practice_training_curves.png"), dpi=150)

fig, axes = plt.subplots(2, 5, figsize=(13, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ok = predicted[i] == y_test[i]
    ax.set_title(f"Pred: {class_names[predicted[i]]}\nActual: {class_names[y_test[i]]}", color="green" if ok else "red", fontsize=10)
    ax.axis("off")
fig.suptitle("Practice Predictions (green = correct, red = wrong)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "practice_predictions.png"), dpi=150)

print("\nCharts saved in:", OUT_DIR)
plt.show()
