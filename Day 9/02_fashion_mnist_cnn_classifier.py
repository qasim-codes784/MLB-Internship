import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
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
print("FASHION MNIST IMAGE CLASSIFIER USING CNN")
print("=" * 60)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print("\nTraining images:", X_train.shape)
print("Testing images :", X_test.shape)
print("Classes        :", class_names)

fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))
for i, ax in enumerate(axes.flat):
    idx = np.where(y_train == i)[0][0]
    ax.imshow(X_train[idx], cmap="gray")
    ax.set_title(f"{i}: {class_names[i]}")
    ax.axis("off")
fig.suptitle("Fashion MNIST - One Sample per Class")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "dataset_samples.png"), dpi=150)

X_train = (X_train / 255.0).reshape(-1, 28, 28, 1)
X_test = (X_test / 255.0).reshape(-1, 28, 28, 1)
print("\nPixels normalized to range:", X_train.min(), "to", X_train.max())
print("Input shape for CNN:", X_train.shape[1:])

model = keras.Sequential(
    [
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", name="conv_1"),
        layers.MaxPooling2D((2, 2), name="pool_1"),
        layers.Conv2D(64, (3, 3), activation="relu", name="conv_2"),
        layers.MaxPooling2D((2, 2), name="pool_2"),
        layers.Flatten(name="flatten"),
        layers.Dense(128, activation="relu", name="dense_1"),
        layers.Dropout(0.3, name="dropout"),
        layers.Dense(10, activation="softmax", name="output_layer"),
    ],
    name="fashion_cnn",
)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("\nMODEL SUMMARY")
model.summary()

print("\nTRAINING")
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1, verbose=2)

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
val_acc = history.history["val_accuracy"][-1]

print("\nRESULTS")
print("Training accuracy  :", round(train_acc, 4))
print("Validation accuracy:", round(val_acc, 4))
print("Test accuracy      :", round(test_acc, 4))
print("Training loss      :", round(train_loss, 4))
print("Test loss          :", round(test_loss, 4))
print("Day 8 ANN test accuracy for comparison: 0.8624")

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
curves_path = os.path.join(OUT_DIR, "training_curves.png")
plt.savefig(curves_path, dpi=150)

probabilities = model.predict(X_test, verbose=0)
predicted = probabilities.argmax(axis=1)
correct_idx = np.where(predicted == y_test)[0]
wrong_idx = np.where(predicted != y_test)[0]
print(f"\nCorrect predictions  : {len(correct_idx)} of {len(y_test)}")
print(f"Incorrect predictions: {len(wrong_idx)} of {len(y_test)}")

print("\nSAMPLE PREDICTIONS (first 10 test images)")
print(f"{'#':<4}{'Actual':<14}{'Predicted':<14}{'Confidence':<12}{'Correct'}")
for i in range(10):
    ok = "Yes" if predicted[i] == y_test[i] else "No"
    print(f"{i:<4}{class_names[y_test[i]]:<14}{class_names[predicted[i]]:<14}{probabilities[i][predicted[i]]:<12.2%}{ok}")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predicted, target_names=class_names, digits=3))

cm = confusion_matrix(y_test, predicted)
fig, ax = plt.subplots(figsize=(10, 9))
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45, colorbar=False)
ax.set_title("Fashion MNIST CNN - Confusion Matrix")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150)

print("MOST CONFUSED PAIRS (actual -> predicted)")
off = cm.copy()
np.fill_diagonal(off, 0)
for flat in np.argsort(off, axis=None)[::-1][:5]:
    a, p = divmod(flat, 10)
    print(f"  {class_names[a]} -> {class_names[p]}: {off[a, p]} images")

print("\nACCURACY PER CLASS")
per_class = cm.diagonal() / cm.sum(axis=1)
for i in np.argsort(per_class)[::-1]:
    print(f"  {class_names[i]:<12} {per_class[i]:.2%}")


def show_grid(indices, title, filename, good):
    fig, axes = plt.subplots(2, 5, figsize=(13, 6.5))
    for ax, idx in zip(axes.flat, indices):
        ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
        ax.set_title(
            f"Pred: {class_names[predicted[idx]]}\nActual: {class_names[y_test[idx]]}\n({probabilities[idx][predicted[idx]]:.0%})",
            color="green" if good else "red",
            fontsize=10,
        )
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150)
    return path


correct_path = show_grid(correct_idx[:10], "10 Correctly Classified Images", "correct_predictions.png", True)
wrong_path = show_grid(wrong_idx[:10], "10 Incorrectly Classified Images", "incorrect_predictions.png", False)

print("\nCharts saved in:", OUT_DIR)
plt.show()
