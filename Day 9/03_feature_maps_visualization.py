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

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = (X_train / 255.0).reshape(-1, 28, 28, 1)
X_test = (X_test / 255.0).reshape(-1, 28, 28, 1)

inputs = keras.Input(shape=(28, 28, 1))
conv1 = layers.Conv2D(8, (3, 3), activation="relu", name="conv_1")(inputs)
pool1 = layers.MaxPooling2D((2, 2), name="pool_1")(conv1)
conv2 = layers.Conv2D(16, (3, 3), activation="relu", name="conv_2")(pool1)
pool2 = layers.MaxPooling2D((2, 2), name="pool_2")(conv2)
flat = layers.Flatten()(pool2)
outputs = layers.Dense(10, activation="softmax")(flat)
model = keras.Model(inputs, outputs, name="small_cnn")
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print("Training a small CNN for 3 epochs to learn its filters...")
model.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1, verbose=2)

print("\nSHAPE OF DATA AS IT PASSES THROUGH EACH LAYER")
for name in ["conv_1", "pool_1", "conv_2", "pool_2"]:
    print(f"{name}: {model.get_layer(name).output.shape}")

filters, biases = model.get_layer("conv_1").get_weights()
print("\nconv_1 filters shape:", filters.shape, "(3x3 kernel, 1 input channel, 8 filters)")

fig, axes = plt.subplots(1, 8, figsize=(14, 2.5))
for i, ax in enumerate(axes):
    ax.imshow(filters[:, :, 0, i], cmap="gray")
    ax.set_title(f"Filter {i + 1}")
    ax.axis("off")
fig.suptitle("Learned 3x3 Filters of the First Convolution Layer")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "learned_filters.png"), dpi=150)

sample = X_test[0:1]
extractor = keras.Model(model.inputs, [model.get_layer(n).output for n in ["conv_1", "pool_1", "conv_2"]])
maps_conv1, maps_pool1, maps_conv2 = extractor.predict(sample, verbose=0)

fig, axes = plt.subplots(3, 8, figsize=(15, 6.5))
axes[0, 0].imshow(sample[0].reshape(28, 28), cmap="gray")
axes[0, 0].set_title("Input")
for j in range(1, 8):
    axes[0, j].axis("off")
axes[0, 0].axis("off")
for i in range(8):
    axes[1, i].imshow(maps_conv1[0, :, :, i], cmap="viridis")
    axes[1, i].set_title(f"conv_1 map {i + 1}", fontsize=9)
    axes[1, i].axis("off")
    axes[2, i].imshow(maps_conv2[0, :, :, i], cmap="viridis")
    axes[2, i].set_title(f"conv_2 map {i + 1}", fontsize=9)
    axes[2, i].axis("off")
fig.suptitle(f"Feature Maps for a Test Image ({class_names[y_test[0]]})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_maps.png"), dpi=150)

print("\nCharts saved in:", OUT_DIR)
plt.show()
