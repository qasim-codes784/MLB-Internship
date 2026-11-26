import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Python version    :", sys.version.split()[0])
print("TensorFlow version:", tf.__version__)
print("Keras version     :", keras.__version__)
print("GPU devices       :", tf.config.list_physical_devices("GPU") or "None (using CPU)")

a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
print("\nTensor multiplication test:")
print(tf.matmul(a, b).numpy())

test_model = keras.Sequential([layers.Input(shape=(2,)), layers.Dense(1)])
print("\nTest model output shape:", test_model.predict(a.numpy(), verbose=0).shape)

print("\nTensorFlow and Keras imported and working successfully.")
