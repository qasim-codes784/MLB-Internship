import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

keras.utils.set_random_seed(42)

model = keras.Sequential(
    [
        layers.Input(shape=(4,), name="input_layer"),
        layers.Dense(8, activation="relu", name="hidden_layer"),
        layers.Dense(3, activation="softmax", name="output_layer"),
    ],
    name="simple_neural_network",
)

print("MODEL SUMMARY")
model.summary()

print("\nLAYER EXPLANATION")
print("input_layer : receives 4 numbers (4 features). It has no weights, it only passes data in.")
print("hidden_layer: 8 neurons. Each neuron takes all 4 inputs, applies weights and a bias, then ReLU.")
print("output_layer: 3 neurons, one per class. Softmax turns the 3 scores into probabilities that sum to 1.")

print("\nPARAMETER COUNT")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        w, b = weights
        print(f"{layer.name}: weights {w.shape} + biases {b.shape} = {w.size + b.size} parameters")
print("Total parameters:", model.count_params())

sample = np.array([[5.1, 3.5, 1.4, 0.2]])
probabilities = model.predict(sample, verbose=0)[0]
print("\nForward pass on one sample (untrained model):")
print("Sample input     :", sample[0].tolist())
print("Output probabilities:", [round(float(p), 4) for p in probabilities])
print("Sum of probabilities:", round(float(probabilities.sum()), 4))
