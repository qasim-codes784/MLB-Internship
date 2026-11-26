# Day 8 – Introduction to Deep Learning

Phase 2 of the internship starts here: neural networks with TensorFlow/Keras. The mini project uses the Fashion MNIST dataset (70,000 grayscale 28x28 images of clothing, 10 classes), built into Keras.

## Files

| File | Purpose |
|---|---|
| `01_tensorflow_setup.py` | Practice 1: verify TensorFlow and Keras, run a tensor test |
| `02_simple_neural_network.py` | Practice 2: input, hidden and output layer, model summary, layer explanation |
| `03_activation_functions.py` | Practice 3: ReLU, Sigmoid and Tanh compared (structure, outputs, training) |
| `04_ann_model.py` | ANN on Iris with training and validation curves |
| `05_fashion_mnist_ann.py` | Mini project: Fashion MNIST ANN |
| `activation_functions.png`, `activation_training_comparison.png` | Activation function graphs |
| `iris_ann_training_curves.png` | Iris ANN accuracy and loss |
| `dataset_samples.png`, `fashion_training_curves.png`, `sample_predictions.png` | Mini project graphs and predictions |

## How to Run

```
pip install tensorflow scikit-learn matplotlib
python 01_tensorflow_setup.py
python 02_simple_neural_network.py
python 03_activation_functions.py
python 04_ann_model.py
python 05_fashion_mnist_ann.py
```

TensorFlow needs 64-bit Python 3.9 to 3.12 on Windows. The first run of script 05 downloads Fashion MNIST (about 30 MB).

## What is Deep Learning?

Deep Learning is a part of Machine Learning that uses neural networks with many layers to learn patterns directly from data. Each layer transforms its input a little, and stacking layers lets the network learn complex patterns such as shapes in images or meaning in text. It powers image classification, object detection, speech recognition and language models.

## Machine Learning vs Deep Learning

| | Machine Learning | Deep Learning |
|---|---|---|
| Features | Often chosen by hand | Learned automatically from raw data |
| Data needed | Works well on small and medium data | Needs large amounts of data |
| Hardware | CPU is usually enough | Benefits from GPUs |
| Best for | Tabular data (Logistic Regression, trees) | Images, audio, text |
| Interpretability | Easier to explain | Harder to explain |

## What is a Perceptron?

The simplest neural network: one neuron. It multiplies each input by a weight, adds them up with a bias, and passes the result through an activation function to give an output: `output = activation(w1*x1 + w2*x2 + ... + bias)`. With a step function it makes a yes/no decision. Learning means adjusting the weights and bias to reduce errors. Many perceptron-like neurons arranged in layers form an Artificial Neural Network.

## Layers in a Neural Network

- **Input layer:** receives the data (784 pixel values for a Fashion MNIST image after flattening).
- **Hidden layers:** neurons that learn patterns (128 and 64 neurons here).
- **Output layer:** gives the prediction (10 neurons, one per clothing class).

## Activation Functions Explored

| Function | Output range | Commonly used in |
|---|---|---|
| ReLU | 0 to infinity | Hidden layers of most modern networks (simple and fast) |
| Sigmoid | 0 to 1 | Output layer for binary classification |
| Tanh | -1 to 1 | Hidden layers, especially in recurrent networks |
| Softmax | 0 to 1, sums to 1 | Output layer for multi-class classification |

Without activation functions, stacked layers would collapse into a single linear model and could not learn curved, complex patterns.

**Observation from Practice 3:** changing the activation function does not change the model structure. All three models had the same layers, output shapes and 67 parameters. It changes the values inside the network (ReLU gives 0 for negative inputs, Sigmoid squeezes values into 0 to 1, Tanh into -1 to 1).

On Iris (200 epochs), test accuracy was 0.933 for ReLU, 0.900 for Sigmoid and 0.933 for Tanh. These differences are small, and with only 30 test samples one flower is 3.3%, so they should not be over-interpreted.

## Mini Project: Fashion MNIST ANN

Architecture: Flatten (784) → Dense 128 (ReLU) → Dense 64 (ReLU) → Dense 10 (Softmax), 109,386 parameters. Pixels were normalized to 0–1. Trained for 10 epochs with Adam and sparse categorical crossentropy, using 10% of the training data for validation.

| Measure | Value |
|---|---|
| Final training accuracy | 0.9132 |
| Final validation accuracy | 0.8658 |
| Test accuracy | 0.8624 |
| Test loss | 0.4063 |

Per-class test accuracy was highest for Sandal (98.1%), Trouser (96.6%) and Bag (96.5%), and lowest for Shirt (45.9%) and Coat (74.1%).

Exact numbers can differ by about one percent between runs and machines.

## Observations

- Training accuracy kept rising (0.822 to 0.913) while validation accuracy flattened at about 0.87 after epoch 4, and validation loss started rising. The model is beginning to overfit, the same pattern as the Decision Tree on Iris in Day 5.
- Shirts are the hardest class. They look like T-shirts, pullovers and coats, so the model confuses them. Distinct shapes like trousers, sandals and bags are almost always correct.
- A simple ANN reaches about 86% on this dataset. Convolutional networks, which understand image structure, are the next step and usually reach 90%+.
- Predictions with 100% confidence can still be wrong. The one error among the first 15 test images (a sneaker predicted as a bag) shows that confidence is not the same as correctness.
