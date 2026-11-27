# Day 9 – Convolutional Neural Networks (CNN) and Image Classification

Fashion MNIST (70,000 grayscale 28x28 images, 10 clothing classes) classified with a CNN built in TensorFlow/Keras.

## Files

| File | Purpose |
|---|---|
| `01_cnn_practice.py` | Practice 1 to 3: load, visualize 10 images, normalize, build a simple CNN, train, evaluate, predict |
| `02_fashion_mnist_cnn_classifier.py` | Mini project: full CNN classifier with curves, confusion matrix, correct and incorrect predictions |
| `03_feature_maps_visualization.py` | Shows the learned filters and the feature maps inside the network |
| `training_curves.png` | Training and validation accuracy and loss (mini project) |
| `confusion_matrix.png` | Confusion matrix on the 10,000 test images |
| `correct_predictions.png`, `incorrect_predictions.png` | 10 correct and 10 incorrect predictions with predicted and actual labels |
| `dataset_samples.png`, `sample_images.png` | Dataset samples |
| `practice_training_curves.png`, `practice_predictions.png` | Practice script graphs and predictions |
| `learned_filters.png`, `feature_maps.png` | Filters and feature maps |

## How to Run

```
pip install tensorflow scikit-learn matplotlib
python 01_cnn_practice.py
python 02_fashion_mnist_cnn_classifier.py
python 03_feature_maps_visualization.py
```

Script 02 trains for 10 epochs and takes a few minutes on a CPU.

## Why CNNs are Better than ANNs for Image Data

- An ANN flattens the image into one long list of pixels immediately, so it loses the position of each pixel. A CNN keeps the 2D structure and looks at small neighbourhoods of pixels, so it can find edges, corners and shapes.
- **Weight sharing:** one small filter is slid across the whole image, so the same pattern is detected wherever it appears. The first convolution layer here has only 320 parameters, while a Dense layer on the flattened image would need about 100,000.
- **Translation tolerance:** a sleeve or a sole is recognised even if it shifts a few pixels.
- Result on the same dataset: the Day 8 ANN reached 0.8624 test accuracy and this CNN reaches 0.912.

## Purpose of Convolution and Pooling Layers

- **Convolution layer:** slides small filters (kernels, here 3x3) over the image. Each filter produces a feature map that shows where a particular pattern appears. Early layers learn simple patterns such as edges, and deeper layers combine them into larger shapes.
- **Max pooling layer:** keeps only the largest value in each 2x2 block. It halves the width and height, reduces computation, and makes the network less sensitive to small shifts. Average pooling does the same using the average instead of the maximum.
- **Flatten layer:** turns the final feature maps into one long vector.
- **Fully connected (Dense) layers:** use those features to decide the class.

## Model Architecture

| Layer | Output shape | Parameters |
|---|---|---|
| Input | 28 x 28 x 1 | 0 |
| Conv2D (32 filters, 3x3, ReLU) | 26 x 26 x 32 | 320 |
| MaxPooling2D (2x2) | 13 x 13 x 32 | 0 |
| Conv2D (64 filters, 3x3, ReLU) | 11 x 11 x 64 | 18,496 |
| MaxPooling2D (2x2) | 5 x 5 x 64 | 0 |
| Flatten | 1600 | 0 |
| Dense (128, ReLU) | 128 | 204,928 |
| Dropout (0.3) | 128 | 0 |
| Dense (10, Softmax) | 10 | 1,290 |
| **Total** | | **225,034** |

Optimizer: Adam. Loss: sparse categorical crossentropy. 10 epochs, batch size 64, 10% of the training data used for validation. Pixels are normalized to 0–1.

## Results

| Measure | Value |
|---|---|
| Training accuracy | 0.9450 |
| Validation accuracy | 0.9172 |
| Test accuracy | 0.9120 |
| Training loss | 0.1469 |
| Test loss | 0.2618 |
| Correct test predictions | 9,120 of 10,000 |

Graphs: `training_curves.png` and `confusion_matrix.png`. Per-class test accuracy was highest for Sandal (98.6%), Bag (98.1%) and Sneaker (97.6%), and lowest for Shirt (69.6%).

## Key Observations

- The CNN improved test accuracy from 0.8624 (Day 8 ANN) to 0.912, with the same dataset and the same 10 epochs.
- Training accuracy (0.945) is higher than validation (0.917) and test (0.912), so there is a small gap, but validation loss kept falling to the last epoch, so the model is not overfitting badly. In Day 8 the ANN's validation loss started rising after epoch 4.
- The confusion matrix shows where the model struggles. The most common mistakes are Shirt predicted as T-shirt/top (125 images), Shirt as Coat (70) and Shirt as Pullover (68). These classes look alike at 28x28 pixels.
- The incorrect predictions image shows that the model is sometimes confidently wrong, so a high confidence does not guarantee a correct answer.

## Challenges Faced and How They Were Solved

- **Input shape error:** `Conv2D` needs a channel dimension, but Fashion MNIST images are `(28, 28)`. I reshaped the data to `(28, 28, 1)` with `.reshape(-1, 28, 28, 1)`.
- **Overfitting risk:** the training score rises faster than the validation score as epochs increase. I added a Dropout layer (0.3) before the output layer and watched the validation curves.
- **Hard-to-separate classes:** Shirt stayed at 69.6% accuracy. I used the confusion matrix to find which classes it is confused with, and noted that higher resolution, more convolution layers or data augmentation would be the next steps.
- **Setup:** TensorFlow needs 64-bit Python (3.9 to 3.12). I installed it for Python 3.12 and ran the scripts with that interpreter.
- **Slow training on CPU:** about 20 seconds per epoch. I kept the network small and used 10 epochs.

## Concepts

- **Epoch:** one full pass through the training data. **Batch size:** how many images are used per weight update (64 here).
- **Training vs validation data:** the model learns from the training data, and the validation data checks it during training without being learned from.
- **Data augmentation (concept):** creating extra training images by flipping, rotating or zooming existing ones, which reduces overfitting. It was not used in this project.
