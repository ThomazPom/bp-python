import json

import tools
from tools import create_dataset

"""
## Setup
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle

"""
## Prepare the data
"""

# Model / data parameters
num_classes = 11
greyscale = False
generated_images = 200
batch_size = 64
epochs = 10

# the data, split between train and test sets
# print(x_train.shape)
(x_train, y_train, y_train_labels, y_train_class_indices) = create_dataset(generated_images, greyscale, num_classes)
(x_test, y_test, a, a) = create_dataset(generated_images, greyscale, num_classes)

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train, y_train, y_train_labels = shuffle(x_train, y_train, y_train_labels)
# x_test, y_test = shuffle(x_test, y_test)

json.dump({
    "class_orders": y_train_class_indices
}, open("config.json", "w"))

# print(y_train_labels[0:15])
# tools.plot_figures(
#     [
#         {
#             str(label): image
#         }
#         for label, image in zip(y_train_labels[0:15], x_train[0:15])
#     ], 5)

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


if greyscale:
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

input_shape = x_train[0].shape

print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model: keras.Sequential = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

model.save("model.h5")
