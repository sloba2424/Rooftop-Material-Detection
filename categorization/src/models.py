import os
from typing import List

import keras
from keras import layers

from constants import *


def load_keras_model(file_path):
    """
    Load a Keras model from a specific file path.

    Args:
        file_path (str): The file path to the Keras model.

    Returns:
        keras.Model or None: Loaded Keras model if it exists, else None.
    """
    if not os.path.exists(file_path):
        print(f"Model file {file_path} does not exist.")
        return None

    try:
        model = keras.models.load_model(file_path)
        print(f"Loaded Keras model from {file_path}")
        return model
    except Exception as e:
        print(f"Error loading Keras model: {e}")
        return None


def get_model(
    input_shape: tuple,
    num_classes: int,
    hidden_dims: List[int] = [256, 512, 728],
) -> keras.Model:
    """
    Create a convolutional neural network (CNN) model for image categorization.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        hidden_dims (List[int]): List of hidden layers dimenstions.

    Returns:
        keras.Model: CNN model.
    """
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in hidden_dims:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.25)(x)

    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation=None)(x)
    return keras.Model(inputs, outputs)
