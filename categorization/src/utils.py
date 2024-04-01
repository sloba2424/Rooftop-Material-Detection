from os.path import exists
from shutil import copy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from constants import *


def visualize_data(data):
    """
    Visualize sample images from the training dataset.

    Args:
        data (keras.utils.data.Dataset): Training dataset.
    """
    class_names = data.class_names

    plt.figure(figsize=(10, 10))
    for images, labels in data.take(1):
        # TODO: fix a constant 9;
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))

            # Find the index of the class with a 1 in the one-hot encoded label
            class_index = np.argmax(labels[i])

            plt.title(class_names[class_index])
            plt.axis("off")
    plt.show()


def visualize_data_augmentation(train_ds, data_augmentation_layers):
    """
    Visualize data augmentation on sample images from the training dataset.

    Args:
        train_ds (keras.utils.data.Dataset): Training dataset.
        data_augmentation_layers: Augmentation layers
    """

    # TODO:
    if not AUGMENT_DATA:
        return

    for images, _ in train_ds.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            augmented_images = data_augmentation(images, data_augmentation_layers)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"))
            plt.axis("off")
    plt.show()


def debug_visualize(data):
    if SHOW_VISUALISATION_IMAGES:
        visualize_data(data)
        visualize_data_augmentation(data, DEFAULT_DATA_AUGMENTATION_LAYERS)


def list_files_in_folder_and_subfolders(folder_path):
    """
    List all files in the specified folder and its subfolders.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list of str: A list of file paths.
    """
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        # Check if the current depth is one level below the given folder_path
        if root == folder_path or root.startswith(folder_path + os.path.sep):
            file_list.extend([os.path.join(root, file) for file in files])

    return file_list


def copy_to_folder(path_copy_from, folder_copy_to, file_name):
    """
    Copy a file to a destination folder, adding a counter to the filename if it already exists.

    Args:
        path_copy_from (str): Path of the file to copy.
        folder_copy_to (str): Destination folder.
        file_name (str): Name of the file to copy.
    """
    counter = 0
    base_name, extension = os.path.splitext(file_name)
    while True:
        file_path_copy_to = os.path.join(folder_copy_to, file_name)
        if not exists(file_path_copy_to):
            copy(path_copy_from, os.path.join(folder_copy_to, file_name))
            break
        counter += 1

        file_name = f"{base_name}_{counter}{extension}"


#
def data_augmentation(
    images: Any, data_augmentation_layers: Any, augment_data: bool = True
):
    """
    Apply data augmentation to a batch of images.

    Args:
        images (numpy.ndarray): Batch of images.
        data_augmentation_layers: Augmentation layers
        augment_data: should data be augmented

    Returns:
        numpy.ndarray: Augmented images.
    """

    if augment_data:
        for layer in data_augmentation_layers:
            images = layer(images)

    return images


# TODO:
# do we really need this one:
def find_latest_epoch():
    """
    Find the latest epoch number from saved model files.

    Returns:
        int or None: Latest epoch number if model files are found, else None.
    """
    model_files = [f for f in os.listdir(MODEL_FOLDER_PATH) if f.endswith(".keras")]

    if not model_files:
        print("No model files found.")
        return None

    epochs = [int(f.split("_")[2].split(".")[0]) for f in model_files]

    return max(epochs)


if __name__ == "__main__":
    pass
