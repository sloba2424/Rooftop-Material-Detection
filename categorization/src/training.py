"""
This script performs image categorization using a deep learning model with TensorFlow and Keras.
It trains a model on a dataset of rooftop material images (asphalt, metal, slate, tile, wood),
and then uses the trained model to categorize new images in the 'new_data' folder.

Requirements:
- TensorFlow
- Keras
- Numpy
- Matplotlib

Script Flow:
1. Import necessary libraries and modules.
2. Configure global constants such as file paths, folders, and hyperparameters.
3. Define utility functions for listing files, checking if a file is an image,
   generating a dataset, visualizing data, data augmentation, and more.
4. Define the architecture of the deep learning model using Keras.
5. Implement functions for training the model, loading a saved model,
   and running inference on new data.
6. Execute the main process:
    a. If 'USE_EXISTING_MODEL' is True, load the existing model from 'model.h5'.
    b. If 'USE_EXISTING_MODEL' is False, generate a new dataset, visualize data,
       perform data augmentation, configure the dataset for performance,
       create and train a new model, and save the trained model.
    c. Run categorization inference on images in the 'new_data' folder using the trained model,
       categorize them into respective rooftop material categories, and save results in folders.
"""

from datetime import datetime
import argparse

import keras
import numpy as np
from models import get_model, load_keras_model
from utils import (
    copy_to_folder,
    data_augmentation,
    debug_visualize,
    list_files_in_folder_and_subfolders,
)
from tensorflow import data as tf_data

from constants import *


def generate_dataset(directory):
    """
    Generate a dataset for training and validation.

    Returns:
        keras.utils.data.Dataset: Training and validation dataset.
    """

    return keras.utils.image_dataset_from_directory(
        directory,
        validation_split=0.2,
        subset="both",
        seed=RANDOM_SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
    )


def standardize_data(
    train_ds,
    data_augmentation_layers,
    num_augmented: int = 8,
):
    """
    Apply data augmentation to the entire training dataset and include both original and augmented images.

    Args:
        train_ds (keras.utils.data.Dataset): Training dataset.
        data_augmentation_layers: Augmentation layers.
        num_augmented (int): Number of augmented images to generate per original image.

    Returns:
        keras.utils.data.Dataset: Augmented training dataset.
    """

    augmented_images_list = []

    for images, labels in train_ds:
        augmented_images = images.numpy()
        for _ in range(num_augmented):
            augmented_images_list.append(
                data_augmentation(images, data_augmentation_layers).numpy()
            )

        augmented_images_list.append(augmented_images)

    augmented_images_list = np.concatenate(augmented_images_list, axis=0)
    labels_list = np.concatenate([labels.numpy()] * (num_augmented + 1), axis=0)

    augmented_train_ds = tf_data.Dataset.from_tensor_slices(
        (augmented_images_list[: len(labels_list)], labels_list)
    )
    # the set size is of (1 + num_augmented) of the original one `train_ds`
    return augmented_train_ds


def configure_dataset_for_performance(train_ds, val_ds, data_augmentation_layers):
    """
    Configure datasets for better performance during training - apply `data_augmentation` to the training images.

    Args:
        train_ds (keras.utils.data.Dataset): Training dataset.
        val_ds (keras.utils.data.Dataset): Validation dataset.
        data_augmentation_layers: Augmentation layers

    Returns:
        tuple: Configured training and validation datasets.
    """
    train_ds = train_ds.map(
        lambda img, label: (data_augmentation(img, data_augmentation_layers), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda img, label: (data_augmentation(img, data_augmentation_layers), label),
        num_parallel_calls=tf_data.AUTOTUNE,
    )
    # Prefetching samples in GPU memory helps maximize GPU utilization.
    train_ds = train_ds.prefetch(tf_data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf_data.AUTOTUNE)
    return train_ds, val_ds


def train_model(
    model,
    train_ds,
    val_ds,
    model_save_path,
    args,
):
    """
    Train the given model on the provided training dataset and save checkpoints.

    Args:
        model (keras.Model): Keras model to be trained.
        train_ds (keras.utils.data.Dataset): Training dataset.
        val_ds (keras.utils.data.Dataset): Validation dataset.
        model_save_path (str): Path to save the trained model and checkpoints.
    """

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(model_save_path, CHECKPOINT_TEMPLATE)
        ),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=args.learning_rate,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
        ),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )
    model.fit(
        train_ds,
        epochs=NUMBER_OF_EPOCHS,
        callbacks=callbacks,
        validation_data=val_ds,
    )


def run_inference(
    model,
    folder_list,
    test_folder,
):
    """
    Run inference on new images in the 'new_data' folder using the provided model.

    Args:
        model (keras.Model): Trained Keras model.
    """

    # Use map to create folders if they don't exist
    list(
        map(
            lambda folder: os.makedirs(folder) if not os.path.exists(folder) else None,
            folder_list,
        )
    )

    for fpath in list_files_in_folder_and_subfolders(test_folder):
        fname = os.path.basename(fpath)
        if not fname.lower().endswith(("jpg", "png", "gif")):
            continue

        probs = get_predictions(fpath, model)
        max_index = np.argmax(probs)

        print(f"Image '{fname}' categorized as: ")
        results = [
            f"{roof.split('/')[-1]}: {100 * prob:.4f}%"
            for (roof, prob) in zip(folder_list, probs)
        ]
        print(" ".join(results))

        list(
            map(
                lambda folder: copy_to_folder(fpath, folder, fname)
                if max_index == folder_list.index(folder)
                else None,
                folder_list,
            )
        )


def get_predictions(fpath, model):
    img = keras.utils.load_img(fpath, target_size=(DESIRED_HEIGHT, DESIRED_WIDTH))
    img_array = keras.utils.img_to_array(img)
    img_array = keras.backend.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    logits = predictions[0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1)
    return probabilities


def process_data(directory: str, augmentation_layers: list):
    train_ds, val_ds = generate_dataset(directory=directory)

    # before
    if DEBUG_FLAG:
        debug_visualize(data=train_ds)

    augmented_train_ds = standardize_data(
        train_ds, augmentation_layers, AUGMENTATION_SIZE
    )

    train_ds, val_ds = configure_dataset_for_performance(
        augmented_train_ds, val_ds, augmentation_layers
    )

    # TODO: implement visualization for processed data

    return train_ds, val_ds


def main(file_path: str, args):
    """
    Main execution function that loads or trains a model and runs inference on new data.
    """

    if USE_EXISTING_MODEL:
        model = load_keras_model(file_path=file_path)
    else:
        model = get_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES)

    # keras.utils.plot_model(model, show_shapes=True)
    print(model.summary())
    print("*" * 80)

    train_ds, val_ds = process_data(
        directory=DATA_FOLDER, augmentation_layers=DEFAULT_DATA_AUGMENTATION_LAYERS
    )

    model_path = os.path.join(
        MODEL_FOLDER_PATH, datetime.now().strftime(MODEL_SUBFOLDER_TEMPLATE)
    )
    train_model(
        model,
        train_ds.batch(BATCH_SIZE),
        val_ds,
        model_path,
        args,
    )
    run_inference(
        model=model,
        folder_list=FOLDER_LIST,
        test_folder=NEW_DATA_FOLDER,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--beta_1", type=float, default=0.9, help="Beta_1 for Adam optimizer"
    )
    parser.add_argument(
        "--beta_2", type=float, default=0.999, help="Beta_2 for Adam optimizer"
    )
    args = parser.parse_args()

    print(
        f"Running with learning_rate={args.learning_rate}, beta1={args.beta_1}, beta2={args.beta_2}"
    )

    main(file_path=MODEL_TO_USE, args=args)
