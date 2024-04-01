import logging
import os
import sys

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from src.models.models import generate_model_one
from src.utils.data_visualization_utils import plot_results
from src.utils.image_utils import (
    generate_image_mask_dataset,
    ndarray_to_image,
    preprocess_image_matrices_and_image_masks,
)
from src.utils.misc_utils import get_utc_timestamp_formatted

sys.path.append("../models/")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration global variables
RANDOM_SEED = 1000
RANDOM_STATE = 0

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

EPOCHS = 1
BATCH_SIZE = 32
DEBUG_MODE = True
MAX_LEN_DATA = 25000

RUNTIME_MODEL = "runtime_models/model_best"
ISOLATED_ROOFTOPS_DIR = "./isolated_rooftops/"


def load_or_generate_data():
    # Define the paths for generated images and masks
    image_dir = 'data/generated_data/generated_images'
    mask_dir = 'data/generated_data/generated_masks'

    # Check if the generated files exist in the filesystem
    if os.path.exists(image_dir) and os.path.exists(mask_dir):
        image_files = [os.path.join(image_dir, filename) for filename in sorted(os.listdir(image_dir))]
        mask_files = [os.path.join(mask_dir, filename) for filename in sorted(os.listdir(mask_dir))]

        # Load the existing images and masks
        image_matrices = [np.array(Image.open(filename)) for filename in image_files]
        data_segmentation_masks = [np.array(Image.open(filename)) for filename in mask_files]
        return image_matrices, data_segmentation_masks
    else:
        # Generate the data if the files don't exist
        (
            image_matrices,
            data_segmentation_masks,
            data_file_names,
        ) = generate_image_mask_dataset(debug_mode=DEBUG_MODE)

        # Save the generated images and masks
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i, (image_array, mask_array) in enumerate(zip(image_matrices, data_segmentation_masks)):
            image_filename = os.path.join(image_dir, f'image_{i + 1:04d}.jpg')
            mask_filename = os.path.join(mask_dir, f'mask_{i + 1:04d}.jpg')

            image = ndarray_to_image(image_array)
            image.save(image_filename)

            mask = ndarray_to_image(mask_array)
            mask.save(mask_filename)

        return image_matrices, data_segmentation_masks


def save_isolated_rooftops(test_data, binary_masks):
    # Create a directory to save the isolated rooftops
    isolated_rooftops_dir = ISOLATED_ROOFTOPS_DIR

    if not os.path.exists(isolated_rooftops_dir):
        os.makedirs(isolated_rooftops_dir)
    # Iterate through the test set to create and save isolated rooftops
    for i, (original_img_matrix, binary_mask_matrix) in enumerate(
            zip(test_data, binary_masks)
    ):
        # Isolate the rooftop in the original image
        original_img_matrix = np.squeeze(original_img_matrix)
        isolated_rooftop_matrix = original_img_matrix * np.repeat(
            binary_mask_matrix, 3, axis=2
        )

        # Convert the isolated rooftop matrix to a PIL Image
        isolated_rooftop_image: Image.Image = ndarray_to_image(isolated_rooftop_matrix)

        # Save the isolated rooftop as a new image
        save_path = f"./isolated_rooftops/isolated_rooftop_{i}.jpg"
        isolated_rooftop_image.save(save_path)
        logger.info(f"Saved isolated rooftop {i} to {save_path}")


def get_models_path(models_dir: str = "models") -> str:
    # Create a 'models' subdirectory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    # Updated path for model checkpoint to be saved in 'models' subdirectory
    path_checkpoint = os.path.join(models_dir, "cp.ckpt")
    return path_checkpoint


def get_predictions(test_data, model, utc_timestamp, percentile: None = 50):
    loss = model.history.history["loss"]
    val_loss = model.history.history["val_loss"]
    accuracy = model.history.history["accuracy"]
    val_accuracy = model.history.history["val_accuracy"]
    # After training, use the model to predict masks on your test set
    predicted_masks = model.predict(test_data)
    mask_avg_perc = np.percentile(predicted_masks, percentile)

    # Convert these predicted masks to binary format
    binary_masks = (predicted_masks > mask_avg_perc).astype(np.uint8)
    plot_results(loss, val_loss, accuracy, val_accuracy, utc_timestamp)
    save_isolated_rooftops(test_data, binary_masks)


def main():
    np.random.seed = RANDOM_SEED
    utc_timestamp = get_utc_timestamp_formatted()

    data_images, data_segmentation_masks = load_or_generate_data()
    X, y = preprocess_image_matrices_and_image_masks(data_images, data_segmentation_masks)
    logger.info(
        "data_images[0] shape: {}, data_segmentation_masks[0] shape: {}".format(
            X.shape, y.shape
        )
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    logger.info(
        "X_train.shape: {}, y_train.shape: {}".format(X_train.shape, y_train.shape)
    )
    logger.info("X_test.shape: {}, y_test.shape: {}".format(X_test.shape, y_test.shape))

    path_checkpoint = get_models_path()

    # Initialize the model
    roof_detection_model = generate_model_one(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    )
    roof_detection_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor="val_loss"),
        tf.keras.callbacks.TensorBoard(log_dir="./logs"),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=path_checkpoint, save_weights_only=True, verbose=1
        ),
    ]

    # Check if checkpoint exists and load weights
    if os.path.exists(path_checkpoint):
        logger.info("Runtime model found found. Loading whole model.")
        runtime_model_path = RUNTIME_MODEL
        directory_checkpoint = os.path.dirname(runtime_model_path)
        roof_detection_model.load(directory_checkpoint)
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        roof_detection_model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
        )
        runtime_model_path = RUNTIME_MODEL
        directory_checkpoint = os.path.dirname(runtime_model_path)
        roof_detection_model.save(directory_checkpoint)

    get_predictions(
        test_data=X_test,
        model=roof_detection_model,
        utc_timestamp=utc_timestamp,
    )


if __name__ == "__main__":
    main()
