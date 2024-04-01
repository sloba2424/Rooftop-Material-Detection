import logging
import os
import sys

import numpy as np
from PIL import Image

from src.utils.image_utils import (
    generate_image_mask_dataset,
    ndarray_to_image,
)
from src.utils.upsampling import ImageUpscaler

sys.path.append("../models/")

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration global variables
RANDOM_SEED = 1000
RANDOM_STATE = 0

IMG_WIDTH = 300
IMG_HEIGHT = 300
IMG_CHANNELS = 3

EPOCHS = 1
BATCH_SIZE = 32
DEBUG_MODE = True
MAX_LEN_DATA = 25000

RUNTIME_MODEL = "runtime_models/model_best"
ISOLATED_ROOFTOPS_DIR = "./isolated_rooftops/"

# Define the paths for generated images and masks
SOURCE_IMAGE_DIR = 'data/generated_data/generated_images'
SOURCE_MASK_DIR = 'data/generated_data/generated_masks'


def load_or_generate_data():
    # Check if the generated files exist in the filesystem
    if os.path.exists(SOURCE_IMAGE_DIR) and os.path.exists(SOURCE_MASK_DIR):
        image_files = [os.path.join(SOURCE_IMAGE_DIR, filename) for filename in sorted(os.listdir(SOURCE_IMAGE_DIR))]
        mask_files = [os.path.join(SOURCE_MASK_DIR, filename) for filename in sorted(os.listdir(SOURCE_MASK_DIR))]

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
        os.makedirs(SOURCE_IMAGE_DIR, exist_ok=True)
        os.makedirs(SOURCE_MASK_DIR, exist_ok=True)
    return image_matrices, data_segmentation_masks


def save_final_images(image_matrices: np.ndarray, data_segmentation_masks: np.ndarray):
    for i, (image_array, mask_array) in enumerate(zip(image_matrices, data_segmentation_masks)):
        image_filename = os.path.join(SOURCE_IMAGE_DIR, f'image_{i + 1:04d}.jpg')

        upscaler = ImageUpscaler("./models/FSRCNN_x4.pb")
        image_array, mask_array = upscaler.upscale_image_and_mask(image_array, mask_array)

        image = ndarray_to_image(image_array * mask_array)
        image.save(image_filename)


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


def main():
    np.random.seed = RANDOM_SEED
    image_matrices, data_segmentation_masks = load_or_generate_data()
    save_final_images(image_matrices, data_segmentation_masks)


if __name__ == "__main__":
    main()
