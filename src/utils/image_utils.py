import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.ndimage import zoom
from skimage.transform import resize
from tqdm import tqdm

IMG_WIDTH = 300
IMG_HEIGHT = 300
NUMBER_OF_DATA_POINTS = 200
RANDOM_SEED = 1000
IMG_CHANNELS = 3

EPOCHS = 1
BATCH_SIZE = 32
DEBUG_MODE = True
MAX_LEN_DATA = 25000

PLOT_FLAG = False


def format_image(img, out_shape=(IMG_HEIGHT, IMG_WIDTH)):
    """Resize and format an image."""
    return resize(img, out_shape, mode="constant", preserve_range=True)


def rotate_ndarray_random_90(img, mask):
    """
    Rotates a given ndarray by a random multiple of 90 degrees.

    Parameters:
    arr (ndarray): The array to rotate.

    Returns:
    ndarray: The rotated array.
    """
    # Generate a random number between 0 and 3
    num_rotations = np.random.randint(1, 4)

    # Rotate the array by 90 degrees clockwise, num_rotations times
    return np.rot90(img, k=-num_rotations), np.rot90(mask, k=-num_rotations)


def random_crop_zoom_image_and_mask(image, mask, max_zoom=2):
    """
    Crops into a ndarray representing an image and its mask by a random small amount and
    then resizes them back to the original dimensions using supersampling methods. The
    operation is performed in parallel on both the image and the mask.

    Parameters:
    image (ndarray): The image array.
    mask (ndarray): The mask array.
    max_zoom (float): The maximum zoom factor.

    Returns:
    tuple: The cropped and resized image and mask arrays.
    """
    # Ensure the max_zoom is at least 1.0
    max_zoom = max(max_zoom, 1.0)

    # Calculate a random zoom factor between 1 and max_zoom
    zoom_factor = np.random.uniform(1.1, max_zoom)

    # Calculate the size of the cropped image and mask
    new_height = int(image.shape[0] / zoom_factor)
    new_width = int(image.shape[1] / zoom_factor)

    # Calculate the starting point of the crop
    start_y = np.random.randint(0, image.shape[0] - new_height + 1)
    start_x = np.random.randint(0, image.shape[1] - new_width + 1)

    # Crop the image and mask
    cropped_image = image[start_y: start_y + new_height, start_x: start_x + new_width]
    cropped_mask = mask[start_y: start_y + new_height, start_x: start_x + new_width]

    # Resize (zoom) back to original dimensions
    resized_image = zoom(
        cropped_image,
        (
            image.shape[0] / cropped_image.shape[0],
            image.shape[1] / cropped_image.shape[1],
            1,
        ),
    )

    # Resize the mask to 300x300 with linear interpolation
    resized_mask = resize(cropped_mask, (300, 300))

    return resized_image, resized_mask


def augment_image_and_mask(image_array: np.ndarray, mask_array: np.ndarray):
    """
    Augment an image and its corresponding mask.

    Parameters:
    image (np.ndarray): The original image to be augmented.
    mask (np.ndarray): The bitmask corresponding to roofs in the image.

    Returns:
    np.ndarray: Augmented image.
    np.ndarray: Augmented mask.
    """

    # Ensure the mask and image are compatible
    assert (
            image_array.shape[0:2] == mask_array.shape[0:2]
    ), "Image and mask must have the same dimensions"

    if mask_array.shape != (300, 300, 1):
        mask_array = np.expand_dims(mask_array, axis=-1)
    # Rotate image and mask
    rotated_image, rotated_mask = rotate_ndarray_random_90(image_array, mask_array)

    # Randomly adjust brightness
    augmented_image = rotated_image
    augmented_mask = rotated_mask

    augmented_image, augmented_mask = random_crop_zoom_image_and_mask(
        augmented_image, augmented_mask
    )

    return augmented_image, augmented_mask


def generate_image_mask_dataset(
        debug_mode: bool = False,
        dir_suffix: str = str(NUMBER_OF_DATA_POINTS),
        min_data_points: int = NUMBER_OF_DATA_POINTS,
        augmentation_factor: int = 2,
):
    """Generate image and mask dataset with augmentation if needed."""

    folder_masks, folder_train, _path = get_folders_paths()

    # Count the number of existing data points
    existing_data_points = 0

    data_images = []
    data_segmentation_masks = []
    data_file_names = []

    file_names = os.listdir(os.path.join(_path, folder_train))
    # Loop through the existing data
    for cnt, file_name in enumerate(file_names):
        arr_image, arr_segmentation_mask = get_arrs_img_seg(
            file_name, folder_masks, folder_train, _path
        )
        if (arr_image is not None) and (arr_segmentation_mask is not None):
            data_images.append(arr_image)
            data_segmentation_masks.append(arr_segmentation_mask)
            data_file_names.append(file_name)

        if cnt % 100 == 0:
            logging.info(f'cnt:{cnt}')
        if debug_mode and cnt > 2000:
            break

    generated_images = len(data_images)

    # Check if you have at least min_data_points data points
    if generated_images >= min_data_points:
        return (
            data_images[0:min_data_points],
            data_segmentation_masks[0:min_data_points],
            data_file_names,
        )

    # Calculate how many additional data points are needed
    total_data_points_needed = min_data_points
    additional_data_points_needed = total_data_points_needed - generated_images

    # Generate additional data points through augmentation
    with tqdm(total=additional_data_points_needed, desc="Augmenting Data") as pbar:
        while generated_images < total_data_points_needed:
            (
                aug_data_images,
                aug_data_file_names,
                aug_data_segmentation_masks,
            ) = augment_images(
                additional_data_points_needed,
                data_file_names,
                data_images,
                data_segmentation_masks,
                pbar,
            )
            # Update list of images for segmentation training
            generated_images += len(aug_data_images)
            data_images.extend(aug_data_images)
            data_segmentation_masks.extend(aug_data_segmentation_masks)
            data_file_names.extend(aug_data_file_names)

    for i, msk in enumerate(data_segmentation_masks):
        if data_segmentation_masks[i].shape != (300, 300, 1):
            data_segmentation_masks[i] = np.expand_dims(data_segmentation_masks[i], axis=-1)
    return (
        data_images[:NUMBER_OF_DATA_POINTS],
        data_segmentation_masks[:NUMBER_OF_DATA_POINTS],
        data_file_names,
    )


def augment_images(
        additional_data_points_needed,
        data_file_names,
        data_images,
        data_segmentation_masks,
        pbar=None,
):
    aug_data_images = []
    aug_data_file_names = []
    aug_data_segmentation_masks = []
    for cnt, image in enumerate(data_images):
        mask = data_segmentation_masks[cnt]

        augmented_image, augmented_mask = augment_image_and_mask(image, mask)
        aug_data_images.extend([augmented_image])
        aug_data_segmentation_masks.extend([augmented_mask])
        aug_data_file_names.extend([data_file_names[cnt]] * len([augmented_image]))

        if pbar:
            pbar.update(len([augmented_image]))
        if cnt >= additional_data_points_needed:
            break
        return aug_data_images, aug_data_file_names, aug_data_segmentation_masks


def get_arrs_img_seg(file_name, folder_masks, folder_train, path):
    """get arrays of an image and its segmentation"""

    file_path = os.path.join(path, folder_train, file_name)
    file_name = file_name[7:]
    mask_file_name = "{}_{}".format("mask", file_name)
    mask_file_path = get_files_paths(folder_masks, mask_file_name, path)

    arr_image, arr_segmentation_mask = None, None
    if os.path.isfile(file_path) and os.path.isfile(mask_file_path):
        arr_image = np.asarray(Image.open(file_path))
        arr_segmentation_mask = np.asarray(Image.open(mask_file_path))
        arr_segmentation_mask = arr_segmentation_mask > 100

    return arr_image, arr_segmentation_mask


def get_files_paths(folder_masks, mask_file_name, path):
    mask_file_path = os.path.join(path, folder_masks, mask_file_name)
    return mask_file_path


def get_folders_paths():
    # Get the parent directory of the script
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Construct the path to the data folder dynamically
    path = os.path.join(parent_dir, "data", "data_{}".format(1000))
    # Define folder names for train images and masks
    folder_train = "train_imgs_{}".format(1000)
    folder_masks = "mask_imgs_{}".format(1000)
    return folder_masks, folder_train, path


# noinspection PyTestUnpassedFixture
def resize_normalize_img(input_img):
    """Resize and normalize an image using TensorFlow (AWS version)."""
    arr = np.array(input_img)
    return tf.image.resize((tf.cast(arr, tf.float32) / 255.0), (128, 128))


# noinspection PyTestUnpassedFixture
def mask_matrix_binarization(input_image, threshold=60, threshold_diff=128):
    """Transform and format an image into a boolean mask with a separate dimension."""
    # Define the target colors
    target_color_1 = np.array([60, 0, 83])
    target_color_2 = np.array([188, 0, 173])

    # Calculate the Euclidean distances to the target colors
    distance_1 = np.linalg.norm(input_image - target_color_1, axis=-1)
    distance_2 = np.linalg.norm(input_image - target_color_2, axis=-1)

    # Create a boolean mask based on the threshold
    boolean_mask = np.abs(distance_1 - distance_2) < threshold_diff

    # Convert boolean mask to a tensor of 0s and 1s
    boolean_mask = tf.convert_to_tensor(boolean_mask, dtype=tf.uint8)

    # Add a third dimension with a value of 1
    boolean_mask = tf.expand_dims(boolean_mask, axis=-1)

    # Perform morphological dilation to connect regions
    boolean_mask = tf.image.resize(
        boolean_mask,
        (128, 128),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    return boolean_mask


def preprocess_image_matrices_and_image_masks(data_images, data_segmentation_masks):
    """Prepare (X, y) data for training from images and masks."""
    len_data = NUMBER_OF_DATA_POINTS

    image_matrices = np.zeros((len_data, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
    mask_matrices = np.zeros((len_data, IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
    logging.debug("X.shape: {}, y.shape: {}".format(image_matrices.shape, mask_matrices.shape))

    for i, img in enumerate(data_images):
        image_matrices[i] = 255 * resize_normalize_img(img).numpy()

    for i, img in enumerate(data_segmentation_masks):
        mask_matrices[i] = mask_matrix_binarization(img).numpy()

    return image_matrices, mask_matrices


def ndarray_to_image(ndarray):
    """Convert a numpy ndarray to a PIL Image."""
    return Image.fromarray(np.uint8(ndarray))


def convert_mask_to_color(mask):
    """
    Convert a binary mask with True and False values to a 3-channel image with 0 and 255 values.

    Args:
        mask (numpy.ndarray): Binary mask as a numpy array with shape (height, width, 1).

    Returns:
        Color mask as a numpy array with shape (height, width, 3).
    """
    # Duplicate the single-channel mask into three channels
    color_mask = np.stack([mask, mask, mask], axis=0)

    # Convert True/False values to 255 and 0
    color_mask = np.where(color_mask, 0, 255).astype(np.uint8)

    return color_mask
