import os

import cv2
import numpy as np

from src.utils.upsampling import ImageUpscaler

# Data directory paths
DATA_DIRECTORY = os.path.abspath(os.getcwd())

MASK_IMGS_DIR = os.path.join(DATA_DIRECTORY, "data_1000", "mask_imgs_1000")
TRAIN_IMGS_DIR = os.path.join(DATA_DIRECTORY, "data_1000", "train_imgs_1000")
OUTPUT_PATH = os.path.join(DATA_DIRECTORY, "new_images")

COUNT = 0

IMG_HEIGHT = 300
IMG_WIDTH = 300


def find_paired_images(original_images_names, mask_images_names):
    names = []
    for original_image_name in original_images_names:
        has_mask_image = [s for s in mask_images_names if original_image_name[:-3] in s]
        if len(has_mask_image) == 1:
            mask_image_name = has_mask_image[0]
            names.append((original_image_name, mask_image_name))
    return names


def crop_and_save_roof_with_black_background(original_image, mask, image_suffix, i):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(original_image, original_image, mask=mask)

    cv2.imwrite(f"{OUTPUT_PATH}/object_{image_suffix}_{i}.png", masked)


def crop_roof_into_rectangle(contour, image, image_suffix, i):
    x, y, w, h = cv2.boundingRect(contour)
    crop_img = image[y:y + h, x:x + w]

    upscaler = ImageUpscaler("./FSRCNN_x4.pb")
    resized = upscaler.upscale_image(crop_img)

    cv2.imwrite(f"{OUTPUT_PATH}/object_{image_suffix}_{i}.png", resized)


def slice_rooftop_with_black_background(original_image_path, mask_image_path):
    original_image = cv2.imread(original_image_path)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_background = np.zeros_like(original_image)
    global COUNT
    # Loop through isolated rooftop masks
    for contour in contours:
        # Draw the contour on the black background
        cv2.drawContours(black_background, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
        COUNT += 1

        # If you need to save the mask as well, uncomment the following line
        # cv2.imwrite(f"{OUTPUT_PATH}/mask_{image_path[-7:-4]}_{i}.png", black_background)

        crop_and_save_roof_with_black_background(original_image, black_background, image_path[-7:-4], COUNT)

        # Reset the black background for the next contour
        black_background = np.zeros_like(original_image)


def slice_image(original_image_path, mask_image_path):
    original_image = cv2.imread(original_image_path)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    global COUNT
    # Loop through isolated rooftop masks
    for contour in contours:
        COUNT += 1
        crop_roof_into_rectangle(contour, original_image, image_path[-7:-4], COUNT)


if __name__ == "__main__":
    original_images_names = os.listdir(TRAIN_IMGS_DIR)
    mask_images_names = os.listdir(MASK_IMGS_DIR)

    names = find_paired_images(original_images_names, mask_images_names)

    for image_name, mask_name in names:
        image_path = os.path.join(TRAIN_IMGS_DIR, image_name)
        mask_path = os.path.join(MASK_IMGS_DIR, mask_name)

        slice_image(image_path, mask_path)