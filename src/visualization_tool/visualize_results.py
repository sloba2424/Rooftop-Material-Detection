import cv2
import matplotlib.pyplot as plt
import os
import random

# Data direcory paths
DATA_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir, "data"))

MASK_IMGS_DIR = os.path.join(DATA_DIRECTORY, "data_1000", "mask_imgs_1000")
TRAIN_IMGS_DIR = os.path.join(DATA_DIRECTORY, "data_1000", "train_imgs_1000")

MATERIALS_DIR = os.path.join(DATA_DIRECTORY, "rooftop_material_examples")


def visualize_results(original_image_path, mask_image_path, material):
    original_image = cv2.imread(original_image_path)
    mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    _, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.set_title("Original")
    ax1.imshow(original_image)
    ax1.axis('off')

    _, thresh = cv2.threshold(mask, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(original_image, contours, -1, color=(0, 255, 0), thickness=2)

    ax2.set_title("Result")
    ax2.imshow(original_image)
    ax2.axis('off')

    ax3.set_title("Material")
    ax3.imshow(cv2.cvtColor(material, cv2.COLOR_BGR2RGB))
    ax3.axis('off')

    plt.show()


if __name__ == "__main__":
    original_images_names = os.listdir(TRAIN_IMGS_DIR)
    mask_images_names = os.listdir(MASK_IMGS_DIR)

    materials_names = [im_name for im_name in os.listdir(MATERIALS_DIR) if not im_name.startswith(".")]
    materials = []
    for m_name in materials_names:
        materials.append(cv2.imread(os.path.join(MATERIALS_DIR, m_name), cv2.IMREAD_COLOR))

    names = []
    for original_image_name in original_images_names:
        has_mask_image = [s for s in mask_images_names if original_image_name[:-3] in s]
        if len(has_mask_image) == 1:
            mask_image_name = has_mask_image[0]
            names.append((original_image_name, mask_image_name))

    image_name, mask_name = random.choice(names)
    image_path = os.path.join(TRAIN_IMGS_DIR, image_name)
    mask_path = os.path.join(MASK_IMGS_DIR, mask_name)

    visualize_results(image_path, mask_path, random.choice(materials))

