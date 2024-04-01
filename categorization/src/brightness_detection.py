import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_average_roof_brightness(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grapy_flatten = gray.flatten()
    # average_roof_brightness
    if np.sum(grapy_flatten) == 0:
        return 0
    else:
        return np.sum(grapy_flatten) / np.sum(grapy_flatten > 0)


def analyze_brightness_in_folder(folder_path):
    file_paths = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if filename.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    brightness_values = list(map(calculate_average_roof_brightness, file_paths))

    mean_brightness = np.mean(brightness_values)
    median_brightness = np.median(brightness_values)
    std_dev_brightness = np.std(brightness_values)
    number_of_values = len(brightness_values)

    print("-" * 50)
    print(f"Folder: {folder_path.split('/')[-1]}")
    print(f"Number of values analyzed: {number_of_values}")
    print(f"Mean Brightness: {mean_brightness}")
    print(f"Median Brightness: {median_brightness}")
    print(f"Standard Deviation of Brightness: {std_dev_brightness}")
    print(
        f"Standard Deviation / Mean of Brightness: {std_dev_brightness / mean_brightness}"
    )

    plt.hist(brightness_values, bins=30, color="blue", edgecolor="black")
    plt.title(f'Histogram of Brightness Values in {folder_path.split("/")[-1]}')
    plt.xlabel("Brightness")
    plt.ylabel("Frequency")
    plt.show()

    return brightness_values


materials = ["asphalt", "metal", "slate", "tile", "wood"]
base_folder = "../data/material_images/"

for material in materials:
    folder_path = os.path.join(base_folder, material)
    analyze_brightness_in_folder(folder_path)
