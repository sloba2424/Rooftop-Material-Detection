import os

from keras import layers

DEBUG_FLAG = False

USE_EXISTING_MODEL = False
# Put here the model path, e.g. "../model/put_folder_here/put_keras_file_name_here.keras" :
MODEL_TO_USE = "../model/put_folder_here/put_keras_file_name_here.keras"
SHOW_VISUALISATION_IMAGES = True

AUGMENT_DATA = True
"""
How many augmented images should be added to training dataset for each original image:
"""
AUGMENTATION_SIZE = 9

MODEL_FOLDER_PATH = "../model"
MODEL_SUBFOLDER_TEMPLATE = "%Y%m%d-%H%M%S"
DATA_FOLDER = "../data/material_images/"
NEW_DATA_FOLDER = "../data/new_data"
RESULTS_FOLDER = "../results"

CHECKPOINT_TEMPLATE = "save_at_{epoch}.keras"

ASPHALT = "asphalt"
METAL = "metal"
SLATE = "slate"
TILE = "tile"
WOOD = "wood"

ASPHALT_RESULTS_FOLDER = os.path.join(RESULTS_FOLDER, ASPHALT)
METAL_RESULTS_FOLDER = os.path.join(RESULTS_FOLDER, METAL)
SLATE_RESULTS_FOLDER = os.path.join(RESULTS_FOLDER, SLATE)
TILE_RESULTS_FOLDER = os.path.join(RESULTS_FOLDER, TILE)
WOOD_RESULTS_FOLDER = os.path.join(RESULTS_FOLDER, WOOD)

FOLDER_LIST = [
    ASPHALT_RESULTS_FOLDER,
    METAL_RESULTS_FOLDER,
    SLATE_RESULTS_FOLDER,
    TILE_RESULTS_FOLDER,
    WOOD_RESULTS_FOLDER,
]

CLASS_NAMES = [ASPHALT, METAL, SLATE, TILE, WOOD]
NUM_CLASSES = len(CLASS_NAMES)

"""
Set the epoch number. For testing purposes, epoch number should be decreased.
"""
NUMBER_OF_EPOCHS = 20

DEFAULT_DATA_AUGMENTATION_LAYERS = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation((-0.5, 0.5)),
    layers.RandomZoom(-0.2, 0.2),
]

DESIRED_HEIGHT, DESIRED_WIDTH = 180, 180
IMAGE_SIZE = (180, 180)
BATCH_SIZE = 128

RANDOM_SEED = 1000
