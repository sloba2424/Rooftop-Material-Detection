# Categorization Module

This module includes a script (`training.py`) that performs image categorization using a deep learning model with TensorFlow and Keras. The model is trained on a dataset of roof top images (asphalt, metal, slate, tile, wood) and is then used to categorize new images in the 'new_data' folder.

## Requirements

- TensorFlow
- Keras
- Numpy
- Matplotlib

## Script Overview

The script follows the following flow:

1. Import necessary libraries and modules.
2. Configure global constants such as file paths, folders, and hyperparameters.
3. Define utility functions for various tasks such as listing files, checking if a file is an image, generating a dataset, visualizing data, data augmentation, and more.
4. Define the architecture of the deep learning model using Keras.
5. Implement functions for training the model, loading a saved model, and running inference on new data.
6. Execute the main process:
    - If 'USE_EXISTING_MODEL' is True, load the existing model from 'model.h5'.
    - If 'USE_EXISTING_MODEL' is False, generate a new dataset, visualize data, perform data augmentation, configure the dataset for performance, create and train a new model, and save the trained model.
    - Run inference on images in the 'new_data' folder using the trained model, categorize them into respective rooftop material categories, and save results in folders.

## Usage

To use the script, follow the steps:
1. put images for training to the folder "material_images"
2. Put images for the categorization inference in the folder "new_data". Current training and inference images are stored [here](https://drive.google.com/file/d/14T4SplkSssOYP6Z6Z7CdSTH7pQ9wr8OA/view?usp=sharing). 
3. Execute the main module:
```bash
python training.py