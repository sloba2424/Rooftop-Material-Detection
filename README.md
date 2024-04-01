# ES Innovation Sprint Q4 | Rooftop Segmentation

## Overview

The Rooftop Segmentation Project is a computer vision project that focuses on detecting and isolating rooftops in aerial images. The project involves several components, including data preprocessing, model training, and post-processing to extract and save isolated rooftop images. This README provides an overview of the project and its main functionalities.

## Project Structure

The project is organized into several modules and subdirectories:

- `src/`: Contains the source code for the project.
  - `models/`: Contains the model generation code.
  - `utils/`: Contains utility functions for data visualization, image processing, and miscellaneous tasks.
- `data/`: Contains the dataset used for training and testing.
- `isolated_rooftops/`: A directory where isolated rooftop images will be saved.
- `logs/`: A directory where TensorFlow logs will be stored.
- `training_checkpoints/`: A directory for storing model checkpoints during training.

## Main Functionality

The main functionality of the project is implemented in the `main()` function, which can be found in the main script of the project. Here's an overview of what the main function does:

1. Data Loading and Augmentation:
   - Loads image and mask data from the dataset, which consists of aerial images and corresponding masks.
   - Augments the data by applying random rotations, translations, and cropping to create additional training samples.

2. Data Preprocessing:
   - Preprocesses the loaded data by resizing images and masks to a common size.
   - Splits the data into training and testing sets.

3. Model Training:
   - Defines a convolutional neural network (CNN) model for rooftop detection.
   - Compiles and trains the model using the training data.
   - Saves model checkpoints during training.

4. Inference and Post-processing:
   - Uses the trained model to predict masks for the testing data.
   - Converts the predicted masks to binary format.
   - Isolates rooftops from the original images using the binary masks.
   - Saves the isolated rooftop images in the `isolated_rooftops/` directory.

5. Logging and Visualization:
   - Logs useful information about data shapes, training progress, and saved file paths.
   - Optionally, generates plots to visualize training and validation loss, accuracy, and timestamps.

## Usage

To run the Rooftop Detection Project, follow these steps:

1. Clone the repository to your local machine.
2. Ensure you have all the required dependencies installed (Python, TensorFlow, PIL, scikit-image, etc.).
3. Place your dataset in the `data/` directory or update the data loading code accordingly.
4. Run the main script to execute the project: `python main_script.py`.

The isolated rooftop images will be saved in the `isolated_rooftops/` directory, and training logs and checkpoints will be stored in the respective directories.

## Configuration and Customization

You can customize various aspects of the project, such as model architecture, training hyperparameters, and data augmentation techniques, by modifying the source code in the `src/` directory.
