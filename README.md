# Flower Species Classifier

A Python package for training an image classifier using transfer learning to recognize different species of flowers. The project includes classes for Binomial and Gaussian distributions, alongside scripts for training and predicting flower species based on image data.

## Overview
The Flower Species Classifier is a Python package designed to train an image classifier using transfer learning techniques. This project aims to recognize different species of flowers from images, making it a valuable tool for botanical studies and applications in horticulture.

## Features
- **Transfer Learning**: Utilizes pre-trained models (e.g., VGG16) to enhance classification accuracy.
- **Data Preprocessing**: Includes scripts for loading and preprocessing datasets.
- **Training and Prediction**: Offers functionality for training the model and making predictions on new images.
- **Statistical Distributions**: Contains classes for Binomial and Gaussian distributions for statistical analysis.

## Dataset
The dataset used for this project is the 102-category flower dataset. You can download the dataset [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) and extract it into the project directory.

## Model Architecture
The project uses a VGG16 network with its classifier replaced by a custom feed-forward neural network with ReLU activation and dropout regularization. The training process includes:
- Data augmentation for the training set.
- Normalization of image data to match ImageNet standards.
- Backpropagation and optimization using the Adam optimizer.

## Results
The model achieves a validation accuracy of 82% on the flower dataset.

## train.py Overview
This script is responsible for training the model using the specified dataset and parameters:

- **Argument Parsing**: Sets up command-line arguments to specify directories, architecture, learning rate, hidden layer size, etc.
- **Data Preparation**: Loads training, validation, and test datasets using ImageFolder and applies transformations.
- **Model Initialization**:
  - Loads a pre-trained model (VGG16 or AlexNet).
  - Modifies the classifier to suit the number of output classes (102).
  - Prepares the model for training on either GPU or CPU.
- **Training Loop**: Defines functions for training and validating the model over a specified number of epochs.
- **Checkpoint Saving**: Saves the best model based on validation loss.

## predict.py Overview
This script is used for making predictions with a trained model:

- **Argument Parsing**: Takes the test image, JSON file for category names, and the checkpoint file as inputs.
- **Model Loading**: Loads the model from a checkpoint.
- **Image Preprocessing**: Prepares the image for model input.
- **Prediction**: Runs the model on the input image and returns the top classes with their probabilities.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- tqdm
- matplotlib
- PIL
- numpy

### Usage
1. Prepare your dataset by placing flower images in the designated folder structure.
2. Modify the `train.py` script to set your parameters (e.g., batch size, learning rate).

### Run the Training Script
To train the model, run the following command:
```bash
python train.py --data_dir <path_to_dataset> --arch vgg16 --epochs 5 --learning_rate 0.001
```
Replace <path_to_dataset> with the path to your flower dataset directory.


**Making Predictions:**

To classify new images, run:
```
python predict.py --image_path path/to/your/image.jpg
```

To make predictions on a new image, run:

```
python predict.py --test_file <path_to_image> --checkpoint_file <path_to_checkpoint> --json_file <path_to_json>
```
Replace <path_to_image>, <path_to_checkpoint>, and <path_to_json> with the respective paths.


### JSON File:

The cat_to_name.json file maps class indices to human-readable flower names. Make sure to include it in your project directory.
