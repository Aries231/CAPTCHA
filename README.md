# CAPTCHA Recognition Project

## Overview

This project implements a deep learning model to solve CAPTCHA images using PyTorch. The model is trained on a dataset of CAPTCHA images, each containing 5 characters composed of uppercase letters and numbers.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- PIL

## Function Description

- **`CaptchaNet`:** The neural network model for CAPTCHA recognition.

- **`CaptchasDataset`:** Dataset class for loading and processing CAPTCHA images.

- **`Captcha`:** Predict characters from new CAPTCHA images.

- **`main`:** Main function for training, validating, and running inference.

## Model

The **`CaptchaNet`** class defines a convolutional neural network (CNN) with dropout layers to prevent overfitting. It predicts 5 characters from each CAPTCHA image.

## Data Augmentation

The training dataset undergoes augmentation (color jitter, Gaussian noise) to improve model generalization.

  
## Training and Validation

Training data is loaded with augmentations.

Validation data is used to evaluate model performance.

The model with the highest validation accuracy is saved.

## Inference

**`Captcha`** class for loading a trained model and performing inference.

The model predicts the text in a given CAPTCHA image.

## Usage

Train the model using **`main`**.

Perform inference using the **`Captcha class`**.

## Note

Adjust the file paths and model parameters according to your setup.

Ensure the dataset path is correctly set in **`main`**.

This guide provides an overview of the project. For detailed usage, refer to comments in the code.
