# Diabetes Prediction Neural Network
## Overview
This repository contains a simple neural network implementation using PyTorch to predict diabetes based on various health indicators.

## Data
The dataset (diabetes.csv) contains various health indicators as features and a binary label indicating the presence or absence of diabetes.

## Setup
To run the code in diabetes_nn.ipynb, you'll need the following dependencies:

Python 3.x
PyTorch
pandas
numpy
scikit-learn
Install the required packages using:

pip install torch pandas numpy scikit-learn

# Implementation
## Data Preprocessing
- Loaded the dataset using Pandas.
- Extracted features and labels, converting labels from string to numeric format.
- Performed feature normalization using StandardScaler.
- Created PyTorch tensors for features and labels.
## Neural Network Architecture
- Designed a simple feedforward neural network using nn.Module.
- The network consists of four fully connected layers with Tanh activation functions and a Sigmoid activation in the output layer.
## Training
- Utilized a custom Dataset and DataLoader from PyTorch for batch processing.
- Defined BCELoss as the loss function and SGD optimizer for training.
- Trained the network for 200 epochs, iterating through the training data and updating weights using backpropagation.
## Evaluation
- Calculated accuracy on the training data after each epoch.
- Added an evaluation phase on a separate test dataset after each epoch to monitor test accuracy.
## Results
Throughout training, the model's loss and accuracy were printed after each epoch, providing insights into its performance.

