# Anthracnose Disease in Chilli Plant Detection (Using deep learning model to find the plants affected with anthracnose disease.)
## Overview
This project aims to build a deep learning model using the ResNet-50 architecture to identify whether a chili plant is affected by anthracnose disease. Anthracnose is a fungal disease that can significantly impact crop yield. By training our model on a custom dataset, we can provide early detection and help farmers take timely action to prevent further spread of the disease.

## Dataset
The dataset was downloaded from kaggle dataset.
link: https://www.kaggle.com/datasets/prudhvi143413s/anthracnose-disease-in-chilli-mobile-captured

## Model Architecture
I chose the ResNet-50 architecture due to its proven performance in image classification tasks. ResNet-50 is a deep convolutional neural network (CNN) with 50 layers, which allows it to learn complex features from the input images.

## Training
Data Preprocessing: I resized all images to a consistent size and normalized pixel values.
Model Initialization: I imported the pre-trained ResNet-50 model (with weights trained on ImageNet) and removed the top classification layer.
Custom Classification Layer: I added a new fully connected layer with two output classes: “Healthy” and “Anthracnose.”
Loss Function: I used Cross Entropy Loss as our criterion to optimize the model during training.
Optimizer: I employed the Adam optimizer for gradient-based weight updates.

## Training Process
We split our dataset into training, validation, and test sets.
During training, we fine-tuned the ResNet-50 model on our chili plant images.
We monitored training loss and validation accuracy to prevent overfitting.
After convergence, we evaluated the model on the test set.

## Augmentation
To improve model robustness, we augmented our dataset by applying random transformations (e.g., rotation, flipping, brightness adjustments) to the training images. Augmentation helps the model generalize better to unseen data.

## Model Evaluation
We assessed the model’s performance using metrics such as accuracy. Additionally, we visualized the confusion matrix to understand false positives and false negatives.
