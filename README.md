# Flower Classification using CNN

This project implements a flower classification model using classic convolutional neural network (CNN) architectures. The model is trained to classify 102 different types of flowers based on images. Pre-trained models like ResNet, VGG, and Inception are fine-tuned to achieve high accuracy in this image classification task.

## Project Overview

This project utilizes transfer learning with pre-trained deep learning models for flower classification. The training process involves fine-tuning pre-trained CNN models to classify flowers into 102 categories. The goal is to train a model that can recognize various flower species with high accuracy, even when faced with new images.

### Features:
- Image preprocessing and augmentation (random cropping, flipping, etc.)
- Training and fine-tuning of classic CNN architectures (ResNet, VGG, Inception)
- Validation and testing accuracy tracking
- Save and load the best-performing model
- Predict flower species for new images

## Dataset

The dataset contains 102 different categories of flowers, each category having multiple images. The images are processed and standardized before being fed into the network.

- **Training set**: Used to train the model.
- **Validation set**: Used to validate the model during training.
- **Test set**: Used to evaluate the final performance of the model.

### Preprocessing:
- Resizing images to the required input size of the model.
- Normalizing pixel values to [0, 1] range based on mean and standard deviation.
- Applying data augmentation techniques for better generalization.

## Model Architecture

We use transfer learning by leveraging pre-trained CNN architectures like:
- **ResNet**
- **VGG**
- **Inception**

The models are modified for our task by adjusting the final fully connected layers to output 102 flower categories.

## Training

### Steps:
1. Load the pre-trained model (ResNet, VGG, etc.).
2. Freeze the initial layers and fine-tune only the final layers.
3. Set up a loss function (cross-entropy loss) and an optimizer (e.g., Adam).
4. Apply a learning rate scheduler to dynamically adjust the learning rate.
5. Train the model using the training dataset, with periodic evaluation on the validation dataset.
6. Save the model with the best validation accuracy.

### Hyperparameters:
- **Optimizer**: Adam or SGD
- **Learning rate**: Dynamic adjustment with a scheduler
- **Loss function**: Cross-entropy loss

## How to Use

1. Clone the repository:

2. Install dependencies:
pip install -r requirements.txt
3. Train the model:
Modify the configuration in main.py to set your dataset path and model parameters.
python main.py
4. Evaluate the model:
python evaluate.py
5. Predict with the trained model:
You can use the saved model for prediction on new images.
Results
![img.png](img.png)