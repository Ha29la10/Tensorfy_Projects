# Sentiment Analysis using Facial Reaction Recognition with CNN

This project implements a sentiment analysis model that recognizes facial reactions from images. It uses a Convolutional Neural Network (CNN) architecture trained from scratch to classify images into 7 different sentiment classes. The project applies data augmentation techniques, model optimization callbacks, and performance evaluation with accuracy and loss plots. The goal is to achieve the best possible accuracy and continuously enhance the model for future improvements.

 

## Overview
This project focuses on sentiment analysis using facial reaction recognition, where a CNN model is built to classify images into 7 sentiment classes. The CNN model is trained on a dataset of facial expressions to detect emotions based on image data. Various training techniques, including data augmentation, model checkpointing, early stopping, and learning rate reduction, are used to optimize the model's performance.

## Dataset
The dataset used for this project includes **28,821 images for training** and **7,066 images for validation**, with each image labeled as one of 7 possible sentiment classes. The images represent various facial expressions, providing a comprehensive dataset for sentiment classification based on facial reactions.

### Classes
The dataset is divided into 7 classes, corresponding to distinct sentiment expressions. Each image in the dataset belongs to one of these classes, allowing the CNN model to learn and predict based on visual features associated with different emotions.

## Data Augmentation
To improve model performance and generalization, data augmentation is applied using `ImageDataGenerator`. This technique generates new image samples by randomly applying transformations to existing images, which reduces overfitting and enhances the model’s ability to generalize to unseen data.

Key transformations include:
- **Rotation**: Slightly rotating images to account for variations in facial orientation.
- **Zooming**: Zooming in on facial features to simulate a variety of facial close-ups.
- **Horizontal Flipping**: Mirroring images horizontally to cover different angles of the face.

 

## Model Architecture
A Convolutional Neural Network (CNN) model is built from scratch, with multiple versions tested to find the optimal architecture. The model includes:
- **Convolutional Layers**: For extracting features from the images.
- **Pooling Layers**: For reducing the dimensionality while retaining important information.
- **Fully Connected Layers**: For classification into 7 sentiment classes.

Hyperparameter tuning is conducted to improve the model’s performance, and different architectures are tried to maximize accuracy.

## Training and Validation
The model is trained on the augmented training dataset with the following optimizations:
- **Model Checkpoint**: Saves the best model during training based on validation accuracy, ensuring the best model parameters are retained.
- **Early Stopping**: Stops training if no improvement is seen in validation accuracy for a defined number of epochs, helping to prevent overfitting.
- **Learning Rate Reduction**: Reduces the learning rate dynamically if validation accuracy plateaus, allowing the model to converge more effectively.

## Evaluation
The model’s performance is evaluated by plotting:
- **Training and Validation Accuracy**: To track model accuracy over epochs and identify overfitting or underfitting.
- **Training and Validation Loss**: To monitor the convergence of the model.

These plots provide insights into model performance and areas for potential improvement.

## Future Work
Future enhancements may include:
- **Improving Model Architecture**: Experimenting with deeper or more complex CNN architectures to improve accuracy.
- **Hyperparameter Tuning**: Testing different batch sizes, learning rates, and optimizer settings for better performance.
- **Advanced Data Augmentation**: Applying additional image transformations like brightness adjustments and image cropping.
- **Transfer Learning**: Incorporating pre-trained models to leverage feature extraction and further improve accuracy.

 
