# Sentiment Analysis using LSTM

This project implements a sentiment analysis model using a Long Short-Term Memory (LSTM) network to classify text data. It includes essential data preprocessing steps, such as tokenization and padding, to prepare text data effectively for model training and evaluation. The model achieves **91.2% accuracy on validation data** and **91.6% accuracy on test data**.

 

## Overview
This project focuses on building a sentiment analysis model to classify text as either positive or negative. An LSTM model is used due to its effectiveness in handling sequential data like text. The preprocessing pipeline consists of essential techniques like tokenization and padding, which help transform raw text into a structured format suitable for input into the model.

## Dataset
The dataset used for this project is the **Amazon Reviews Dataset**, which includes customer reviews from Amazon across various product categories (e.g., electronics, books, clothing, home goods). This dataset is a robust resource for sentiment analysis as it provides extensive textual data and associated review ratings.

Each review entry typically contains:
- **Review Text**: The content of the customer review.
- **label**: A score, usually on a 1 or 2 star scale, indicating the sentiment.
 
Given the dataset's size (over 130 million reviews in full versions), a subset is used in this project for efficient processing and manageable computation.

## Data Preprocessing
Data preprocessing is essential for preparing raw text data for input into the LSTM model. The following steps are applied:

1. **Tokenization**: Each text is broken down into individual words or tokens. These tokens are then converted into numeric indices, representing each word as an integer.
2. **Padding**: After tokenization, each sequence is padded to ensure uniform length across all samples, essential for batch processing in the model while retaining sequence information.

## Model Architecture
The model is built using an LSTM layer that processes sequential data effectively by capturing dependencies over various time steps. Additional layers, such as dense layers, are added for better classification performance. The architecture has been tuned to achieve high accuracy on both validation and test data.

## Training and Validation
The model is trained using the preprocessed data, with the training set used for optimization and the validation set for monitoring the model's performance. Hyperparameters such as batch size, learning rate, and number of epochs are tuned to improve the model’s accuracy and generalization.

## Evaluation
After training, the model is evaluated on a separate test dataset to gauge its performance on unseen data. The model achieves an accuracy of **91.2% on validation data** and **91.6% on test data**, indicating strong generalization and robustness.

## Results
- **Validation Accuracy**: 91.2%
- **Test Accuracy**: 91.6%

The results demonstrate the model’s effectiveness in performing sentiment analysis, with high accuracy on both validation and test sets.

 
