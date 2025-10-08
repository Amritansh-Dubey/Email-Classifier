# Email Spam Classifier

This project classifies emails or text messages as spam or not spam using machine learning.  
It applies text preprocessing, vectorization, and classification techniques to identify unwanted or promotional content.

## Project Overview

The objective of this project is to build a simple machine learning model that can detect spam messages based on the text content.  
The model is trained using a labeled dataset containing both spam and non-spam (ham) messages.

## Features

- Loads and processes text data from a labeled dataset  
- Cleans and prepares text by converting to lowercase, removing punctuation, and filtering out stop words  
- Converts text into numerical vectors using TF-IDF representation  
- Trains a Naive Bayes classifier for binary classification (spam or not spam)  
- Evaluates performance using accuracy and a classification report  
- Allows testing on new sample messages

## Dataset

The dataset used is the SMS Spam Collection Dataset from Kaggle, which includes thousands of SMS messages labeled as spam or ham.

## Technologies Used

- Python  
- pandas  
- scikit-learn  
- NumPy

## Sample Output
Accuracy: 0.9738

Classification Report:
              precision    recall  f1-score   support

           0       0.98      1.00      0.99       965
           1       0.97      0.89      0.93       150

    accuracy                           0.97      1115
   macro avg       0.97      0.94      0.96      1115
weighted avg       0.97      0.97      0.97      1115

Message: Congratulations! You've won a free iPhone. Click here to claim now!

Prediction: Spam

Message: Hey, are we still meeting for lunch tomorrow?

Prediction: Not Spam

