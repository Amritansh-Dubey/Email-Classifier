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

## Project Structure

