# Car Price Prediction Project

## Overview

This project aims to predict the prices of used cars based on various features using a machine learning model. The project includes data cleaning, feature engineering, model training, and evaluation. The final model is a Linear Regression model implemented using scikit-learn.

## Features

- **Name**: The name of the car.
- **Company**: The manufacturer of the car.
- **Year**: The manufacturing year of the car.
- **Kms Driven**: The distance the car has been driven in kilometers.
- **Fuel Type**: The type of fuel the car uses (Petrol, Diesel, CNG, etc.).

## Data Cleaning

The dataset is cleaned to ensure that:
- All years are numeric.
- Prices are numeric and valid.
- Kilometers driven are numeric and valid.
- Fuel types are valid and non-missing.
- Car names are preserved without truncation.

## Model Training

The data is split into training and testing sets. The training set is used to train the Linear Regression model, and the testing set is used to evaluate its performance.

## Dependencies

- Python 3.12.4
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage

1. Clone the repository.
2. Install the required dependencies.
3. Run the data cleaning script to prepare the data.
4. Train the model using the training script.
5. Use the model to make predictions on new data.

## Files

- **quikr_car.csv**: The raw dataset.
- **Cleaned_Car_data.csv**: The cleaned dataset.
- **model_training.py**: Script to clean the data and train the model.
- **app.py**: Flask/Django application to serve the model and make predictions.

## Contributing

Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.
