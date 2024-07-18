from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer                   #machine learning model
from sklearn.pipeline import Pipeline


def home(request):
    return render(request, 'home.html')


def result(request):
    if request.method == 'POST':
        # Get input data from the form
        name = request.POST.get('name')
        company = request.POST.get('company')
        year = int(request.POST.get('year'))
        kms_driven = int(request.POST.get('kms_driven').replace(',', '').replace(' km', ''))
        fuel_type = request.POST.get('fuel_type')

        # Load and clean the data
        car = pd.read_csv("quikr_car.csv")
        car = car[car['year'].str.isnumeric()]
        car['year'] = car['year'].astype(int)
        car = car[car['Price'] != 'Ask For Price']
        car['Price'] = car['Price'].str.replace(',', '').astype(int)
        car['kms_driven'] = car['kms_driven'].str.replace(' km', '').str.replace(',', '').str.replace('s', '')
        car['kms_driven'] = pd.to_numeric(car['kms_driven'], errors='coerce')
        car = car.dropna(subset=['kms_driven'])
        car = car[~car['fuel_type'].isna()]
        car['name'] = car['name'].str.split().str.slice(start=0, stop=3).str.join(' ')
        car = car.reset_index(drop=True)

        # Define features and target variable
        X = car[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
        Y = car['Price']

        # Split the data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        # Create the pipeline
        column_trans = ColumnTransformer(
            transformers=[
                ('name_company_fuel', OneHotEncoder(handle_unknown='ignore'), ['name', 'company', 'fuel_type'])
            ],
            remainder='passthrough'
        )

        pipe = Pipeline(steps=[
            ('preprocessor', column_trans),
            ('regressor', LinearRegression())
        ])

        # Fit the pipeline
        pipe.fit(X_train, Y_train)

        # Prepare the input data as a DataFrame
        input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

        # Make prediction
        prediction = pipe.predict(input_data)

        return render(request, 'result.html', {'prediction': prediction[0]})
