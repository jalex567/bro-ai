import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
try:
    dt_model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    # Load the list of columns the model was trained on
    model_columns = joblib.load("weather_features_encoded_cols.joblib")
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'weather.joblib', 'scaler.joblib', and 'weather_features_encoded_cols.joblib' are in the same directory.")
    st.stop()

# Define the unique weather conditions your model was trained on
# This should be extracted from your data preprocessing step
# For example:
# unique_weather_conditions = data['Weather_Condition'].unique().tolist() # Assuming 'data' was your original DataFrame

# For now, we'll use the columns from the loaded model and reverse engineer the weather conditions
# A better approach would be to save the unique weather conditions during preprocessing
unique_weather_conditions = [col.replace('Weather_Condition_', '') for col in model_columns if col.startswith('Weather_Condition_')]
unique_weather_conditions.insert(0, 'Fair') # Add the base case if it was dropped by drop_first=True

st.title('Accident Severity Prediction based on Weather')

st.write("Enter the current weather conditions to predict the accident severity.")


temperature = st.slider("Temperature (F)", min_value=-45.0, max_value=196.0, step=0.1)
wind_chill = st.slider("Wind_Chill (F)", min_value=-63.0, max_value=196.0, step=0.1)
humidity = st.slider("Humidity (F)", min_value=1.0, max_value=100.0, step=0.1)
pressure = st.slider("Pressure (in)", min_value=0.0, max_value=58.63, step=0.1)
visibility = st.slider("Visibility (mi)", min_value=0.0, max_value=100.0, step=0.1)
wind_speed = st.slider("Wind_Speed (mph)", min_value=0.0, max_value=1087.0, step=0.1)
precipitation = st.slider("Precipitation (in)", min_value=0.0, max_value=24.0, step=0.1)



# Get user input for categorical feature (Weather Condition)
weather_condition = st.selectbox('Weather Condition', unique_weather_conditions)

if st.button('Predict Severity'):
    # Create a DataFrame from user input
    input_data = pd.DataFrame([[temperature, wind_chill, humidity, pressure, visibility, wind_speed, precipitation, weather_condition]],
                               columns=['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                                        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
                                        'Weather_Condition'])

    # Preprocess the input data
    # Apply one-hot encoding to 'Weather_Condition'
    input_data_encoded = pd.get_dummies(input_data, columns=['Weather_Condition'], drop_first=True)

    # Ensure all columns from the training data are present in the input data, filling missing ones with 0
    for col in model_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0

    # Reorder columns to match the training data
    input_data_encoded = input_data_encoded[model_columns]


    # Scale numerical features
    numerical_features = ['Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)',
                          'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']

    input_data_encoded[numerical_features] = scaler.transform(input_data_encoded[numerical_features])


    # Make prediction
    prediction = dt_model.predict(input_data_encoded)

    # Display the prediction
    st.write(f"Predicted Accident Severity: {prediction[0]}")
