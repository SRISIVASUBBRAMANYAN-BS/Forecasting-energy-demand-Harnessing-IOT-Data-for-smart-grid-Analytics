import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('Dataset.csv', dtype='str')  # Specifying dtype as 'str' for all columns

# Convert Time column to datetime 
df['time'] = pd.to_datetime(df['time'], errors='coerce')

# Drop rows with missing timestamps
df.dropna(subset=['time'], inplace=True)

# Select relevant features for prediction
features = ['temperature', 'humidity', 'visibility', 'apparentTemperature', 'pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint', 'precipProbability']

# Function to train and predict
def train_and_predict(features, target, future_date):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    future_prediction = model.predict(future_date)
    return future_prediction

# Streamlit app
def main():
    st.title('IOT Grid: Forecasting Energy Demand with IoT Data')

    # Date input box
    st.subheader('Select Future Date')
    month = st.number_input('Month', min_value=1, max_value=12, value=6)
    day = st.number_input('Day', min_value=1, max_value=31, value=15)
    hour = st.number_input('Hour', min_value=0, max_value=23, value=12)
    minute = st.number_input('Minute', min_value=0, max_value=59, value=0)

    future_date = pd.DataFrame({'temperature': [25], 'humidity': [50], 'visibility': [10], 
                                'apparentTemperature': [24], 'pressure': [1010], 'windSpeed': [10], 
                                'cloudCover': [0], 'windBearing': [180], 'precipIntensity': [0], 
                                'dewPoint': [15], 'precipProbability': [0], 'Month': [month], 
                                'Day': [day], 'Hour': [hour], 'Minute': [minute]})

    if st.button('Predict'):
        future_prediction = train_and_predict(features, 'House overall [kW]', future_date)

        # Display predicted value
        st.subheader('Predicted Electricity Consumption')
        st.write(f'Predicted Electricity Consumption: {future_prediction[0]:.2f} kW')

if __name__ == '__main__':
    main()
