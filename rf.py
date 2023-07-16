import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import streamlit as st
import pickle

st.title('Netflix Stock Prediction')

# Load the dataset
df = pd.read_csv(
    'C:/Users/HP/OneDrive/Niesya/project_ds/dataset/NFLX (10).csv')

# data_to_visualization = df
# data_to_visualization['Date'] = pd.DatetimeIndex(df['Date'])
# data_to_visualization.set_index('Date', drop=True, inplace=True)

# # Describing Data
# st.subheader('Data from 2010-2023')
# st.write(df)

# # Visualization
# # st.subheader('Closing Price History')
# # fig = plt.figure(figsize=(16, 8))
# # plt.plot(df['Adj Close'])
# # plt.xlabel('Date', fontsize=18)
# # plt.ylabel('Close Price', fontsize=18)
# # st.pyplot(fig)

# st.title('Netflix Stock Prediction')

# # Load the dataset
# df = pd.read_csv('C:/Users/HP/OneDrive/Niesya/project_ds/dataset/NFLX (10).csv')

# # Convert the 'Date' column to datetime and set it as the index
# df['Date'] = pd.to_datetime(df['Date'])
# df.set_index('Date', inplace=True)

# # Describing Data
# st.subheader('Data from 2010-2023')
# st.write(df)

# # Visualization
# st.subheader('Closing Price History')
# fig = plt.figure(figsize=(16, 8))
# plt.plot(df['Adj Close'])
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price', fontsize=18)
# plt.title('Closing Price History', fontsize=20)
# st.pyplot(fig)

# Convert the 'Date' column to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Describing Data
st.subheader('Data from 2010-2023')
st.write(df)

# Visualization
st.subheader('Closing Price History')
fig = plt.figure(figsize=(16, 8))
plt.plot(df['Adj Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.title('Closing Price History', fontsize=20)
st.pyplot(fig)

# Determine the split percentage (e.g., 80/20)
train_split = 0.8
test_split = 1 - train_split

# Calculate the index that separates the training and test sets
split_index = int(train_split * len(df))

# Split the data into training and test sets
train_data = df[:split_index]
test_data = df[split_index:]

# Print the lengths of the training and test sets
print(f"Training set length: {len(train_data)}")
print(f"Test set length: {len(test_data)}")

# Load model
with open('C:/Users/HP/OneDrive/Niesya/project_ds/Project/randomforest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

last_date = df.index[-1]
start_date = last_date + timedelta(days=1)  # Start date for prediction

# Calculate the end date for prediction (here we predict for the next 30 days from the start date)
end_date = start_date + timedelta(days=30)

# Generate the date range for forecasting
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Make predictions for the selected date range
new_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(
    len(date_range))   # Data baru yang ingin diprediksi
predictions = loaded_model.predict(new_data)

# Create DataFrame for the predicted values
df_predictions = pd.DataFrame(
    {'Date': date_range, 'Predicted_Stock_Price': predictions})

# Create a slider for selecting the forecasting range in months
forecast_range = st.slider(
    'Select Forecasting Range (in months)', min_value=1, max_value=12, value=3)

# Calculate the start and end dates for the forecasting range
start_date = datetime(2023, 7, 1)
# Assuming 30 days per month
end_date = start_date + timedelta(days=(forecast_range * 30))

# Generate the date range for forecasting
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Make predictions for the selected date range
new_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(
    len(date_range))   # Data baru yang ingin diprediksi
predictions = loaded_model.predict(new_data)

# Create DataFrame for the predicted values
df_predictions = pd.DataFrame(
    {'Date': date_range, 'Predicted_Stock_Price': predictions})

# Visualization - Closing Price Prediction
st.subheader(f'Closing Price Prediction for {forecast_range} Months')
fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(df['Adj Close'], label='Historical Closing Price')
plt.plot(df_predictions['Date'], df_predictions['Predicted_Stock_Price'],
         'r', label='Predicted Closing Price')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.title('Netflix Stock Price Prediction', fontsize=20)
plt.legend()
st.pyplot(fig)
