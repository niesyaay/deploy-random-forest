from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


st.title('Netflix Stock Prediction')

user_input = st.text_input('Stock Ticker', 'NFLX')
df = pd.read_csv(
    'C:/Users/HP/OneDrive/Niesya/project_ds/dataset/NFLX (10).csv')

# Describing Data
st.subheader('Data from 2010-2023')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


# Create a new dataframe with only the 'Close column
data = df.filter(['Close'])

# Convert the dataframe to a numpy array
dataset = data.values

# Get the number of rows to train the model on
dtrain_len = int(np.ceil(len(dataset) * .80))

# scale

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)


# Load model
model = load_model(
    'C:/Users/HP/OneDrive/Niesya/project_ds/Project/keras_model.h5')

# Testing Part
# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
test_data = scaled_data[dtrain_len - 60:, :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[dtrain_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))


# FINAL GRAPH
# Plot the data
train = data[:dtrain_len]
valid = data[dtrain_len:]
valid['Predictions'] = predictions

# Visualize the data
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(16, 6))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
st.pyplot(fig2)

# Show the valid and predicted prices
st.subheader('Closing Price Prediction')
valid
