# p1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the stock price data
data = pd.read_csv("apple_stock.csv")

# Get the closing price of the stock
closing_price = data["Close"]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_closing_price = scaler.fit_transform(closing_price.values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_closing_price) * 0.8)
test_size = len(scaled_closing_price) - train_size

train_data = scaled_closing_price[:train_size]
test_data = scaled_closing_price[train_size:]

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=128, activation="tanh", input_shape=(1, 1)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss="mse", optimizer="adam")

# Train the model
model.fit(train_data, train_data, epochs=100, batch_size=32)

# Predict the test set
predictions = model.predict(test_data)

# Inverse the scaling
predictions = scaler.inverse_transform(predictions)

# Plot the predictions
plt.plot(test_data, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()
