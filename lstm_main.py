import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. Fetch and Prepare Data
ticker = "NVDA"
data = yf.download(ticker, period="5y", interval="1d") # LSTMs need a lot of data
df = data[['Close']].ffill()

# Scale data to (0, 1) - Crucial for Neural Networks
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# 2. Create the "Sliding Window" (X=Last 60 days, y=Next day)
prediction_days = 60
x_train, y_train = [], []

for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i-prediction_days:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
# Reshape for LSTM [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 3. Build the LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
    Dropout(0.2), # Prevents overfitting
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25), # Fully connected layer
    Dense(units=1)  # Final Output (The predicted price)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=10) # 10 epochs for demo purposes

# 4. Forecast the Next 30 Days (Iterative Prediction)
last_60_days = scaled_data[-60:]
current_batch = last_60_days.reshape((1, prediction_days, 1))
future_predictions = []

for i in range(30):
    # Predict the next day
    current_pred = model.predict(current_batch)[0]
    future_predictions.append(current_pred)
    # Update batch: remove first day, add newest prediction
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Inverse transform to get actual prices
future_predictions = scaler.inverse_transform(future_predictions)

# 5. Plot the result
plt.figure(figsize=(10, 5))
plt.plot(df.index[-100:], df['Close'][-100:], label='Historical')
# Generate future dates
future_dates = pd.date_range(start=df.index[-1], periods=30, freq='B')
plt.plot(future_dates, future_predictions, label='LSTM Forecast', color='green')
plt.title(f'{ticker} Deep Learning Forecast')
plt.legend()
plt.savefig('lstm_forecast.png')
plt.show()
