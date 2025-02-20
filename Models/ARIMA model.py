# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 15:45:43 2024

@author: emily
"""

# Ensure the necessary packages are installed
# You can install statsmodels by running `pip install statsmodels` if not already installed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the Excel file with injury data
file_path = r"C:\Users\emily\OneDrive - Duke University\Bass Connections\test 2 daily injuries.xlsx"
data = pd.read_excel(file_path)

# Convert the Date column to datetime type if not already
data['Date'] = pd.to_datetime(data['Date'], format="%m/%d")

# Interpolating missing injury data
data['Injuries'] = data['Injuries'].interpolate(method='linear')

# Visualize the data (optional)
print(data.head())

# Split the data into training (80%) and testing (20%) sets
train_size = int(len(data) * 0.8)
train_data = data['Injuries'][:train_size]
test_data = data['Injuries'][train_size:]

# Define the ARIMA model (adjust the order (p,d,q) based on data)
# Initial order is (1,1,1) but you can use AIC/BIC to choose a better model
arima_order = (1, 1, 1)  # (p, d, q)

# Fit the ARIMA model on the training data
arima_model = ARIMA(train_data, order=arima_order)
arima_result = arima_model.fit()

# Print model summary
print(arima_result.summary())

# Forecast injuries for the test period
forecast = arima_result.forecast(steps=len(test_data))

# Evaluate the model with RMSE (Root Mean Squared Error)
rmse = np.sqrt(mean_squared_error(test_data, forecast))
print(f'RMSE: {rmse}')

# Combine train, test, and forecast data for plotting
train_dates = data['Date'][:train_size]
test_dates = data['Date'][train_size:]

# Plot the historical data, train, test, and forecast
plt.figure(figsize=(10, 6))
plt.plot(train_dates, train_data, label='Training Data', color='blue')
plt.plot(test_dates, test_data, label='Test Data', color='orange')
plt.plot(test_dates, forecast, label='Forecasted Injuries', color='green', linestyle='dashed')
plt.title('Injury Forecast Using ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Number of Injuries')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Future Forecast for n_days (e.g., 90 days)
n_days = 90
future_forecast = arima_result.forecast(steps=n_days)

# Create future dates for the forecast
forecast_dates = pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=n_days)

# Plot future forecast
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Injuries'], label='Historical Injuries', color='blue')
plt.plot(forecast_dates, future_forecast, label='Projected Injuries', color='red', linestyle='dashed')
plt.title('Future Injury Forecast Using ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Number of Injuries')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
