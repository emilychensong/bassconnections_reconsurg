# -*- coding: utf-8 -*-
"""

@author: emily
"""

# Ensure the necessary packages are installed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d

# Load the Excel file with injury data
file_path = r"C:\Users\emily\OneDrive - Duke University\Bass Connections\test 2 daily injuries.xlsx"
data = pd.read_excel(file_path)

# Convert the Date column to datetime type if not already
data['Date'] = pd.to_datetime(data['Date'], format="%m/%d")

# Visualize the data (optional)
print(data.head())

# Interpolating missing injury data
data['Injuries'] = data['Injuries'].interpolate(method='linear')

# Check for NA values in Injuries column
na_count = data['Injuries'].isna().sum()
if na_count > 0:
    print(f"There are {na_count} NA values in the Injuries column after interpolation.")
else:
    print("No NA values in the Injuries column after interpolation.")

# Define parameters for the simulation
n_days = 300  # Number of days to project (3 months)
n_simulations = 10000  # Number of Monte Carlo simulations
mean_injuries = data['Injuries'].mean()
sd_injuries = data['Injuries'].std()
daily_change = 0.00  # Assume a daily percentage change in injuries (adjust as needed)

# Function to simulate injuries based on historical mean and daily change
def simulate_injuries(n_days, mean_injuries, sd_injuries, daily_change):
    simulated = np.zeros(n_days)
    simulated[0] = mean_injuries  # Start with the mean value
    for i in range(1, n_days):
        simulated[i] = simulated[i-1] * (1 + daily_change) + np.random.normal(0, sd_injuries)
    return simulated

# Run Monte Carlo simulations
np.random.seed(123)  # Ensure reproducibility
simulation_results = np.array([simulate_injuries(n_days, mean_injuries, sd_injuries, daily_change) for _ in range(n_simulations)])

# Calculate average simulation result (or median) across all simulations
average_simulated = simulation_results.mean(axis=0)

# Create future dates for the projection
forecast_dates = pd.date_range(start=data['Date'].max() + pd.Timedelta(days=1), periods=n_days, freq='D')

# Create a DataFrame for the simulation results
simulation_df = pd.DataFrame({'Date': forecast_dates, 'Injuries': average_simulated})

# Plot the historical data and projections
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Injuries'], label='Historical Injuries', color='blue', linewidth=2)
plt.plot(simulation_df['Date'], simulation_df['Injuries'], label='Projected Injuries', color='red', linestyle='dashed', linewidth=2)
plt.title('Historical and Projected Future Injuries in Gaza')
plt.xlabel('Date')
plt.ylabel('Number of Injuries')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
