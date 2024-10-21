"""
Multivariate Linear Regression Model

Independent Variables: # of attacks (up to Oct 11 24), distance from Al Shifa hospital (up to Oct 11 24)
Dependent Variables: # of injuries (up to Sep 6 24)

Changes:
prints out how many data points (rows in DataFrame)
removes outlier data points (IQR method)
added residual plot

test-train split:
    Training Set: A portion of the dataset used to train the model. (70%)
    Test Set: A separate portion of the dataset used to evaluate the model’s performance after training.(30%)
    residual plot is created for the test predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Use pandas to extract data from excel files (Excel file 1: has # of attacks and # of injuries for certain dates)
attacks_injuries = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\attacks and injuries 2.xlsx')
attack_distances = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\ACLED_OCT_11_UPDATED.xlsx')

# timezone inconsistency between two excel files, remove timezone aspect
attacks_injuries['date'] = pd.to_datetime(attacks_injuries['date']).dt.tz_localize(None)
attack_distances['event_date'] = pd.to_datetime(attack_distances['event_date']).dt.tz_localize(None)

# Renaming columns to variable names that are appropriate
attacks_injuries.rename(columns={'number of attacks': 'number_of_attacks', 'number of injuries': 'number_of_injuries'}, inplace=True)
attack_distances.rename(columns={'Distance (km)': 'Distance_km'}, inplace=True)

# Aggregate distances (between attack locations and Al Shifa hospital) by calculating the mean for each unique data
attack_distances_agg = attack_distances.groupby('event_date').agg({'Distance_km': 'mean'}).reset_index()

# Merge the DataFrames from two Excel files to one
merged_data = pd.merge(attacks_injuries, attack_distances_agg, left_on='date', right_on='event_date')

# Drop rows/dates with missing values
clean_data = merged_data.dropna(subset=['number_of_attacks', 'number_of_injuries'])

# Calculate IQR for number_of_injuries
Q1 = clean_data['number_of_injuries'].quantile(0.25)
Q3 = clean_data['number_of_injuries'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
clean_data = clean_data[(clean_data['number_of_injuries'] >= lower_bound) & (clean_data['number_of_injuries'] <= upper_bound)]

# Print the number of observations
print("Number of observations (days) in clean_data:", len(clean_data))

# Create the multivariate linear regression model with statsmodels
model = smf.ols('number_of_injuries ~ number_of_attacks + Distance_km', data=clean_data).fit()

# Print the model summary: includes equation of plane, R^2, 
print(model.summary())

#%% 3D PLOT

# Create a grid of values for the independent variables
x = np.linspace(clean_data['number_of_attacks'].min(), clean_data['number_of_attacks'].max(), 100)
y = np.linspace(clean_data['Distance_km'].min(), clean_data['Distance_km'].max(), 100)
x, y = np.meshgrid(x, y)

# Create a DataFrame for predictions
pred_data = pd.DataFrame({'number_of_attacks': x.ravel(), 'Distance_km': y.ravel()})

# Predict using the model
pred_data['number_of_injuries'] = model.predict(pred_data)

# Reshape predictions to match the grid
z = pred_data['number_of_injuries'].values.reshape(x.shape)

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)

# Plot the actual data points
ax.scatter(clean_data['number_of_attacks'], clean_data['Distance_km'], clean_data['number_of_injuries'], color='r', s=50, label='Actual data')

# Labels and title
ax.set_xlabel('Number of Attacks')
ax.set_ylabel('Distance to Hospital (km)')
ax.set_zlabel('Number of Injuries')
ax.set_title('3D Multivariate Regression Model')

# Show the legend
ax.legend()

# Show the plot
plt.show()

#%% RESIDUAL PLOT

# Calculate the fitted values and residuals
clean_data['fitted_values'] = model.predict(clean_data)
clean_data['residuals'] = clean_data['number_of_injuries'] - clean_data['fitted_values']

# Create a residual plot
plt.figure(figsize=(10, 6))
plt.scatter(clean_data['fitted_values'], clean_data['residuals'], alpha=0.7)
plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y=0
plt.title('Residual Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.grid()
plt.show()

#%% TRAIN-TEST SPLIT

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create the feature matrix (X) and target variable (y)
X = clean_data[['number_of_attacks', 'Distance_km']]
y = clean_data['number_of_injuries']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the multivariate linear regression model with statsmodels
model = smf.ols('number_of_injuries ~ number_of_attacks + Distance_km', data=clean_data).fit()

# Fit the model to the training data
model_train = smf.ols('number_of_injuries ~ number_of_attacks + Distance_km', data=pd.concat([X_train, y_train], axis=1)).fit()

# Print the model summary
print(model_train.summary())

# Make predictions on the test set
y_pred = model_train.predict(X_test)

# Calculate model performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the performance metrics
print(f'Mean Absolute Error: {mae}') #off by this many injuries
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Create a residual plot for the test predictions
residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y=0
plt.title('Residual Plot')
plt.xlabel('Fitted Values (Predicted)')
plt.ylabel('Residuals')
plt.grid()
plt.show()

#%% PREDICTING RECONSTRUCTIVE SURGERY

# Print the equation of the regression model
intercept = model_train.params['Intercept']
coef_attacks = model_train.params['number_of_attacks']
coef_distance = model_train.params['Distance_km']

print(f"Regression Equation: Number of Injuries = {intercept:.2f} + {coef_attacks:.2f} * Number of Attacks + {coef_distance:.2f} * Distance (km)")

# Predict the number of injuries requiring reconstructive surgery (23.5% of the predicted injuries)
predicted_injuries = model_train.predict(X)
injuries_requiring_surgery = predicted_injuries * 0.235  # 23.5%

# Create a grid for the 3D plot
x = np.linspace(clean_data['number_of_attacks'].min(), clean_data['number_of_attacks'].max(), 100)
y = np.linspace(clean_data['Distance_km'].min(), clean_data['Distance_km'].max(), 100)
x, y = np.meshgrid(x, y)

# Create a DataFrame for predictions
pred_data = pd.DataFrame({'number_of_attacks': x.ravel(), 'Distance_km': y.ravel()})

# Predict injuries requiring surgery using the model
pred_data['predicted_injuries_surgery'] = model_train.predict(pred_data) * 0.235  # 23.5%

# Reshape predictions to match the grid
z = pred_data['predicted_injuries_surgery'].values.reshape(x.shape)

# Create a 3D plot for predicted injuries requiring reconstructive surgery
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)

# Plot the actual data points
ax.scatter(clean_data['number_of_attacks'], clean_data['Distance_km'], injuries_requiring_surgery, color='r', s=50, label='Actual data (Injuries requiring Surgery)')

# Labels and title
ax.set_xlabel('Number of Attacks')
ax.set_ylabel('Distance to Hospital (km)')
ax.set_zlabel('Predicted Injuries Requiring Reconstructive Surgery')
ax.set_title('3D Multivariate Regression Model for Injuries Requiring Surgery')

# Show the legend
ax.legend()

# Show the plot
plt.show()
