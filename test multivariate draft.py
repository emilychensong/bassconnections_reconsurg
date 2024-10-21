"""
First iteration of Multivariate Linear Regression Model mainly for experimentation purposes.
Independent variables: attacks and injuries (outdated data), attack distances from al shifa hospital (outdated data)
Dependent variable: # of injuries
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Use pandas to extract data from excel files (Excel file 1: has # of attacks and # of injuries for certain dates)
attacks_injuries = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\attacks and injuries.xlsx')
attack_distances = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Attack Distances from Al Shifa.xlsx')

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

# Create the multivariate linear regression model with statsmodels
model = smf.ols('number_of_injuries ~ number_of_attacks + Distance_km', data=clean_data).fit()

# Print the model summary: includes equation of plane, R^2, 
print(model.summary())

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
