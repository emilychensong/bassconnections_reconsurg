import pandas as pd
import numpy as np
from geopy.distance import geodesic
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Step 1: Load the data from Excel files
refugee_camps_df = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\refugee camp.xlsx')  
attacks_df = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\ACLED_OCT_11_UPDATED.xlsx')  
injuries_df = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\attacks and injuries 2.xlsx')  

# Convert dates and remove timezone inconsistencies
injuries_df['date'] = pd.to_datetime(injuries_df['date']).dt.tz_localize(None)
attacks_df['event_date'] = pd.to_datetime(attacks_df['event_date']).dt.tz_localize(None)

# Rename 'event_date' column in attacks_df to 'date' for consistency
attacks_df.rename(columns={'event_date': 'date'}, inplace=True)

# Step 2: Calculate distance from each attack to the closest refugee camp
def find_closest_camp(attack_coords):
    distances = refugee_camps_df.apply(
        lambda row: geodesic(attack_coords, (row['latitude'], row['longitude'])).kilometers, axis=1
    )
    return distances.min()

# Apply the function to find the closest camp distance for each attack
attacks_df['closest_camp_distance'] = attacks_df.apply(
    lambda row: find_closest_camp((row['latitude'], row['longitude'])), axis=1
)

# Step 3: Aggregate distances by date
aggregated_distances = attacks_df.groupby('date')['closest_camp_distance'].mean().reset_index()
aggregated_distances.columns = ['date', 'mean_distance']

# Step 4: Merge with injury data
merged_df = pd.merge(aggregated_distances, injuries_df, on='date', how='inner')

# Check for missing values and drop them
merged_df.dropna(subset=['mean_distance', 'number of injuries'], inplace=True)

# Step 5: Remove outliers using the IQR method for 'number of injuries'
Q1 = merged_df['number of injuries'].quantile(0.25)
Q3 = merged_df['number of injuries'].quantile(0.75)
IQR = Q3 - Q1

# Determine lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the merged_df to remove outliers
filtered_df = merged_df[(merged_df['number of injuries'] >= lower_bound) & (merged_df['number of injuries'] <= upper_bound)]

# Step 6: Set up the linear regression model
X = filtered_df['mean_distance']  
y = filtered_df['number of injuries']  

# Add a constant for intercept in regression
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()
results_summary = model.summary()

# Print the summary of the regression analysis
print(results_summary)

# Step 7: Plotting the linear regression
plt.figure(figsize=(10, 6))

# Scatter plot of filtered data points
plt.scatter(filtered_df['mean_distance'], filtered_df['number of injuries'], color='blue', label='Data Points')

# Plotting the regression line
predicted_y = model.predict(X)
plt.plot(filtered_df['mean_distance'], predicted_y, color='red', label='Regression Line', linewidth=2)

# Adding titles and labels
plt.title('Linear Regression: Mean Distance vs. Number of Injuries (Outliers Removed)')
plt.xlabel('Mean Distance to Closest Refugee Camp (km)')
plt.ylabel('Number of Injuries')
plt.legend()
plt.grid()

# Show the plot
plt.show()

# Create a residual plot
residuals = y - predicted_y

plt.figure(figsize=(10, 6))
plt.scatter(predicted_y, residuals, color='green', label='Residuals')
plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at 0
plt.title('Residual Plot')
plt.xlabel('Predicted Number of Injuries')
plt.ylabel('Residuals')
plt.legend()
plt.grid()

# Show the residual plot
plt.show()

