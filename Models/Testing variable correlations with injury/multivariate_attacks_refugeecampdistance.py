import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from geopy.distance import geodesic


# Load data
attacks_injuries_df = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\attacks and injuries 2.xlsx')
acled_data = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\ACLED_OCT_11_UPDATED.xlsx')
refugee_camps_df = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\refugee camp.xlsx')  

# Process dates and timezone
attacks_injuries_df['date'] = pd.to_datetime(attacks_injuries_df['date']).dt.tz_localize(None)
acled_data['event_date'] = pd.to_datetime(acled_data['event_date']).dt.tz_localize(None)
acled_data.rename(columns={'event_date': 'date'}, inplace=True)

# Calculate mean distance to closest refugee camp for each attack event
def find_closest_camp(attack_coords):
    distances = refugee_camps_df.apply(
        lambda row: geodesic(attack_coords, (row['latitude'], row['longitude'])).kilometers, axis=1
    )
    return distances.min()

acled_data['closest_camp_distance'] = acled_data.apply(
    lambda row: find_closest_camp((row['latitude'], row['longitude'])), axis=1
)

# Aggregate distances and merge data
aggregated_distances = acled_data.groupby('date')['closest_camp_distance'].mean().reset_index()
aggregated_distances.columns = ['date', 'mean_distance']

merged_df = pd.merge(aggregated_distances, attacks_injuries_df, on='date', how='inner')
merged_df.rename(columns={'number of attacks': 'number_of_attacks', 'number of injuries': 'number_of_injuries'}, inplace=True)

# Drop NA values and filter outliers
merged_df.dropna(subset=['number_of_attacks', 'mean_distance', 'number_of_injuries'], inplace=True)
Q1, Q3 = merged_df['number_of_injuries'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_df = merged_df[(merged_df['number_of_injuries'] >= lower_bound) & (merged_df['number_of_injuries'] <= upper_bound)]

# Fit the multivariate linear regression model
X = filtered_df[['number_of_attacks', 'mean_distance']]
X = sm.add_constant(X)
y = filtered_df['number_of_injuries']
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

# Generate 3D plot data
x_vals = np.linspace(filtered_df['number_of_attacks'].min(), filtered_df['number_of_attacks'].max(), 100)
y_vals = np.linspace(filtered_df['mean_distance'].min(), filtered_df['mean_distance'].max(), 100)
x_vals, y_vals = np.meshgrid(x_vals, y_vals)

# Create DataFrame for predictions
predict_df = pd.DataFrame({
    'const': 1,
    'number_of_attacks': x_vals.ravel(),
    'mean_distance': y_vals.ravel()
})
z_vals = model.predict(predict_df).values.reshape(x_vals.shape)

# Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot for regression plane
ax.plot_surface(x_vals, y_vals, z_vals, color='skyblue', alpha=0.5, rstride=100, cstride=100)

# Scatter plot for actual data points
ax.scatter(filtered_df['number_of_attacks'], filtered_df['mean_distance'], filtered_df['number_of_injuries'], color='r', s=50)

# Set labels
ax.set_xlabel('Number of Attacks')
ax.set_ylabel('Mean Distance to Refugee Camp (km)')
ax.set_zlabel('Number of Injuries')
ax.set_title('3D Multivariate Regression: Attacks & Distance vs. Injuries')

plt.show()

# Residual Plot
predicted_injuries = model.predict(X)
residuals = y - predicted_injuries

plt.figure(figsize=(10, 6))
plt.scatter(predicted_injuries, residuals, color='blue', alpha=0.6)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.xlabel('Predicted Number of Injuries')
plt.ylabel('Residuals')
plt.title('Residual Plot for Multivariate Linear Regression Model')
plt.show()