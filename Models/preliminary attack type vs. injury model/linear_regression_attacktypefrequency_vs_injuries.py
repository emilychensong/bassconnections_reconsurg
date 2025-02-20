import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the updated ACLED data with dates
acled_df = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\ACLED_OCT_11_UPDATED.xlsx')

# Load the injury data
attack_injury_df = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\attacks and injuries 2.xlsx')

# Convert the event_date to datetime format if it's not already
acled_df['event_date'] = pd.to_datetime(acled_df['event_date'])

# Extract month and year for grouping (if needed)
acled_df['month'] = acled_df['event_date'].dt.to_period('M')  # Create a new column for year-month

# Define a mapping of attack types to broader categories
attack_type_mapping = {
    'Air/drone strike': 'Air Attacks',
    'Armed clash': 'Ground Attacks',
    'Attack': 'Ground Attacks',
    'Mob violence': 'Civil Unrest',
    'Looting/property destruction': 'Property Crimes',
    'Remote explosive/landmine/IED': 'Ground Attacks',
    'Shelling/artillery/missile attack': 'Air Attacks',
    'Violent demonstration': 'Civil Unrest',
    'Other': 'Other',
    'Peaceful protest': 'Civil Unrest',
    'Disrupted weapons use': 'Civil Unrest',
    'Grenade': 'Ground Attacks',
    'Change to group/activity': 'Other',
    'Excessive force against protesters': 'Civil Unrest',
    'Arrests': 'Civil Unrest',
    'Protest with intervention': 'Civil Unrest',
    'Abduction/forced disappearance': 'Civil Unrest',
    'Sexual violence': 'Civil Unrest',
}

# Map the categories to a new column in the DataFrame
acled_df['attack_category'] = acled_df['sub_event_type'].map(attack_type_mapping)

# Step 1: Aggregate the number of attacks per day and by type of attack
attack_counts_per_day = acled_df.groupby(['event_date', 'attack_category']).size().reset_index(name='attack_count')

# Step 2: Pivot the table to have attack categories as columns (each attack type becomes a column)
attack_counts_pivot = attack_counts_per_day.pivot(index='event_date', columns='attack_category', values='attack_count').fillna(0).reset_index()

# Rename the 'date' column in attacks_injury_df to match 'event_date' in acled_df
attack_injury_df.rename(columns={'date': 'event_date'}, inplace=True)

# Convert 'event_date' in acled_df to remove timezone
acled_df['event_date'] = pd.to_datetime(acled_df['event_date']).dt.tz_localize(None)

# Convert 'event_date' in attack_injury_df to ensure it's timezone-naive
attack_injury_df['event_date'] = pd.to_datetime(attack_injury_df['event_date']).dt.tz_localize(None)

# Now merge based on the standardized 'event_date'
merged_data = pd.merge(attack_counts_pivot, attack_injury_df, on='event_date', how='inner')

# Rename columns to replace spaces with underscores
merged_data.columns = merged_data.columns.str.replace(' ', '_')

# Step 3: Clean the merged data by dropping rows where the 'number of injuries' column is missing
merged_data_cleaned = merged_data.dropna(subset=['number_of_injuries'])

# Step 4: Reorder columns if needed (move the 'event_date' and 'number_of_injuries' to the front)
final_data = merged_data_cleaned[['event_date', 'number_of_injuries'] + [col for col in merged_data_cleaned.columns if col not in ['event_date', 'number_of_injuries']]]

# Remove the time part from the 'event_date' column (keep only the date)
final_data['event_date'] = final_data['event_date'].dt.date

# Print the final table to verify
print(final_data)

# Optionally, save the final data to a new Excel file for analysis or visualization
final_data.to_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\final_attack_injury_table.xlsx', index=False)

# Apply the IQR method to remove outliers from the 'number_of_injuries' column

# Calculate the Q1 (25th percentile) and Q3 (75th percentile) for 'number_of_injuries'
Q1 = final_data['number_of_injuries'].quantile(0.25)
Q3 = final_data['number_of_injuries'].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the rows where 'number_of_injuries' is outside the IQR bounds
final_data_cleaned = final_data[(final_data['number_of_injuries'] >= lower_bound) & (final_data['number_of_injuries'] <= upper_bound)]

# Print the number of rows before and after removing outliers
print(f"Original data points: {len(final_data)}")
print(f"Data points after removing outliers: {len(final_data_cleaned)}")

# Prepare the model using the cleaned data
X_cleaned = final_data_cleaned[['Air_Attacks', 'Civil_Unrest', 'Ground_Attacks', 'Other', 'Property_Crimes', 'number_of_attacks']]
y_cleaned = final_data_cleaned['number_of_injuries']

# Add a constant (intercept) to the model
X_cleaned = sm.add_constant(X_cleaned)

# Fit a linear regression model
model_cleaned = sm.OLS(y_cleaned, X_cleaned).fit()

# Display the model summary
print(model_cleaned.summary())

# Calculate the predicted values from the cleaned model
y_pred_cleaned = model_cleaned.predict(X_cleaned)

# Calculate the residuals
residuals_cleaned = y_cleaned - y_pred_cleaned

# Plot the residuals
plt.figure(figsize=(8, 6))

# Scatter plot of residuals vs predicted values
plt.scatter(y_pred_cleaned, residuals_cleaned, color='blue', alpha=0.5)

# Add labels and title
plt.axhline(y=0, color='black', linestyle='--')  # Add a horizontal line at 0
plt.title('Residual Plot (Cleaned Data)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# Show the plot
plt.show()




