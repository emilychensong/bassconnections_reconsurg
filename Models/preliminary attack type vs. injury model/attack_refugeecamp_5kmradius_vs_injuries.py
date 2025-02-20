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

# Step 2: Determine if each attack is within a 5 km radius of the nearest refugee camp
def within_5km_radius(attack_coords):
    distances = refugee_camps_df.apply(
        lambda row: geodesic(attack_coords, (row['latitude'], row['longitude'])).kilometers, axis=1
    )
    return "yes" if distances.min() <= 5 else "no"

# Apply the function to label each attack as "yes" or "no" based on proximity
attacks_df['within_5km'] = attacks_df.apply(
    lambda row: within_5km_radius((row['latitude'], row['longitude'])), axis=1
)

# Step 3: Count the number of "yes" occurrences for each date
attack_counts_by_date = attacks_df.groupby('date')['within_5km'].apply(lambda x: (x == 'yes').sum()).reset_index()
attack_counts_by_date.columns = ['date', 'within_5km_count']

# Step 4: Merge with injury data
merged_df = pd.merge(attack_counts_by_date, injuries_df, on='date', how='inner')

# Check for missing values and drop them
merged_df.dropna(subset=['within_5km_count', 'number of injuries'], inplace=True)

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
X = filtered_df['within_5km_count']  # Independent variable: count of attacks within 5km
y = filtered_df['number of injuries']  # Dependent variable: number of injuries

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
plt.scatter(filtered_df['within_5km_count'], filtered_df['number of injuries'], color='blue', label='Data Points')

# Plotting the regression line
predicted_y = model.predict(X)
plt.plot(filtered_df['within_5km_count'], predicted_y, color='red', label='Regression Line', linewidth=2)

# Adding titles and labels
plt.title('Linear Regression: Count of Attacks within 5 km vs. Number of Injuries (Outliers Removed)')
plt.xlabel('Count of Attacks within 5 km')
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

# Show the plot
plt.show()



#%% COUNTS OF ATTACK TYPES THAT WERE IN 5 KM RADIUS

# Define the attack type mapping
sub_event_type_mapping = {
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

# Step 1: Filter attacks within 5 km radius
attacks_within_5km = attacks_df[attacks_df['within_5km'] == 'yes']
print("Sample of attacks within 5 km radius:")
print(attacks_within_5km.head())

# Step 2: Map each attack to its broader category
attacks_within_5km['attack_category'] = attacks_within_5km['sub_event_type'].map(sub_event_type_mapping)
print("Sample after mapping event types to attack categories:")
print(attacks_within_5km[['event_type', 'attack_category']].head())

# Step 3: Count occurrences of each attack category
attack_type_counts = attacks_within_5km['attack_category'].value_counts()
print("Attack type counts within 5 km radius:")
print(attack_type_counts)



# Step 4: Plotting the attack type counts in a bar graph
plt.figure(figsize=(10, 6))
attack_type_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Adding titles and labels
plt.title('Count of Attack Types within 5 km Radius of Refugee Camps')
plt.xlabel('Attack Type')
plt.ylabel('Number of Attacks')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Show the plot
plt.show()

#%% COUNTS OF ATTACK TYPES THAT WERE IN 5 KM RADIUS (PROPORTIONAL TO TOTAL NUMBER OF EACH ATTACK TYPE)

# Step 1: Calculate total counts of each attack type (not limited to 5 km radius)
total_attack_counts = attacks_df['sub_event_type'].map(sub_event_type_mapping).value_counts()

# Step 2: Calculate counts within 5 km as already done
attack_type_counts_within_5km = attacks_within_5km['attack_category'].value_counts()

# Step 3: Calculate proportion of each attack type within the 5 km radius
attack_type_proportions = (attack_type_counts_within_5km / total_attack_counts).fillna(0)  # Fill NaN with 0 if no attacks of that type

print("Proportion of each attack type within 5 km radius:")
print(attack_type_proportions)

# Step 4: Plot the proportions in a bar graph
plt.figure(figsize=(10, 6))
attack_type_proportions.plot(kind='bar', color='lightcoral', edgecolor='black')

# Adding titles and labels
plt.title('Proportion of Attack Types within 5 km Radius of Refugee Camps')
plt.xlabel('Attack Type')
plt.ylabel('Proportion of Attacks within 5 km')
plt.xticks(rotation=45)
plt.grid(axis='y')

# Show the plot
plt.show()


#%% COUNTS OF ATTACK TYPES THAT WERE IN 5 KM RADIUS (PROPORTIONAL TO TOTAL NUMBER OF ATTACKS THAT ARE IN 5KM RADIUS)

# Step 1: Calculate the total number of "yes" attacks within the 5 km radius
total_yes_attacks = attack_type_counts_within_5km.sum()

# Step 2: Calculate the proportion of each attack type relative to the total "yes" attacks
attack_type_proportions_within_5km = attack_type_counts_within_5km / total_yes_attacks

# Step 3: Plot the pie chart
plt.figure(figsize=(12, 12))
attack_type_proportions_within_5km.plot(
    kind='pie',
    autopct='%1.1f%%',  # Display percentages with one decimal place
    startangle=90,      # Rotate pie chart to start from the top
    colors=plt.cm.Paired.colors  # Optional: use a colormap for variety
)

# Add title and equal aspect ratio for a circular pie chart
plt.title('Proportion of Attack Types within 5 km Radius of Refugee Camps')
plt.ylabel('')  # Hide the y-label for a cleaner look
plt.axis('equal')  # Ensure the pie chart is a circle

# Show the plot
plt.show()

