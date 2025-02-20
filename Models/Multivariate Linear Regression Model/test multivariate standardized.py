import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# Load the original data with number of attacks and number of injuries
attacks_injuries = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\attacks and injuries 2.xlsx')
acled_data = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\ACLED_OCT_11_UPDATED.xlsx')

# Remove timezone aspect from the 'date' columns to avoid inconsistency
attacks_injuries['date'] = pd.to_datetime(attacks_injuries['date']).dt.tz_localize(None)
acled_data['event_date'] = pd.to_datetime(acled_data['event_date']).dt.tz_localize(None)

# Rename the columns for clarity
attacks_injuries.rename(columns={'number of attacks': 'number_of_attacks', 'number of injuries': 'number_of_injuries'}, inplace=True)

# Merge the new ACLED data (which has sub_event_type) with the original data
merged_data = pd.merge(attacks_injuries, acled_data[['event_date', 'sub_event_type']], left_on='date', right_on='event_date')

# Drop any rows with missing values in the key columns
clean_data = merged_data.dropna(subset=['number_of_attacks', 'number_of_injuries', 'sub_event_type'])

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

# One-hot encode the 'sub_event_type' categorical variable
clean_data_encoded = pd.get_dummies(clean_data, columns=['sub_event_type'], drop_first=True)

# Replace special characters in column names for easier reference
clean_data_encoded.columns = clean_data_encoded.columns.str.replace(r'[^\w]', '_', regex=True)

# Create broader standardized categories
clean_data_encoded['Air_Attacks'] = clean_data_encoded['sub_event_type_Shelling_artillery_missile_attack']
clean_data_encoded['Ground_Attacks'] = (
    clean_data_encoded['sub_event_type_Armed_clash'] +
    clean_data_encoded['sub_event_type_Attack'] +
    clean_data_encoded['sub_event_type_Remote_explosive_landmine_IED'] +
    clean_data_encoded['sub_event_type_Grenade']
)
clean_data_encoded['Property_Crimes'] = clean_data_encoded['sub_event_type_Looting_property_destruction']
clean_data_encoded['Other'] = (
    clean_data_encoded['sub_event_type_Change_to_group_activity'] +
    clean_data_encoded['sub_event_type_Disrupted_weapons_use'] +
    clean_data_encoded['sub_event_type_Sexual_violence'] +
    clean_data_encoded['sub_event_type_Other'] +
    clean_data_encoded['sub_event_type_Arrests']
)
clean_data_encoded['Civil_Unrest'] = (
    clean_data_encoded['sub_event_type_Excessive_force_against_protesters'] +
    clean_data_encoded['sub_event_type_Mob_violence'] +
    clean_data_encoded['sub_event_type_Peaceful_protest'] +
    clean_data_encoded['sub_event_type_Protest_with_intervention'] +
    clean_data_encoded['sub_event_type_Violent_demonstration']
)

# Drop original sub_event_type columns (optional)
clean_data_encoded.drop(columns=[col for col in clean_data_encoded.columns if col.startswith('sub_event_type_')], inplace=True)

# One-hot encode the new broader categories
clean_data_encoded = pd.get_dummies(clean_data_encoded, columns=['Air_Attacks', 'Ground_Attacks', 'Property_Crimes', 'Other', 'Civil_Unrest'], drop_first=True)

# Now build the multivariate regression model
formula = 'number_of_injuries ~ number_of_attacks + ' + ' + '.join(
    [col for col in clean_data_encoded.columns if col.startswith(('Air_Attacks_', 'Ground_Attacks_', 'Property_Crimes_', 'Other_', 'Civil_Unrest_'))]
)

model = smf.ols(formula, data=clean_data_encoded).fit()

# Print the model summary
print(model.summary())

# Create a residual plot
clean_data_encoded['fitted_values'] = model.predict(clean_data_encoded)
clean_data_encoded['residuals'] = clean_data_encoded['number_of_injuries'] - clean_data_encoded['fitted_values']

plt.figure(figsize=(10, 6))
plt.scatter(clean_data_encoded['fitted_values'], clean_data_encoded['residuals'], alpha=0.7)
plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y=0
plt.title('Residual Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.grid()
plt.show()