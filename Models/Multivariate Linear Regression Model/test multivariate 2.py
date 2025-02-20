"""
Multivariate Linear Regression Model

Independent Variables: # of attacks (up to Oct 11 24), sub_event_type (up to Oct 11 24)
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

print("Upper_bound:", upper_bound)

# Filter out outliers
clean_data = clean_data[(clean_data['number_of_injuries'] >= lower_bound) & (clean_data['number_of_injuries'] <= upper_bound)]

# Print the number of observations
print("Number of observations (days) in clean_data:", len(clean_data))

# One-hot encode the 'sub_event_type' categorical variable
clean_data_encoded = pd.get_dummies(clean_data, columns=['sub_event_type'], drop_first=True)

# Replace special characters (spaces, slashes, etc.) in column names for easier reference
clean_data_encoded.columns = clean_data_encoded.columns.str.replace(r'[^\w]', '_', regex=True)

# Check the columns to see how they look after replacing spaces
print("Columns after replacing spaces and special characters:\n", clean_data_encoded.columns)


# Build the new multivariate regression model
formula = 'number_of_injuries ~ number_of_attacks + ' + ' + '.join(
    [col for col in clean_data_encoded.columns if col.startswith('sub_event_type_')]
)
model = smf.ols(formula, data=clean_data_encoded).fit()

# Print the model summary
print(model.summary())

#%% 3D PLOT

# Create a 3D plot (for one specific sub_event_type)
# Assuming we are visualizing 'sub_event_type_Shelling_artillery_missile_attack'
x = np.linspace(clean_data_encoded['number_of_attacks'].min(), clean_data_encoded['number_of_attacks'].max(), 100)

# Define 'y' for the specific event type. Since it's binary, set it to 0 or 1.
y = np.array([0, 1])  # Binary variable for sub_event_type

# Meshgrid for plotting
x, y = np.meshgrid(x, y)

# Create DataFrame for predictions
pred_data = pd.DataFrame({
    'number_of_attacks': x.ravel(),
    'sub_event_type_Shelling_artillery_missile_attack': y.ravel()
})

# Ensure all other sub_event_type columns are zero in the prediction data
for col in clean_data_encoded.columns:
    if col.startswith('sub_event_type_') and col != 'sub_event_type_Shelling_artillery_missile_attack':
        pred_data[col] = 0

# Predict using the model
pred_data['number_of_injuries'] = model.predict(pred_data)

# Reshape predictions to match the grid
z = pred_data['number_of_injuries'].values.reshape(x.shape)

# Plot the surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)

# Plot the actual data points
ax.scatter(
    clean_data_encoded['number_of_attacks'], 
    clean_data_encoded['sub_event_type_Shelling_artillery_missile_attack'], 
    clean_data_encoded['number_of_injuries'], 
    color='r', s=50, label='Actual data'
)

# Set labels
ax.set_xlabel('Number of Attacks')
ax.set_ylabel('Attack Type (Shelling/Artillery/Missile)')
ax.set_zlabel('Number of Injuries')
ax.set_title('3D Multivariate Regression Model')

# Show the plot
plt.show()

#%% REISDUAL PLOT

# Calculate the fitted values and residuals
clean_data_encoded['fitted_values'] = model.predict(clean_data_encoded)
clean_data_encoded['residuals'] = clean_data_encoded['number_of_injuries'] - clean_data_encoded['fitted_values']

# Create a residual plot
plt.figure(figsize=(10, 6))
plt.scatter(clean_data_encoded['fitted_values'], clean_data_encoded['residuals'], alpha=0.7)
plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y=0
plt.title('Residual Plot')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.grid()
plt.show()

#%% TRAIN-TEST SPLIT

from sklearn.model_selection import train_test_split

# One-hot encode the 'sub_event_type' categorical variable (if not already done)
clean_data_encoded = pd.get_dummies(clean_data, columns=['sub_event_type'], drop_first=True)

# Replace special characters (spaces, slashes, etc.) in column names for easier reference
clean_data_encoded.columns = clean_data_encoded.columns.str.replace(r'[^\w]', '_', regex=True)

# Split the data into training and testing sets (70% training, 30% testing)
train_data, test_data = train_test_split(clean_data_encoded, test_size=0.3, random_state=42)

# Build the multivariate regression model using the training data
formula = 'number_of_injuries ~ number_of_attacks + ' + ' + '.join(
    [col for col in train_data.columns if col.startswith('sub_event_type_')]
)
model = smf.ols(formula, data=train_data).fit()

# Print the model summary
print("Training model summary:\n", model.summary())

# Make predictions on the test data
test_data['fitted_values'] = model.predict(test_data)
test_data['residuals'] = test_data['number_of_injuries'] - test_data['fitted_values']

# Create a residual plot for the test predictions
plt.figure(figsize=(10, 6))
plt.scatter(test_data['fitted_values'], test_data['residuals'], alpha=0.7)
plt.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y=0
plt.title('Residual Plot for Test Predictions')
plt.xlabel('Fitted Values (Test Set)')
plt.ylabel('Residuals (Test Set)')
plt.grid()
plt.show()

# Optionally, print the number of observations in the train and test sets
print("Number of observations in training set:", len(train_data))
print("Number of observations in test set:", len(test_data))


#%% PREDICTING RECONSTRUCTIVE SURGERY

# Print the equation of the regression model
intercept = model.params['Intercept']
coef_attacks = model.params['number_of_attacks']
# Get coefficients for the sub_event_type variables
sub_event_coefs = {col: model.params[col] for col in model.params.index if col.startswith('sub_event_type_')}

# Print the regression equation
sub_event_terms = " + ".join([f"{coef:.2f} * {col.replace('sub_event_type_', '')}" for col, coef in sub_event_coefs.items()])
print(f"Regression Equation: Number of Injuries = {intercept:.2f} + {coef_attacks:.2f} * Number of Attacks + {sub_event_terms}")

# Create a grid for predictions
x = np.linspace(clean_data['number_of_attacks'].min(), clean_data['number_of_attacks'].max(), 100)
y = np.array([0, 1])  # Binary variable for sub_event_type

# Create a meshgrid for plotting
x, y = np.meshgrid(x, y)

# Create DataFrame for predictions
pred_data = pd.DataFrame({
    'number_of_attacks': x.ravel(),
    'sub_event_type_Shelling_artillery_missile_attack': y.ravel()  # Change this according to the event type you want to visualize
})

# Ensure all other sub_event_type columns are zero in the prediction data
for col in clean_data_encoded.columns:
    if col.startswith('sub_event_type_') and col != 'sub_event_type_Shelling_artillery_missile_attack':
        pred_data[col] = 0

# Predict using the model
pred_data['number_of_injuries'] = model.predict(pred_data)

# Calculate the injuries requiring surgery
pred_data['injuries_requiring_surgery'] = pred_data['number_of_injuries'] * 0.235  # 23.5%

# Reshape predictions to match the grid for injuries requiring surgery
z = pred_data['injuries_requiring_surgery'].values.reshape(x.shape)

# Create a 3D plot for predicted injuries requiring reconstructive surgery
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)

# Plot the actual data points for injuries requiring surgery
ax.scatter(
    clean_data_encoded['number_of_attacks'], 
    clean_data_encoded['sub_event_type_Shelling_artillery_missile_attack'], 
    clean_data_encoded['number_of_injuries'] * 0.235,  # Adjusted for surgery prediction
    color='r', s=50, label='Actual data (Injuries requiring Surgery)'
)

# Set labels
ax.set_xlabel('Number of Attacks')
ax.set_ylabel('Attack Type (Shelling/Artillery/Missile)')
ax.set_zlabel('Predicted Injuries Requiring Reconstructive Surgery')
ax.set_title('3D Multivariate Regression Model for Injuries Requiring Surgery')

# Show the legend
ax.legend()

# Show the plot
plt.show()
