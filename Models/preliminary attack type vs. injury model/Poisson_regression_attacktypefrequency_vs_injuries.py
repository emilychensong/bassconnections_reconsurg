import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load the data (final_attack_injury_table.xlsx)
df = pd.read_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Multivariate Linear Regression Model\final_attack_injury_table.xlsx')


# Rename columns to replace spaces with underscores for easier access
df.columns = df.columns.str.replace(' ', '_')

# Apply the IQR method to remove outliers from the 'number_of_injuries' column

# Calculate the Q1 (25th percentile) and Q3 (75th percentile) for 'number_of_injuries'
Q1 = df['number_of_injuries'].quantile(0.25)
Q3 = df['number_of_injuries'].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the rows where 'number_of_injuries' is outside the IQR bounds
final_data_cleaned = df[(df['number_of_injuries'] >= lower_bound) & (df['number_of_injuries'] <= upper_bound)]

# Print the number of rows before and after removing outliers
print(f"Original data points: {len(df)}")
print(f"Data points after removing outliers: {len(final_data_cleaned)}")

# Prepare the model, treating 'number_of_injuries' as the dependent variable
# and 'Air_Attacks', 'Civil_Unrest', etc., as independent variables
X = final_data_cleaned[['Air_Attacks', 'Civil_Unrest', 'Ground_Attacks', 'Other', 'Property_Crimes', 'number_of_attacks']]
y = final_data_cleaned['number_of_injuries']

# Add a constant (intercept) to the model
X = sm.add_constant(X)

# Fit a Poisson regression model
poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

# Display the model summary
print(poisson_model.summary())

# Calculate the predicted values from the model
y_pred = poisson_model.predict(X)

# Plot the predicted vs actual values (for example)
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='blue', alpha=0.5)
plt.plot([0, max(y)], [0, max(y)], color='red', linestyle='--')  # Line of equality
plt.title('Poisson Regression: Actual vs Predicted Injuries')
plt.xlabel('Actual Injuries')
plt.ylabel('Predicted Injuries')
plt.show()

# Calculate residuals (difference between actual and predicted)
residuals = y - y_pred

# Plot residuals vs. predicted values (residual plot)
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='green', alpha=0.5)
plt.axhline(0, color='red', linestyle='--')  # Zero line
plt.title('Poisson Regression: Residuals vs Predicted Injuries')
plt.xlabel('Predicted Injuries')
plt.ylabel('Residuals')
plt.show()
