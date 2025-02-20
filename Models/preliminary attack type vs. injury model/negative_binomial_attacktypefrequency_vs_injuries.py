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

# Step 1: Define the independent variables (predictors)
# Using the cleaned data for the regression
X = final_data_cleaned[['Air_Attacks', 'Civil_Unrest', 'Ground_Attacks', 'Other', 'Property_Crimes', 'number_of_attacks']]

# Step 2: Add a constant (intercept) term to the model
X = sm.add_constant(X)

# Step 3: Define the dependent variable (number of injuries)
y = final_data_cleaned['number_of_injuries']

# Step 4: Fit the Negative Binomial regression model
# Using Generalized Linear Model (GLM) with the Negative Binomial family
model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
results = model.fit()

# Step 5: Print the summary of the regression results
print(results.summary())

# Step 6: Visualize the residuals to check for model fit
# Plot the residuals to see if there's any pattern
residuals = results.resid_response  # Get the residuals
plt.figure(figsize=(8,6))
plt.scatter(final_data_cleaned['number_of_attacks'], residuals, color='blue', alpha=0.5)
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Number of Attacks')
plt.ylabel('Residuals')
plt.title('Residuals vs. Number of Attacks')
plt.show()

# Step 7: Plot the predicted vs. actual values to assess the model fit
predictions = results.predict(X)  # Get predictions from the model
plt.figure(figsize=(8,6))
plt.scatter(y, predictions, color='red', alpha=0.5)
plt.plot([min(y), max(y)], [min(y), max(y)], color='black', linestyle='--')  # Add a diagonal line for reference
plt.xlabel('Actual Number of Injuries')
plt.ylabel('Predicted Number of Injuries')
plt.title('Actual vs. Predicted Number of Injuries')
plt.show()
