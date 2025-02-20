import pandas as pd

# Function to load data from multiple sheets
def load_governorate_data(file_path):
    xls = pd.ExcelFile(file_path)
    sheets = ['Gaza City', 'Deir Al-Balah', 'Khan Younis', 'North Gaza', 'Rafah']  # List of relevant sheets
    data = {}

    for sheet in sheets:
        data[sheet] = pd.read_excel(xls, sheet_name=sheet)
    
    return data

# Function to process each sheet and aggregate by month
def aggregate_by_month(data):
    for governorate, df in data.items():
        # Ensure the 'Date' column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Group by month and year, then average the 'IDPs at Shelters' for each month
        df['YearMonth'] = df['Date'].dt.to_period('M')  # Creates a Year-Month period (e.g., '2023-10')
        monthly_avg = df.groupby('YearMonth')['IDPs at Shelters'].mean().reset_index()
        
        # Add the result back to the dictionary
        data[governorate] = monthly_avg
        
    return data

# Function to assign population density levels based on relative values
def assign_population_density_levels(monthly_data):
    # Get the overall average IDP values for each month across all governorates
    all_governorates_avg = pd.concat(monthly_data.values())
    overall_monthly_avg = all_governorates_avg.groupby('YearMonth')['IDPs at Shelters'].mean()
    
    # Compute thresholds (high, medium, low) based on percentiles or fixed ranges
    thresholds = overall_monthly_avg.quantile([0.33, 0.67]).values  # For low, medium, high thresholds
    
    # Add density levels to each governorate's data
    for governorate, df in monthly_data.items():
        # Classify into High, Medium, Low based on thresholds
        df['Density Level'] = df['IDPs at Shelters'].apply(
            lambda x: 'Low' if x <= thresholds[0] else ('Medium' if x <= thresholds[1] else 'High')
        )
        
    return monthly_data

# Function to merge data with population density sheet
def merge_with_population_density(pop_density_df, governorate_data):
    # Loop through each governorate's monthly data and merge with population density sheet
    for governorate, data in governorate_data.items():
        # Merge by matching the YearMonth to the population density sheet
        pop_density_df.loc[pop_density_df['City'] == governorate, ['PD_Oct', 'PD_Nov', 'PD_Dec', 'PD_Jan']] = data['Density Level'].values
    
    return pop_density_df

# Main function to orchestrate the process
def process_data(idp_shelter_file, population_density_file):
    # Load the population density sheet
    pop_density_df = pd.read_excel(population_density_file)
    
    # Load and process IDP shelter data
    idp_data = load_governorate_data(idp_shelter_file)
    monthly_data = aggregate_by_month(idp_data)
    governorate_data_with_levels = assign_population_density_levels(monthly_data)
    
    # Merge the density levels into the population density DataFrame
    updated_pop_density_df = merge_with_population_density(pop_density_df, governorate_data_with_levels)
    
    return updated_pop_density_df

# Example usage
idp_shelter_file = r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Gaza IDPs.xlsx'  # Update with your file path
population_density_file = r'C:\Users\emily\OneDrive - Duke University\Bass Connections\IDP_pop_density.xlsx'  # Update with your file path

# Process the data
updated_pop_density = process_data(idp_shelter_file, population_density_file)

# Output the updated DataFrame
updated_pop_density.to_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\IDP_pop_density.xlsx', index=False)



#%%
"""
Changes Summary:
Population Density Calculation: Now, the population density is calculated using IDPs at Shelters divided by the area of the governorate. This should reflect the actual density, not just the raw number of people.
Weighted Average: Instead of calculating a simple average of IDPs, we calculate a weighted average population density using the total number of IDPs as the weight. This ensures that larger governorates have a more significant impact on the overall density.
Updated Density Level Assignment: The population density levels (low, medium, high) are now based on the weighted average population density.
"""

import pandas as pd

# Sample area data (in square kilometers) for each governorate
governorate_area = {
    'Gaza City': 70, 
    'Deir Al-Balah': 56, 
    'Khan Younis': 108, 
    'North Gaza': 61, 
    'Rafah': 65
}

# Function to load data from multiple sheets
def load_governorate_data(file_path):
    xls = pd.ExcelFile(file_path)
    sheets = ['Gaza City', 'Deir Al-Balah', 'Khan Younis', 'North Gaza', 'Rafah']  # List of relevant sheets
    data = {}

    for sheet in sheets:
        data[sheet] = pd.read_excel(xls, sheet_name=sheet)
    
    return data

# Function to process each sheet and aggregate by month, considering population density
def aggregate_by_month(data):
    for governorate, df in data.items():
        # Ensure the 'Date' column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Calculate population density for each row (IDPs / Area)
        df['Population Density'] = df['IDPs at Shelters'] / governorate_area.get(governorate, 1)  # Default area to 1 if not found
        
        # Group by month and year, then calculate the weighted average of population density
        df['YearMonth'] = df['Date'].dt.to_period('M')  # Creates a Year-Month period (e.g., '2023-10')
        
        # Weighted average population density by the number of IDPs (or total population)
        weighted_avg = df.groupby('YearMonth').apply(
            lambda x: (x['Population Density'] * x['IDPs at Shelters']).sum() / x['IDPs at Shelters'].sum()
        ).reset_index(name='Weighted Population Density')
        
        # Add the result back to the dictionary
        data[governorate] = weighted_avg
        
    return data

# Function to assign population density levels based on relative values
def assign_population_density_levels(monthly_data):
    # Combine all governorate data to calculate overall density distribution
    all_governorates_data = pd.concat(monthly_data.values())
    
    # Compute thresholds (low, medium, high) using quantiles (33rd and 67th percentiles)
    thresholds = all_governorates_data['Weighted Population Density'].quantile([0.33, 0.67]).values
    
    # Add density levels to each governorate's data
    for governorate, df in monthly_data.items():
        df['Density Level'] = df['Weighted Population Density'].apply(
            lambda x: 'Low' if x <= thresholds[0] else ('Medium' if x <= thresholds[1] else 'High')
        )
        
    return monthly_data

# Function to merge data with population density sheet (with overwrite behavior)
def merge_with_population_density(pop_density_df, governorate_data):
    # Identify the month columns to be replaced (e.g., '2023-10', '2023-11', ...)
    month_columns = ['2023-10', '2023-11', '2023-12', '2024-01']  # Update this list with actual columns if needed

    # Loop through each governorate's monthly data and merge with population density sheet
    for governorate, data in governorate_data.items():
        for idx, row in data.iterrows():
            year_month = row['YearMonth']
            density_level = row['Density Level']
            
            # Check if the year_month is a valid column in the population density DataFrame
            if year_month in month_columns:
                # If so, overwrite the existing value in the respective month column
                pop_density_df.loc[pop_density_df['City'] == governorate, year_month] = density_level
    
    return pop_density_df

# Main function to orchestrate the process
def process_data(idp_shelter_file, population_density_file):
    # Load the population density sheet
    pop_density_df = pd.read_excel(population_density_file)
    
    # Load and process IDP shelter data
    idp_data = load_governorate_data(idp_shelter_file)
    monthly_data = aggregate_by_month(idp_data)
    governorate_data_with_levels = assign_population_density_levels(monthly_data)
    
    # Merge the density levels into the population density DataFrame
    updated_pop_density_df = merge_with_population_density(pop_density_df, governorate_data_with_levels)
    
    return updated_pop_density_df

# Example usage
idp_shelter_file = r'C:\Users\emily\OneDrive - Duke University\Bass Connections\Data\IDP\Gaza IDPs UNWRA Government Shelters.xlsx'  # Update with your file path
population_density_file = r'C:\Users\emily\OneDrive - Duke University\Bass Connections\IDP_pop_density_updated.xlsx'  # Update with your file path

# Process the data
updated_pop_density = process_data(idp_shelter_file, population_density_file)

# Output the updated DataFrame, replacing the old data
updated_pop_density.to_excel(r'C:\Users\emily\OneDrive - Duke University\Bass Connections\IDP_pop_density_updated.xlsx', index=False)
