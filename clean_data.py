import pandas as pd

# Load dataset
df = pd.read_csv("2018_clean.csv")

# Fill 'laboratory_species' missing values with "Unknown"
if 'laboratory_species' in df.columns:
    df['laboratory_species'].fillna("Unknown", inplace=True)

# Fill missing R/S values with "Unknown" instead of dropping them
for col in df.columns:
    if df[col].dtype == object:  # Only apply to text-based columns (like R/S)
        df[col].fillna("Unknown", inplace=True)

# Remove columns where **ALL** values are missing
df.dropna(axis=1, how='all', inplace=True)

# Save cleaned dataset
cleaned_filename = "2018_cleaned.csv"
df.to_csv(cleaned_filename, index=False)

print(f"Data cleaning completed. Cleaned file saved as '{cleaned_filename}'.")