import pandas as pd
import json

# Load dataset
df = pd.read_csv("2018_cleaned.csv")

# Dropping unnecessary columns
df = df.drop(columns=["code", "laboratory_species"], errors="ignore")

# Get list of antibiotic columns (excluding species)
antibiotic_columns = df.columns[2:]

# Dictionary to store alternative antibiotics
alternative_antibiotics = {}

# Loop through each microbe species
for microbe in df["species"].unique():
    microbe_data = df[df["species"] == microbe]

    # Identify resistant (R) and susceptible (S) antibiotics
    resistant_antibiotics = set(antibiotic_columns[(microbe_data[antibiotic_columns] == "R").any(axis=0)])
    susceptible_antibiotics = set(antibiotic_columns[(microbe_data[antibiotic_columns] == "S").any(axis=0)])

    # Map resistant antibiotics to available susceptible alternatives
    for antibiotic in resistant_antibiotics:
        alternative_antibiotics.setdefault(microbe, {})[antibiotic] = list(susceptible_antibiotics)

# Save the extracted alternatives in a JSON file
with open("models/alternative_antibiotics.json", "w") as f:
    json.dump(alternative_antibiotics, f, indent=4)

print("✅ Alternative antibiotics extracted and saved.")