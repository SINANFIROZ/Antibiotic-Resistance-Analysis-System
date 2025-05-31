import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Define paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv("2018_cleaned.csv")  # Update filename if needed

# Drop unnecessary columns
df = df.drop(columns=["code", "laboratory_species"], errors="ignore")

# Encode the species column
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# Save label encoder
joblib.dump(le, os.path.join(MODEL_DIR, "species_encoder.pkl"))

# Identify antibiotic columns
antibiotic_columns = df.columns[2:]

# Dictionary to store alternative antibiotics
alternative_antibiotics = {}

# Train models for each antibiotic
for antibiotic in antibiotic_columns:
    print(f"Training model for {antibiotic}...")

    # Filter rows with valid resistance data (R/S)
    df_filtered = df[df[antibiotic].isin(["R", "S"])]

    # Convert R/S labels to binary (1 = Resistant, 0 = Susceptible)
    df_filtered[antibiotic] = df_filtered[antibiotic].map({"R": 1, "S": 0})

    # Define input (X) and output (y)
    X = df_filtered[["species"]]
    y = df_filtered[antibiotic]

    # Skip antibiotics with insufficient data
    if len(y.unique()) < 2:
        print(f"⚠ Skipping {antibiotic} (Not enough data)")
        continue

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    # Save the trained model
    model_filename = os.path.join(MODEL_DIR, f"model_{antibiotic}.pkl")
    joblib.dump(model, model_filename)
    print(f"✅ Saved model: {model_filename}")

    # Identify alternative antibiotics based on correlations
    resistant_cases = df_filtered[df_filtered[antibiotic] == 1]
    susceptible_alternatives = {}
    for other_antibiotic in antibiotic_columns:
        if other_antibiotic != antibiotic:
            susceptible_count = resistant_cases[other_antibiotic].value_counts().get(0, 0)
            susceptible_alternatives[other_antibiotic] = susceptible_count

    # Store the top alternative antibiotics
    alternative_antibiotics[antibiotic] = sorted(susceptible_alternatives, key=susceptible_alternatives.get, reverse=True)[:3]

# Save alternative antibiotic mappings
joblib.dump(alternative_antibiotics, os.path.join(MODEL_DIR, "alternative_antibiotics.pkl"))
print("✅ Training complete! All models updated.")