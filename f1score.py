import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

# Paths
MODEL_DIR = "models"
data_path = "2018_cleaned.csv"

# Load dataset
df = pd.read_csv(data_path)

# Drop unused columns
df = df.drop(columns=["code", "laboratory_species"], errors="ignore")

# ✅ Encode species here temporarily
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"].astype(str))

# Get antibiotic columns (after 'species')
antibiotic_columns = df.columns[2:]

print("Evaluating F1 scores for each model...\n")

for antibiotic in antibiotic_columns:
    model_path = os.path.join(MODEL_DIR, f"model_{antibiotic}.pkl")

    if not os.path.exists(model_path):
        print(f"❌ Model not found for {antibiotic}")
        continue

    df_filtered = df[df[antibiotic].isin(["R", "S"])].copy()
    if len(df_filtered[antibiotic].unique()) < 2:
        print(f"⚠ Skipping {antibiotic}: Not enough class variety.")
        continue

    df_filtered[antibiotic] = df_filtered[antibiotic].map({"R": 1, "S": 0})
    X = df_filtered[["species"]]
    y = df_filtered[antibiotic]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    score = f1_score(y_test, y_pred)
    print(f"✅ {antibiotic}: F1 Score = {score:.2f}")

print("\nEvaluation complete.")
