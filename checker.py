import pickle

import pickle
import os

MODEL_DIR = "models"  # Your models folder

species_encoder_path = os.path.join(MODEL_DIR, "species_encoder.pkl")

with open(species_encoder_path, "rb") as f:
    species_encoder = pickle.load(f)

print("Loaded species_encoder type:", type(species_encoder))