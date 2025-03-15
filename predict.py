import os
import joblib
import numpy as np
import pandas as pd

# Ensure correct model path
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get location of predict.py
models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))  # Navigate to models/

# Debugging: Print model path
print(f"🔍 Looking for models in: {models_dir}")

# Load Model & Preprocessing Tools
model = joblib.load(os.path.join(models_dir, "nids_model.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))

print("✅ Model and pre-processing tools loaded successfully!")

def predict(sample):
    """Preprocess input sample and return predicted class."""
    if len(sample) != len(feature_names):
        raise ValueError(f"❌ Expected {len(feature_names)} features, but got {len(sample)}")
    
    sample_df = pd.DataFrame([sample], columns=feature_names)
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)
    return label_encoder.inverse_transform(prediction)[0]

# 📌 Example Usage
if __name__ == "__main__":
    new_sample = np.random.rand(len(feature_names))  # Replace with real feature values
    result = predict(new_sample)
    print(f"🔍 Predicted Class: {result}")
