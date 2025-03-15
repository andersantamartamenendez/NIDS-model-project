from fastapi import FastAPI, HTTPException
import os
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="NIDS ML Model API", description="API for Network Intrusion Detection System", version="1.0")

# Load model and preprocessing tools
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "..", "models")

try:
    model = joblib.load(os.path.join(models_dir, "nids_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
    print(" Model and preprocessing tools loaded successfully!")
except Exception as e:
    raise RuntimeError(f" Failed to load model or preprocessing tools: {e}")

# Define input data model
class InputData(BaseModel):
    features: list[float]  # Expecting a list of float numbers

@app.get("/")
def home():
    return {"message": " NIDS ML Model API is running! Send a POST request to /predict."}

@app.post("/predict")
def predict(data: InputData):
    """Predicts network attack type based on input features."""
    if len(data.features) != len(feature_names):
        raise HTTPException(status_code=400, detail=f" Expected {len(feature_names)} features, but got {len(data.features)}")
    
    sample_df = pd.DataFrame([data.features], columns=feature_names)
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    
    return {"prediction": predicted_label}

# Run with: uvicorn app:app --reload
