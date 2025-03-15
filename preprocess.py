import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import os
import joblib

def load_data(data_path):
    """Loads and combines all Parquet files from the data directory."""
    parquet_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".parquet")]
    
    if not parquet_files:
        raise FileNotFoundError(" No Parquet files found in the data directory!")
    
    df_list = [pd.read_parquet(file, engine='pyarrow') for file in parquet_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f" Loaded {len(df)} rows from {len(parquet_files)} files.")
    return df

def preprocess_data(df):
    """Cleans, encodes labels, and scales the dataset."""
    df.dropna(inplace=True)  # Remove missing values

    # Encode categorical labels
    label_encoder = LabelEncoder()
    df["Label"] = label_encoder.fit_transform(df["Label"])

    # Drop non-relevant or highly correlated features
    if "Flow Duration" in df.columns:
        df.drop(["Flow Duration"], axis=1, inplace=True)
    
    # Oversample rare attack classes
    df_majority = df[df["Label"] == 0]  # Benign class (majority)
    df_minority = df[df["Label"].isin([13, 14, 9, 8])]  # Rare classes
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=1000, random_state=42)
    df_balanced = pd.concat([df, df_minority_upsampled])
    
    # Split into features and target variable
    X = df_balanced.drop(columns=["Label"])
    y = df_balanced["Label"]
    
    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(" Data preprocessing completed.")
    return X_scaled, y, scaler, label_encoder, X

def save_preprocessing_tools(models_dir, scaler, label_encoder, feature_names):
    """Saves the preprocessing tools and feature names."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, "label_encoder.pkl"))
    joblib.dump(feature_names, os.path.join(models_dir, "feature_names.pkl"))
    print(" Preprocessing tools saved successfully.")

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    
    df = load_data(data_path)
    X_scaled, y, scaler, label_encoder, X_original = preprocess_data(df)
    save_preprocessing_tools(models_dir, scaler, label_encoder, X_original.columns.tolist())