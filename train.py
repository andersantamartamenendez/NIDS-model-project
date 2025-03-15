import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_data, preprocess_data

# Step 1: Load Preprocessed Data
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data")
models_dir = os.path.join(script_dir, "..", "models")

# Load dataset
df = load_data(data_path)

# Preprocess data (this includes feature scaling and encoding)
X_scaled, y, scaler, label_encoder, X_original = preprocess_data(df)

# Ensure the models directory exists
os.makedirs(models_dir, exist_ok=True)

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, stratify=y, random_state=42)

# Ensure each class is present in training
if len(np.unique(y_train)) < len(np.unique(y)):
    print("⚠️ Some classes are missing in training! Consider adjusting test_size.")

# Print label distribution in training and test sets
print("Training Class Distribution:", dict(pd.Series(y_train).value_counts()))
print("Test Class Distribution:", dict(pd.Series(y_test).value_counts()))

# Fill missing values to avoid crashes
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

# Step 3: Train the Model
print("Step 3: Train the Model...")
class_weights = {0: 1, 1: 3, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 2, 13: 2, 14: 2}

model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2, random_state=42, class_weight=class_weights, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training completed.")

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 5: Save the Model
joblib.dump(model, os.path.join(models_dir, "nids_model.pkl"))
print(f"\n✅ Model saved successfully in: {models_dir}")