
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from scipy.stats import boxcox, yeojohnson

# Initialize FastAPI app
app = FastAPI(title="Diabetes Prediction API")
@app.get("/")
def home():
    return {"message": "Diabetes ML FastAPI API is running 🚀"}

# Load the pre-trained model and preprocessing tools
try:
    with open('diabetes_prediction_model.pkl', 'rb') as file:
        components = pickle.load(file)

    imputation_medians = components['imputation_medians']
    transformation_lambdas = components['transformation_lambdas']
    scaler = components['scaler']
    model = components['model']

    print("Model and preprocessing tools loaded successfully for FastAPI.")
except FileNotFoundError:
    print("Error: diabetes_prediction_model.pkl not found. Make sure it's in the same directory.")
    imputation_medians = None
    transformation_lambdas = None
    scaler = None
    model = None

# Define the input data model for FastAPI
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Define the full list of feature columns, including potential missing indicators
# This must match the order and names used during model training
feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                   'DiabetesPedigreeFunction', 'Age', 'Insulin_Missing', 'SkinThickness_Missing',
                   'BMI_Missing', 'BloodPressure_Missing', 'Glucose_Missing']

# Columns that originally had 0s that were imputed and need missing indicators
columns_to_process = ['Insulin', 'SkinThickness', 'BMI', 'BloodPressure', 'Glucose']

# Columns for Box-Cox transformation (requires positive values)
columns_for_boxcox = ['Insulin', 'DiabetesPedigreeFunction', 'Age']

# Columns for Yeo-Johnson transformation (handles zero and negative values)
columns_for_yeojohnson = ['Pregnancies', 'BloodPressure']

# Numerical columns to be scaled
columns_to_scale = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age']

# Prediction endpoint
@app.post("/predict")
async def predict_diabetes(data: DiabetesInput):
    if model is None:
        return {"error": "Model not loaded. Please ensure 'diabetes_prediction_model.pkl' exists."}

    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Apply preprocessing steps exactly as in training
    processed_input_df = input_df.copy()

    # 1. Handle 0s and create missing indicators
    for col in columns_to_process:
        processed_input_df[f'{col}_Missing'] = 0  # Initialize as not missing
        if processed_input_df[col].iloc[0] == 0:
            processed_input_df.loc[0, f'{col}_Missing'] = 1
            processed_input_df.loc[0, col] = imputation_medians[col]

    # 2. Apply transformations
    for col in columns_for_boxcox:
        if transformation_lambdas[col] is not None:
            # Ensure the input is positive before applying boxcox
            if processed_input[col].iloc[0] <= 0:
                # Handle cases where value might become non-positive after imputation
                # For simplicity, we'll use a small positive number if this happens
                processed_input_df.loc[0, col] = 1e-6 # A very small positive number
            processed_input_df[col] = boxcox(processed_input_df[col], lmbda=transformation_lambdas[col])

    for col in columns_for_yeojohnson:
        if transformation_lambdas[col] is not None:
            processed_input_df[col] = yeojohnson(processed_input_df[col], lmbda=transformation_lambdas[col])

    # Reorder all columns (both original and missing indicators) to match the training data
    # First, ensure all missing indicator columns are present, even if all zeros
    for col in columns_to_process:
        if f'{col}_Missing' not in processed_input_df.columns:
            processed_input_df[f'{col}_Missing'] = 0

    processed_input_df = processed_input_df[feature_columns] # Ensure all columns are present and ordered

    # Separate numerical columns from missing indicator columns
    numerical_input_for_scaling = processed_input_df[columns_to_scale]
    missing_indicators = processed_input_df[[col for col in feature_columns if col.endswith('_Missing')]]

    # Apply scaling only to the numerical columns
    scaled_numerical_input = scaler.transform(numerical_input_for_scaling)
    scaled_numerical_df = pd.DataFrame(scaled_numerical_input, columns=columns_to_scale)

    # Recombine scaled numerical columns and original missing indicators
    final_input_df_for_prediction = pd.DataFrame(index=[0], columns=feature_columns)
    for col in columns_to_scale:
        final_input_df_for_prediction[col] = scaled_numerical_df[col]
    for col in [c for c in feature_columns if c.endswith('_Missing')]:
        final_input_df_for_prediction[col] = missing_indicators[col]

    # Make prediction
    prediction = model.predict(final_input_df_for_prediction)[0]
    prediction_proba = model.predict_proba(final_input_df_for_prediction)[:, 1][0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return {"prediction": result, "probability": round(prediction_proba, 4)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
