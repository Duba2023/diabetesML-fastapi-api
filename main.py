import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from scipy.stats import boxcox, yeojohnson
import joblib # Import joblib

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="A simple API to predict diabetes based on patient's diagnostic measurements."
)

# Load the model and preprocessing tools
try:
    with open('diabetes_prediction_model.pkl', 'rb') as file:
        components = pickle.load(file)
    imputation_medians = components['imputation_medians']
    transformation_lambdas = components['transformation_lambdas']
    model = components['model'] # Scaler is no longer loaded from here
    print("Model and preprocessing tools loaded successfully.")
except FileNotFoundError:
    print("Error: diabetes_prediction_model.pkl not found. Please ensure it's in the same directory.")
    exit()

try:
    # Load scaler separately
    scaler = joblib.load('scaler.joblib')
    print("Scaler loaded successfully from scaler.joblib.")
except FileNotFoundError:
    print("Error: scaler.joblib not found. Please ensure it's in the same directory.")
    exit()

# Define input data schema using Pydantic
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Define the prediction endpoint
@app.post("/predict")
async def predict_diabetes(data: DiabetesInput):
    # Convert input data to a DataFrame
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    # Make a copy for processing
    processed_input_df = input_df.copy()

    # Preprocessing steps (must match the Streamlit app and training)
    columns_to_process_for_zeros = ['Insulin', 'SkinThickness', 'BMI', 'BloodPressure', 'Glucose']
    for col in columns_to_process_for_zeros:
        processed_input_df[f'{col}_Missing'] = 0  # Initialize as not missing
        if processed_input_df[col].iloc[0] == 0:
            processed_input_df.loc[0, f'{col}_Missing'] = 1
            processed_input_df.loc[0, col] = imputation_medians[col]

    # Apply transformations
    columns_for_boxcox = ['Insulin', 'DiabetesPedigreeFunction', 'Age']
    for col in columns_for_boxcox:
        if transformation_lambdas[col] is not None:
            # Box-Cox expects positive values, which should be ensured by imputation
            processed_input_df[col] = boxcox(processed_input_df[col], lmbda=transformation_lambdas[col])

    columns_for_yeojohnson = ['Pregnancies', 'BloodPressure']
    for col in columns_for_yeojohnson:
        if transformation_lambdas[col] is not None:
            processed_input_df[col] = yeojohnson(processed_input_df[col], lmbda=transformation_lambdas[col])

    # Define the full list of feature columns, including missing indicators
    feature_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                       'DiabetesPedigreeFunction', 'Age', 'Insulin_Missing', 'SkinThickness_Missing',
                       'BMI_Missing', 'BloodPressure_Missing', 'Glucose_Missing']

    # Reorder all columns to match the training data
    # Ensure all columns exist, fill with 0 if newly added for missing indicators and not yet present
    for col in feature_columns:
        if col not in processed_input_df.columns:
            processed_input_df[col] = 0
    processed_input_df = processed_input_df[feature_columns]

    # Identify numerical columns to be scaled
    columns_to_scale = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                        'BMI', 'DiabetesPedigreeFunction', 'Age']

    # Separate numerical columns from missing indicator columns
    numerical_input_for_scaling = processed_input_df[columns_to_scale]
    missing_indicators = processed_input_df[[col for col in feature_columns if col.endswith('_Missing')]]

    # Apply scaling only to the numerical columns
    scaled_numerical_input = scaler.transform(numerical_input_for_scaling)
    scaled_numerical_df = pd.DataFrame(scaled_numerical_input, columns=columns_to_scale)

    # Recombine scaled numerical columns and original missing indicators
    # Create a new DataFrame with the correct column order for the model
    final_input_df_for_prediction = pd.DataFrame(index=[0], columns=feature_columns)
    for col in columns_to_scale:
        final_input_df_for_prediction[col] = scaled_numerical_df[col]
    for col in [c for c in feature_columns if c.endswith('_Missing')]:
        final_input_df_for_prediction[col] = missing_indicators[col]


    # Make prediction
    prediction = model.predict(final_input_df_for_prediction)[0]
    prediction_proba = model.predict_proba(final_input_df_for_prediction)[:, 1][0]

    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    probability = float(prediction_proba) # Ensure it's a standard float for JSON serialization

    return {
        "prediction": result,
        "probability": probability,
        "input_data": data.dict()
    }
