import pickle
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from scipy.stats import boxcox, yeojohnson

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Prediction API",
    description="Predict diabetes based on diagnostic measurements."
)

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running 🚀"}

# Load model and preprocessing tools
try:
    with open("diabetes_prediction_model.pkl", "rb") as f:
        components = pickle.load(f)
    imputation_medians = components["imputation_medians"]
    transformation_lambdas = components["transformation_lambdas"]
    model = components["model"]
    print("Model and preprocessing tools loaded successfully.")
except FileNotFoundError:
    print("diabetes_prediction_model.pkl not found.")
    exit()

try:
    scaler = joblib.load("scaler.joblib")
    print("Scaler loaded successfully from scaler.joblib.")
except FileNotFoundError:
    print("scaler.joblib not found.")
    exit()

# Define Pydantic input model
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Handle zeros as missing
    cols_to_check = ['Insulin', 'SkinThickness', 'BMI', 'BloodPressure', 'Glucose']
    for col in cols_to_check:
        input_df[f"{col}_Missing"] = 0
        if input_df.at[0, col] == 0:
            input_df.at[0, f"{col}_Missing"] = 1
            input_df.at[0, col] = imputation_medians[col]

    # Apply transformations safely
    for col in ['Insulin', 'DiabetesPedigreeFunction', 'Age']:
        if transformation_lambdas.get(col) is not None and input_df.at[0, col] > 0:
            input_df[col] = boxcox(input_df[col], lmbda=transformation_lambdas[col])
    for col in ['Pregnancies', 'BloodPressure']:
        if transformation_lambdas.get(col) is not None:
            input_df[col] = yeojohnson(input_df[col], lmbda=transformation_lambdas[col])

    # Define features
    feature_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
                    'BMI','DiabetesPedigreeFunction','Age',
                    'Insulin_Missing','SkinThickness_Missing','BMI_Missing',
                    'BloodPressure_Missing','Glucose_Missing']

    # Ensure all missing indicator columns exist
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    # Scale numeric columns
    numeric_cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
                    'BMI','DiabetesPedigreeFunction','Age']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    try:
        probability = model.predict_proba(input_df)[0][1]
    except AttributeError:
        probability = float(prediction)  # fallback if model has no predict_proba

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    return {
        "predicted_outcome": result,
        "prediction_probability": float(probability)
    }
