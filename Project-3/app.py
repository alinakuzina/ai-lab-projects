from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load pre-trained model, scaler, and preprocessing artifacts
model = joblib.load('survival_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')  # Dict of LabelEncoders
dataset_columns = joblib.load('dataset_columns.pkl')  # List of feature names

app = FastAPI()

class PatientData(BaseModel):
    features: dict

@app.post("/predict")
def predict(patient: PatientData):
    # Convert input features into a DataFrame
    input_df = pd.DataFrame([patient.features])

    # Align input DataFrame with the dataset columns
    input_df = input_df.reindex(columns=dataset_columns, fill_value=0)

    # Remove the target column if it exists
    if 'hospital_death' in input_df.columns:
        input_df = input_df.drop(columns=['hospital_death'])

    # Encode categorical columns
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str).map(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    # Scale numerical columns
    numerical_columns = scaler.feature_names_in_
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

    # Predict survival probability
    probability = model.predict_proba(input_df)[0, 1]
    return {"survival_probability": probability}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
