from fastapi import FastAPI
from pydantic_models import CustomerData, PredictionResponse
import mlflow.sklearn
import pandas as pd

app = FastAPI(title="Credit Risk Prediction API")

# Load the best model from MLflow
model_uri = "models:/RandomForest_Model/Production"  # replace with your model's name and stage
model = mlflow.sklearn.load_model(model_uri)

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict probability and class
    prob = model.predict_proba(input_df)[:, 1][0]
    pred_class = int(prob > 0.5)

    return PredictionResponse(is_high_risk=pred_class, probability=prob)
