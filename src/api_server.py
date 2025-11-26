import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
# --- Configuration ---
MODEL_FILE = 'voting_model_ai4i_enriched.joblib'
SCALER_FILE = 'scaler_ai4i_enriched.joblib'

# --- 1. Initialize FastAPI App ---
app = FastAPI(
    title="Predictive Maintenance API",
    description="API for predicting machine failure based on sensor readings.",
    version="1.0"
)

# --- 2. Load Models at Startup ---
# Global variables to hold the model and scaler
model = None
scaler = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        raise FileNotFoundError("Model or Scaler file not found. Please ensure .joblib files are in the directory.")
    
    print("Loading model and scaler...")
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    print("Artifacts loaded successfully.")

# --- 3. Define Input Data Schema ---
# We use Pydantic to validate the input JSON. 
# Field aliases allow the API to accept clean names (e.g., "air_temperature") 
# while mapping them to the specific columns the model expects.
class SensorReading(BaseModel):
    # 'Type' is categorical (L, M, H)
    Type: str = Field(..., description="Machine Type: L, M, or H")
    
    # Numerical features
    air_temperature: float = Field(..., description="Air Temperature [K]")
    process_temperature: float = Field(..., description="Process Temperature [K]")
    rotational_speed: int = Field(..., description="Rotational Speed [rpm]")
    torque: float = Field(..., description="Torque [Nm]")
    tool_wear: int = Field(..., description="Tool Wear [min]")

    class Config:
        schema_extra = {
            "example": {
                "Type": "M",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551,
                "torque": 42.8,
                "tool_wear": 0
            }
        }

# --- 4. Prediction Endpoint ---
@app.post("/predict")
def predict_failure(data: SensorReading):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 1. Convert input data to a Pandas DataFrame
        # The keys here MUST match the column names expected by the scaler/model
        input_data = {
            'Type': [data.Type],
            'Air temperature [K]': [data.air_temperature],
            'Process temperature [K]': [data.process_temperature],
            'Rotational speed [rpm]': [data.rotational_speed],
            'Torque [Nm]': [data.torque],
            'Tool wear [min]': [data.tool_wear]
        }
        
        input_df = pd.DataFrame(input_data)

        # 2. Preprocess the data (Scaling + OneHotEncoding)
        # The scaler is a ColumnTransformer pipeline
        input_processed = scaler.transform(input_df)

        # 3. Make Prediction
        prediction = model.predict(input_processed)[0]
        
        # Get probability if available (VotingClassifier usually has predict_proba)
        probability = 0.0
        if hasattr(model, "predict_proba"):
            # Probability of class 1 (Failure)
            probability = model.predict_proba(input_processed)[0][1]

        # 4. Return Result
        return {
            "prediction": int(prediction),
            "prediction_label": "Failure" if prediction == 1 else "No Failure",
            "failure_probability": float(round(probability, 4))
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def log_prediction(input_data, prediction):
    record = {
        "timestamp": str(datetime.now()),
        "input": input_data,
        "prediction": int(prediction)
    }
    with open("logs.json", "a") as f:
        f.write(json.dumps(record) + "\n")
# --- 5. Run Standalone (Optional) ---

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)