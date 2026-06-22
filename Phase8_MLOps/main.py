
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import json

app = FastAPI(title="ML Model API", version="1.0")

# Charger le modèle au démarrage
model = joblib.load("models/random_forest_v1.pkl")
with open("models/metadata_v1.json") as f:
    metadata = json.load(f)

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str
    model_version: str

@app.get("/")
def root():
    return {"status": "ML API is running", "model": metadata["model_name"]}

@app.get("/health")
def health():
    return {"status": "healthy", "accuracy": metadata["accuracy"]}

@app.get("/model-info")
def model_info():
    return metadata

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        X = np.array(request.features).reshape(1, -1)

        prediction = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0].max())

        if proba > 0.8:
            confidence = "High"
        elif proba > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"

        return PredictionResponse(
            prediction=prediction,
            probability=proba,
            confidence=confidence,
            model_version=metadata["version"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
