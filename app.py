from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")

# Model yükle
data = joblib.load("heart_model.pkl")
model = data["model"]
features = data["features"]


# Input schema
class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.post("/predict")
def predict_heart_disease(input: HeartInput):
    input_data = np.array([getattr(input, feature) for feature in features]).reshape(
        1, -1
    )

    proba = model.predict_proba(input_data)[0][1]

    threshold = 0.3
    prediction = int(proba >= threshold)

    return {
        "heart_disease_probability": round(float(proba), 3),
        "prediction": prediction,
        "threshold": threshold,
    }
