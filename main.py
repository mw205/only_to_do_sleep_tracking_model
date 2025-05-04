import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load model
model = joblib.load("model.pkl")  # or use pickle.load

# Define input format
class InputData(BaseModel):
    Age: int
    Gender: int
    Sleep_Duration: float
    Physical_Activity_Level: float
    Stress_Level: int
    BMI_Category: int
    Heart_Rate: int
    Daily_Steps: int
    Sleep_Disorder: int
    Systolic_BP: int
    Diastolic_BP: int

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.Age, data.Gender, data.Sleep_Duration,
                        data.Physical_Activity_Level, data.Stress_Level,
                        data.BMI_Category, data.Heart_Rate,
                        data.Daily_Steps, data.Sleep_Disorder,
                        data.Systolic_BP, data.Diastolic_BP]])
    prediction = model.predict(features)
    return {"predicted_quality_of_sleep": float(prediction[0])}
