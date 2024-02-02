import os
from typing import List
from joblib import load
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from uvicorn.workers import UvicornWorker

classifier = load("../train/classifier.joblib")
app = FastAPI()


class ParameterInput(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float

class GeneratedPred(BaseModel):
    prediction: int

@app.post("/generate", response_model=GeneratedPred)
async def predict(data: ParameterInput):
    features = [data.CreditScore, data.Age, data.Tenure, data.Balance, data.NumOfProducts, data.HasCrCard,data.IsActiveMember, data.EstimatedSalary]
    prediction = classifier.predict([features])
    print(prediction)
    return {"prediction": prediction[0]}
