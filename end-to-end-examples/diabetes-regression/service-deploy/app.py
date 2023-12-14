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
    input_data = list[int]

class GeneratedPred(BaseModel):
    prediction: int

@app.post("/generate", response_model=GeneratedPred)
async def predict(data: ParameterInput):
    prediction = classifier.predict([data.input_data])
    return {"predictions": prediction}
