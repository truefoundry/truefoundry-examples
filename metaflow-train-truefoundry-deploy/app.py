import os
from typing import List

import mlfoundry
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_VERSION_FQN = os.getenv("MODEL_VERSION_FQN")
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        client = mlfoundry.get_client()
        model_version = client.get_model(MODEL_VERSION_FQN)
        _MODEL = model_version.load()
    return _MODEL


app = FastAPI(root_path=os.getenv("TFY_SERVICE_ROOT_PATH"), docs_url="/")


class PredictionRequest(BaseModel):
    pickup_year: float
    pickup_dow: int
    pickup_hour: int
    abs_distance: float
    pickup_longitude: float
    dropoff_longitude: float


@app.post("/predict")
def predict(inference_requests: List[PredictionRequest]):
    predictions = []
    model = get_model()
    features_list = [r.dict() for r in inference_requests]
    predictions = [float(p)
                   for p in model.predict(pd.DataFrame(features_list))]
    return {
        "model": MODEL_VERSION_FQN,
        "predictions": predictions,
    }
