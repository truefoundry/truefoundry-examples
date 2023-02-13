import os
from typing import List
import uuid
from datetime import datetime

import mlfoundry as mlf
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI(docs_url="/")


MODEL_VERSION_FQN = os.environ["MODEL_VERSION_FQN"]
client = mlf.get_client()
model = client.get_model(MODEL_VERSION_FQN).load()


class Instance(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class Request(BaseModel):
    instances: List[Instance]

class Response(BaseModel):
    predictions: List[float]


@app.post("/predict", response_model=Response)
def predict(request: Request):
    features = request.dict()["instances"]

    features_df = pd.DataFrame(features)
    predictions = [float(p) for p in model.predict(features_df)]
    client.log_predictions(
        model_version_fqn=MODEL_VERSION_FQN,
        predictions=[
            mlf.Prediction(
                data_id=uuid.uuid4().hex,
                features=feature,
                prediction_data={
                    "value": prediction,
                },
                occurred_at=datetime.utcnow(),
                raw_data={"data": "any_data"},
            )
            for feature, prediction in zip(features, predictions)
        ],
    )
    return Response(predictions=predictions)