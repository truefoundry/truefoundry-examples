import os
from typing import List
import uuid
from datetime import datetime
import mlfoundry as mlf
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI

# instantiate the FastAPI application via `FastAPI`
app = FastAPI(docs_url="/")

# get the model from `Truefoundry Model Registry`
MODEL_VERSION_FQN = os.environ["MODEL_VERSION_FQN"]
client = mlf.get_client()
model = client.get_model(MODEL_VERSION_FQN).load()

# define pydantic classes for the Instance, Request and Response
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

# defining a fastapi endpoint for predictions
@app.post("/predict", response_model=Response)
# define the path operation function, specifically `predicit` function
def predict(request: Request):
    features = request.dict()["instances"]

    # create a dataframe out of the features dictionary
    features_df = pd.DataFrame(features)
    # get the predictions from the model
    predictions = [float(p) for p in model.predict(features_df)]
    # use the mlfoundry client to log the predictions
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