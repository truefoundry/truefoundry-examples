import os
import time
import uuid
from typing import List

import mlfoundry as mlf
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# get the model from `Truefoundry Model Registry`.
MODEL_VERSION_FQN = os.getenv("MLF_MODEL_VERSION_FQN")
client = mlf.get_client()
model_version = client.get_model(MODEL_VERSION_FQN)
model = model_version.load()

# create a fastapi instance
app = FastAPI(root_path=os.getenv("TFY_SERVICE_ROOT_PATH"), docs_url="/")

# creating the inference request object using pydantic
class WinePredictionRequest(BaseModel):
    data_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

    class Config:
        alias_generator = lambda string: string.replace("_", " ")
        allow_population_by_field_name = True

# defining a fastapi endpoint for predictions
@app.post("/predict")
def predict(inference_requests: List[WinePredictionRequest]):
    s = time.time()
    predictions = []
    prediction_logs = []
    # get the data_ids and features from the request
    data_ids_list = [request.data_id for request in inference_requests]
    features_list = [
        request.dict(exclude={"data_id"}, by_alias=True)
        for request in inference_requests
    ]
    # get the predictions and probabilities for the features
    prediction_values = [value for value in model.predict(pd.DataFrame(features_list))]
    prediction_probabilities_list = [
        {pred: float(prob) for pred, prob in zip(model.classes_, prediction_prob)}
        for prediction_prob in model.predict_proba(pd.DataFrame(features_list))
    ]

    # create predictions and predictions log's list
    for data_id, features, value, probabilities in zip(
        data_ids_list, features_list, prediction_values, prediction_probabilities_list
    ):
        # create predictions list
        predictions.append(
            {
                "data_id": data_id,
                "model_version_fqn": MODEL_VERSION_FQN,
                "value": value,
                "probabilities": probabilities,
            }
        )
        # create prediction_logs
        prediction_logs.append(
            # parse the prediction and associated metadata in `mlfoundry.Prediction` format
            mlf.Prediction(
                data_id=data_id,
                features=features,
                prediction_data={
                    "value": value,
                    "probabilities": probabilities,
                },
            )
        )
    e = time.time()
    # log the prediction_logs
    client.log_predictions(
        model_version_fqn=MODEL_VERSION_FQN, predictions=prediction_logs
    )
    print(f"Made {len(prediction_logs)} predictions in {e-s} seconds")
    # return the main predictions list
    return predictions
