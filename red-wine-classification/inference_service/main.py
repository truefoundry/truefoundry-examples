import os
import uuid
from datetime import datetime, timezone

import mlfoundry as mlf
import pandas as pd
from fastapi import FastAPI

# fetching the model object from the model vesion fqn
MODEL_VERSION_FQN = os.getenv("MLF_MODEL_VERSION_FQN")
client = mlf.get_client()
model_version = client.get_model(MODEL_VERSION_FQN)
model = model_version.load()

app = FastAPI(docs_url="/")


@app.get("/predict")
def predict(
    fixed_acidity: float,
    volatile_acidity: float,
    citric_acid: float,
    residual_sugar: float,
    chlorides: float,
    free_sulfur_dioxide: float,
    total_sulfur_dioxide: float,
    density: float,
    pH: float,
    sulphates: float,
    alcohol: float,
    data_id: str = "",
):
    data = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }
    output_classes = model.classes_
    if not data_id:
        data_id = uuid.uuid4().hex
    prediction = {
        "data_id": data_id,
        "model_version_fqn": MODEL_VERSION_FQN,
        "value": str(model.predict(pd.DataFrame([data]))[0]),
        "probabilities": {
            str(pred): float(prob)
            for pred, prob in zip(
                output_classes, model.predict_proba(pd.DataFrame([data]))[0]
            )
        },
    }

    # logging the predictions
    client.log_predictions(
        model_version_fqn=MODEL_VERSION_FQN,
        predictions=[
            mlf.Prediction(
                data_id=data_id,
                model_version_fqn=MODEL_VERSION_FQN,
                features=data,
                prediction_data={
                    "value": prediction["value"],
                    "probabilities": prediction["probabilities"],
                    "shap_values": {},
                },
                occurred_at=datetime.now(tz=timezone.utc),
                raw_data={},
            )
        ],
    )
    return prediction
