import logging
import os
from typing import List, Dict, Any, Tuple, Optional

import mlfoundry as mlf
from fastapi import Body
from pydantic import BaseModel
from starlette.responses import RedirectResponse 
from fastapi import FastAPI


logger = logging.getLogger(__name__)

app = FastAPI(docs_url="/")



_model = None
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']


def _get_model():
    global _model
    if _model is None:
        api_key = os.environ.get('TFY_API_KEY')
        host=os.environ.get('TFY_HOST')
        model_version_fqn = os.environ.get('TFY_MODEL_VERSION_FQN')
        client = mlf.get_client(tracking_uri=host, api_key=api_key)
        _model = client.get_model(model_version_fqn).load()
        
    return _model


class Payload(BaseModel):
    instances: List[Tuple[float, float, float, float]]
    class Config:
        schema_extra = {
            "example": {
                "instances": [[1, 1, 1, 1]]
            }
        }


class Prediction(BaseModel):
    setosa: float
    versicolor: float
    virginica: float


class Response(BaseModel):
    success: bool
    error: Optional[str]
    predictions: List[Prediction]


@app.get('/')
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.post('/predict', response_model=Response)
async def predict(payload: Payload):
    """
    Predicts the iris plant for each instance in given list of instances.
    Each instance is made of four float numbers corresponding to features
    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    """
    try:
        model = _get_model()
        instances = payload.instances
        predictions = []
        if instances:
            _predictions = model.predict_proba(instances).tolist()
            predictions = [dict(zip(CLASS_NAMES, prediction)) for prediction in _predictions]
        return {"success": True, "error": None, "predictions": predictions}
    except Exception as e:
        logging.exception("failed to serve request")
        return {"success": False, "error": str(e), "predictions": []}

