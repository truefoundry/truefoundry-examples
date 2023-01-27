import logging
import os
from typing import List, Optional

import mlfoundry as mlf
from pydantic import BaseModel
from starlette.responses import RedirectResponse 
from servicefoundry.service import fastapi

app = fastapi.app()

CLASS_NAMES = ['no', 'yes']

_model = None


def _get_model():
    global _model
    if _model is None:
        api_key = os.environ.get('TFY_API_KEY')
        run_id = os.environ.get('TFY_RUN_ID')
        client = mlf.get_client(api_key=api_key)
        run = client.get_run(run_id)
        _model = run.get_model()
    return _model


class Payload(BaseModel):
    instances: List[List[float]]

class Prediction(BaseModel):
    no: float
    yes: float


class Response(BaseModel):
    success: bool
    error: Optional[str]
    predictions: List[Prediction]


@app.get('/')
async def docs_redirect():
    return RedirectResponse(url='/docs')


@app.post('/predict', response_model=Response)
async def predict(payload: Payload):
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