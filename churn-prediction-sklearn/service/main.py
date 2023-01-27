import os
import logging
from typing import List, Optional

import mlfoundry as mlf
from pydantic import BaseModel
from starlette.responses import RedirectResponse 
from fastapi import FastAPI

app = FastAPI(root_path=os.getenv("TFY_SERVICE_ROOT_PATH"), docs_url="/")

CLASS_NAMES = ['no', 'yes']

_model = None


def _get_model():
    global _model
    if _model is None:
        run_id = os.environ.get('TFY_RUN_ID')
        client = mlf.get_client()
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