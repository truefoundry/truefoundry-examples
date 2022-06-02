import logging
import os
from typing import Dict, Any

import mlfoundry as mlf
from fastapi import Body
from servicefoundry.service import fastapi

logger = logging.getLogger(__name__)

app = fastapi.app()


_model = None
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']


def _get_model():
    global _model
    if _model is None:
        api_key = os.environ.get('TFY_API_KEY')
        run_id = os.environ.get('TFY_RUN_ID')
        client = mlf.get_client(api_key=api_key)
        run = client.get_run(run_id)
        _model = run.get_model()
    return _model


@app.post('/predict')
async def predict(payload: Dict[str, Any] = Body(...)):
    try:
        model = _get_model()
        instances = payload["instances"]
        predictions = []
        if instances:
            _predictions = model.predict_proba(instances).tolist()
            predictions = [dict(zip(CLASS_NAMES, prediction)) for prediction in _predictions]
        return {"success": True, "predictions": predictions}
    except Exception as e:
        logging.exception("failed to serve request")
        return {"success": False, "error": str(e)}

