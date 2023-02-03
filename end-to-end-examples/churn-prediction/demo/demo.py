import gradio as gr
import datetime
import os
import requests
from urllib.parse import urljoin

MODEL_DEPLOYED_URL = os.environ['MODEL_DEPLOYED_URL']

inputs = [
    gr.Number(label="CreditScore", value=619),
    gr.Number(label="Age", value=42),
    gr.Number(label="Tenure", value=2),
    gr.Number(label="Balance", value=0),
    gr.Number(label="NumOfProducts", value=1),
    gr.Number(label="HasCrCard", value=1),
    gr.Number(label="IsActiveMember", value=1),
    gr.Number(label="EstimatedSalary", value=101348.88)
]


def predict(*val):
    json_body = {"parameters": {
        "content_type": "pd"
    }, "inputs": []}

    for v, inp in zip(val, inputs):
        json_body["inputs"].append(
            {
                "name": inp.label,
                "datatype": "FP32",
                "data": [v],
                "shape": [1]
            }
        )
    resp = requests.post(
        url=urljoin(MODEL_DEPLOYED_URL, "v2/models/churn-model/infer"),
        json=json_body
    )
    r = resp.json()
    return [ r["outputs"][0]["data"][0],  r["model_version"]]


desc = f"""## Demo Deployed at {datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")}"""

app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=[gr.Textbox(label="Churn"), gr.Textbox(label="Model Version")],
    description=desc,
    title="Churn Predictor",
)
app.launch(server_name="0.0.0.0", server_port=8080)