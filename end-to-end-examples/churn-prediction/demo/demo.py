import gradio as gr
import datetime
import os
import requests
from urllib.parse import urljoin

# Endpoint of the deployed model
MODEL_DEPLOYED_URL = os.environ['MODEL_DEPLOYED_URL']

# specifying the desired input components
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
    # request body in dictionary format
    json_body = {"parameters": {
        "content_type": "pd"
    }, "inputs": []}

    # add the values into inputs list of json_body
    for v, inp in zip(val, inputs):
        json_body["inputs"].append(
            {
                "name": inp.label,
                "datatype": "FP32",
                "data": [v],
                "shape": [1]
            }
        )
    # use the requests library, post the request and get the response
    resp = requests.post(
        url=urljoin(MODEL_DEPLOYED_URL, "v2/models/churn-model/infer"),
        json=json_body
    )
    
    r = resp.json()
    # return the output and model_version
    return [ r["outputs"][0]["data"][0]]


# setup Gradio Interface
app = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=[gr.Textbox(label="Churn")],
    title="Churn Predictor",
)
# launch the gradio interface
app.launch(server_name="0.0.0.0", server_port=8080)