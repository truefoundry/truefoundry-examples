import gradio as gr
import requests
import threading
import queue
import json
import os

MODEL_NAME = os.environ["MODEL_NAME"]
PRETRAINED_MODEL_URL = os.environ["PRETRAINED_MODEL_URL"]
FINETUNED_MODEL_URL = os.environ["FINETUNED_MODEL_URL"]

def send_request(url, input_text, output_box):
    response = requests.post(
        url,
        json={
            "parameters": {
                "content_type": "str"
            },
            "inputs": [
                {
                    "name": "array_inputs",
                    "shape": [
                        1
                    ],
                    "datatype": "BYTES",
                    "data": [
                        input_text
                    ]
                }
            ]
        }
    )
    print(response.json())
    output_json = response.json()["outputs"][0]["data"][0]
    output_dict = json.loads(output_json)
    output_str = output_dict[0]["generated_text"][len(input_text):]
    output_box.put(output_str)

def parallel_requests(input_text):
    output_box1 = queue.Queue()
    output_box2 = queue.Queue()

    thread1 = threading.Thread(target=send_request, args=(PRETRAINED_MODEL_URL, input_text, output_box1))
    thread2 = threading.Thread(target=send_request, args=(FINETUNED_MODEL_URL, input_text, output_box2))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    return output_box1.get(), output_box2.get()

def predict(input_text):
    result1, result2 = parallel_requests(input_text)
    return result1, result2


inputs = gr.inputs.Textbox(
    placeholder="Enter your prompt here",
    default="",   
)
outputs = [
    gr.outputs.Textbox(label="Pre-Trained"),
    gr.outputs.Textbox(label="Finetuned"),
]

iface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title=f"{MODEL_NAME} Pre-Trained vs Finetuned",
    examples=[
        "Input Prompt 1",
        "Input Prompt 2",
        "Input Prompt"]
    )
iface.launch(server_name="0.0.0.0", server_port=8080)