import json
import queue
import threading

import gradio as gr
import requests

PRETRAINED = "pretrained"
FINETUNED = "finetuned"

MODEL_TO_CONFIG = {
    "pythia-70m": {
        PRETRAINED: "https://pythia-70m-llm-demo.demo2.truefoundry.tech/v2/models/pythia-70m/infer",
        FINETUNED: "https://pythia-70m-finetune-llm-demo.demo2.truefoundry.tech/v2/models/finetuned-pythia-70m-2023-06-07T15-19-18/infer",
        "generation_config": {
            "min_new_tokens": 20,
            "max_new_tokens": 150,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 500,
        },
    },
    "pythia-1b": {
        PRETRAINED: "https://pythia-1b-llm-demo.demo2.truefoundry.tech/v2/models/pythia-1b/infer",
        FINETUNED: "https://pythia-1b-finetune-llm-demo.demo2.truefoundry.tech/v2/models/finetuned-pythia-1b-2023-06-07T16-14-45/infer",
        "generation_config": {
            "min_new_tokens": 20,
            "max_new_tokens": 300,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 500,
        },
    },
}

_EXAMPLES = [
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nCalculate the average value from the given numbers\n\n### Input:\n2, 4, 6, 9\n\n### Response:\n",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nGive me two examples of a type of bird.\n\n### Response:\n",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nDescribe the security risks of using public wifi networks.\n\n### Response:\n",
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nSuggest one use case for a robot assistant in a doctor's office.\n\n### Response:\n",
]


EXAMPLES = [[model, example] for model in MODEL_TO_CONFIG for example in _EXAMPLES]


def send_request(model_name, model_type, input_text, output_box):
    inputs = [{"name": "array_inputs", "shape": [1], "datatype": "BYTES", "data": [input_text]}]
    for key, value in MODEL_TO_CONFIG[model_name]["generation_config"].items():
        inputs.append(
            {
                "name": key,
                "shape": [1],
                "datatype": "BYTES",
                "data": [json.dumps(value)],
                "parameters": {"content_type": "hg_json"},
            }
        )
    payload = {"inputs": inputs}
    response = requests.post(MODEL_TO_CONFIG[model_name][model_type], json=payload)
    response.raise_for_status()
    print(response.json())
    output_json = response.json()["outputs"][0]["data"][0]
    output_dict = json.loads(output_json)
    output_str = output_dict[0]["generated_text"][len(input_text) :]
    output_box.put(output_str)


def parallel_requests(model_name, input_text):
    output_box1 = queue.Queue()
    output_box2 = queue.Queue()

    thread1 = threading.Thread(target=send_request, args=(model_name, PRETRAINED, input_text, output_box1))
    thread2 = threading.Thread(target=send_request, args=(model_name, FINETUNED, input_text, output_box2))

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

    return output_box1.get(), output_box2.get()


def predict(model_name: str, input_text: str):
    result1, result2 = parallel_requests(model_name, input_text)
    return result1, result2


inputs = [
    gr.inputs.Dropdown(
        label="Pick model",
        choices=[
            "pythia-70m",
            "pythia-1b",
        ],
        default="pythia-70m",
    ),
    gr.inputs.Textbox(
        placeholder="Enter your prompt here",
        default="",
    ),
]
outputs = [
    gr.outputs.Textbox(label="PreTrained"),
    gr.outputs.Textbox(label="Finetuned"),
]

iface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title=f"PreTrained vs Finetuned",
    examples=EXAMPLES,
)
iface.launch(server_name="0.0.0.0", server_port=8080)
