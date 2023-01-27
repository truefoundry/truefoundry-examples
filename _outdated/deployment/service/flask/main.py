from flask import Flask, request, abort, jsonify
from inference import t5_model_inference

app = Flask(__name__)


@app.get("/infer")
def infer():
    input_text = request.args.get("input_text")
    if input_text is None:
        abort(400, "'input_text' is a required query param.")
    output_text = t5_model_inference(input_text=input_text)
    return jsonify({"output": output_text})
