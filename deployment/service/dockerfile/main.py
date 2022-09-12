import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")


def t5_model_inference(input_text: str) -> str:
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


gr.Interface(
    fn=t5_model_inference,
    inputs="text",
    outputs="textbox",
    allow_flagging="never",
).launch(server_name="0.0.0.0", server_port=8080)
# NOTE: we need to set `server_name` to `0.0.0.0` so that this process
# can accept requests coming from outside the container.
