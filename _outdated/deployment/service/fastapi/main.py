from fastapi import FastAPI

from inference import t5_model_inference

app = FastAPI(docs_url="/")


@app.get("/infer")
async def infer(input_text: str):
    output_text = t5_model_inference(input_text=input_text)
    return {"output": output_text}
