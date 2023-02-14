import logging
import os
import uuid
from timeit import default_timer as timer

from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import GPTJForCausalLM, AutoTokenizer, pipeline

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info(f"CUDA: {torch.cuda.is_available()}")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
generator = None

def get_generator():
    global generator
    if generator is None:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        model = GPTJForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            revision="float16",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).eval().to(device, dtype=torch.float16)
        generator = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device=device
        )
    return generator


app = FastAPI(root_path=os.getenv("TFY_SERVICE_ROOT_PATH"), docs_url="/")


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = 0.9
    max_length: int = 100


@app.post("/generate")
def generate(request: GenerateRequest):
    request_id = str(uuid.uuid4())
    logging.info(f"{request_id} Request Started")
    load_start = timer()
    generator = get_generator()
    load_end = timer()
    load_time = load_end - load_start

    generate_start = timer()
    outputs = generator(
        request.prompt,
        num_return_sequences=1, 
        return_full_text=False,
        do_sample=True,
        temperature=request.temperature,
        max_length=request.max_length,
    )
    generate_end = timer()
    generation_time = generate_end - generate_start

    logging.info(f"{request_id} Request Finished in {generation_time} seconds")

    return {
        "outputs": outputs,
        "time": {
            "load": load_time,
            "generate": generation_time,
        }
    }
