import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

if not torch.cuda.is_available():
    raise Exception("GPU devices are required to load and run this model")

model_id = 'stabilityai/stable-diffusion-2-1'
pipe = None

def get_txt2img_pipeline():
    global pipe
    if pipe is None:
        print("Now loading model ...")
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            revision="fp16",
            torch_dtype=torch.float16,
            scheduler=scheduler
        ).to("cuda")
        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention()
    return pipe
