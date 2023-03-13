import random

import gradio as gr
import torch

from pipeline import get_txt2img_pipeline

state = None
current_steps = 25
modes = {
    'txt2img': 'Text to Image',
}
current_mode = modes['txt2img']


def error_str(error, title="Error"):
    return f"""#### {title} {error}""" if error else ""


def update_state(new_state):
    global state
    state = new_state


def update_state_info(old_state):
    if state and state != old_state:
        return gr.update(value=state)


def pipe_callback(step: int, timestep: int, latents: torch.FloatTensor):
    # \nTime left, sec: {timestep/100:.0f}")
    update_state(f"{step}/{current_steps} steps")


def inference(prompt, n_images, guidance, steps, width=768, height=768, seed=0, neg_prompt=""):
    if seed == 0:
        seed = random.randint(0, 2147483647)

    generator = torch.Generator('cuda').manual_seed(seed)
    prompt = prompt

    try:
        return txt_to_img(prompt, n_images, neg_prompt, guidance, steps, width, height, generator, seed), gr.update(visible=False, value=None)
    except Exception as e:
        return None, gr.update(visible=True, value=error_str(e))


def txt_to_img(prompt, n_images, neg_prompt, guidance, steps, width, height, generator, seed):
    update_state(f"Loading Model...")
    pipe = get_txt2img_pipeline()
    result = pipe(
        prompt,
        num_images_per_prompt=n_images,
        negative_prompt=neg_prompt,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
        callback=pipe_callback
    ).images
    update_state(f"Done. Seed: {seed}")
    return result


def on_steps_change(steps):
    global current_steps
    current_steps = steps


def run():
    with gr.Blocks(css="style.css") as demo:
        gr.HTML(
            f"""
            <div class="main-div">
                <div>
                <h1>Stable Diffusion 2.1</h1>
                </div><br>
                <p> 
                    Model used: <a href="https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.ckpt" target="_blank">v2-1_768-ema-pruned.ckpt</a>
                </p>
                Running on 
                <b>
                    {"GPU 🔥" if torch.cuda.is_available() else "CPU 🥶"} 
                </b> 
                with 
                <a href="https://truefoundry.com/", target="_blank"><img style="background: black;max-width: 10%;max-height: 10%;display: inline;" src="https://uploads-ssl.webflow.com/6291b38507a5238373237679/6291e20016f0c749e47497d5_logo-header.png"/></a>
            </div>
            """
        )
        with gr.Row():
            with gr.Column(scale=70):
                with gr.Group():
                    with gr.Row():
                        prompt = gr.Textbox(
                            label="Prompt",
                            show_label=False,
                            max_lines=2,
                            placeholder=f"Enter prompt"
                        ).style(container=False)
                        generate = gr.Button(
                            value="Generate"
                        ).style(rounded=(False, True, True, False))
                    gallery = gr.Gallery(
                        label="Generated images",
                        show_label=False
                    ).style(grid=[2], height="auto")
                state_info = gr.Textbox(
                    label="State",
                    show_label=False,
                    max_lines=2,
                    interactive=False
                ).style(container=False)
                error_output = gr.Markdown(visible=False)

            with gr.Column(scale=30):
                gr.Radio(
                    label="Inference Mode",
                    choices=list(modes.values()),
                    value=modes['txt2img']
                )

                with gr.Group():
                    neg_prompt = gr.Textbox(
                        label="Negative prompt",
                        placeholder="What to exclude from the image"
                    )
                    n_images = gr.Slider(
                        label="Number of images",
                        value=1,
                        minimum=1,
                        maximum=4,
                        step=1
                    )
                    with gr.Row():
                        guidance = gr.Slider(
                            label="Guidance scale",
                            value=7.5,
                            maximum=15
                        )
                        steps = gr.Slider(
                            label="Steps",
                            value=current_steps,
                            minimum=2,
                            maximum=100,
                            step=1
                        )
                    with gr.Row():
                        width = gr.Slider(
                            label="Width",
                            value=768,
                            minimum=64,
                            maximum=1024,
                            step=8
                        )
                        height = gr.Slider(
                            label="Height",
                            value=768,
                            minimum=64,
                            maximum=1024,
                            step=8
                        )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=2147483647,
                        value=0,
                        label='Seed (0 = random)',
                        step=1
                    )

        steps.change(on_steps_change, inputs=[steps], outputs=[], queue=False)
        inputs = [prompt, n_images, guidance,
                  steps, width, height, seed, neg_prompt]
        outputs = [gallery, error_output]
        prompt.submit(inference, inputs=inputs, outputs=outputs)
        generate.click(inference, inputs=inputs, outputs=outputs)

        demo.load(
            update_state_info,
            inputs=state_info,
            outputs=state_info,
            every=0.5,
            show_progress=False
        )

        gr.HTML("""
        <div style="border-top: 1px solid #303030;">
        <br>
        <p>Original App by: <a href="https://twitter.com/hahahahohohe"><img style="display: inline;" src="https://img.shields.io/twitter/follow/hahahahohohe?label=%40anzorq&style=social" alt="Twitter Follow"></a></p><br>
        </div>
        """)
    demo.queue()
    demo.launch(debug=True, height=768, server_name="0.0.0.0", server_port=8080)


if __name__ == '__main__':
    run()
