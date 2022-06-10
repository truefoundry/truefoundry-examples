import os

import gradio as gr
import mlfoundry as mlf

MLF_API_KEY = os.environ.get('TFY_API_KEY')
MLF_RUN_ID = os.environ.get('TFY_RUN_ID')

mlf_api = mlf.get_client(api_key=MLF_API_KEY)
run = mlf_api.get_run(MLF_RUN_ID)
model = run.get_model()

def recognize_digit(image):
    image = image.reshape(1, 28, 28, 1)  # add a batch dimension
    prediction = model.predict(image).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

gr.Interface(fn=recognize_digit,
             inputs="sketchpad",
             outputs=gr.outputs.Label(num_top_classes=3),
             title="MNIST Sketchpad").launch(server_name="0.0.0.0");