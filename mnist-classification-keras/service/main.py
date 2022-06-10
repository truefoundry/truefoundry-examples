import os
import numpy as np
import mlfoundry as mlf
from PIL import Image
from io import BytesIO
from fastapi import File, UploadFile
from skimage.transform import resize
from servicefoundry.service import fastapi

app = fastapi.app()
mlf_client = mlf.get_client(api_key=os.environ.get('TFY_API_KEY'))
run = mlf_client.get_run(os.environ.get('TFY_RUN_ID'))
model = run.get_model()

def recognize_digit(image):
    image = image.reshape(1, 28, 28, 1)  # add a batch dimension
    prediction = model.predict(image).tolist()[0]
    return {str(i): prediction[i] for i in range(10)}

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    np_image = load_image_into_numpy_array(await image.read())
    return {'predictions': recognize_digit(resize(np_image, (28, 28, 1)))}