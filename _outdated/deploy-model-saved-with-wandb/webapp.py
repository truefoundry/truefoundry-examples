import gradio as gr
from tensorflow.keras.models import load_model
import wandb

# restore the model file "model.h5" from a specific run by user "sri_rad"
# in project "save_and_restore" from run "3edbaise"
best_model = wandb.restore('model.h5', run_path="sri_rad/save_and_restore/3uys8fy6")
reconstructed_model = load_model(best_model.name)

LABELS = ["T-shirt/top","Trouser","Pullover","Dress","Coat", "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

def classify_image(image):
    image = image.reshape(1, 28, 28)  # add a batch dimension
    image = image.astype('float32') /255.0
    prediction = reconstructed_model.predict(image).tolist()[0] # [0] because prediction only one image
    return {LABELS[i]: prediction[i] for i in range(len(prediction))}

gr.Interface(fn=classify_image,
             inputs=gr.Image(shape=(28, 28), image_mode="L"),
             outputs=gr.Label(num_top_classes=3)).launch(server_name="0.0.0.0", server_port=8080)
