import gradio as gr

def greet(name):
    return "Hello " + name.capitalize() + "!"

gr.Interface(fn=greet, inputs="text", outputs="text").launch(server_name='0.0.0.0', server_port=8080)