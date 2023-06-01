from pydantic import BaseModel
from fastapi import FastAPI
from functions import getGeneratedSummary, generate_answers
import gradio as gr

def process_text(context, question, compress: bool = False):
    summary = context
    if compress == True:
        summary = getGeneratedSummary(context)
    questions = [question]
    answer, tokens_consumed = generate_answers(summary, questions=questions)
    return answer, tokens_consumed, '$'+ str(tokens_consumed * 	0.000002)


gr.Interface(fn=process_text, inputs=[gr.TextArea(label='Context'), gr.Textbox(label='Context'), gr.Checkbox(label='Apply context compression')], outputs=[gr.TextArea(label='Answer'), gr.Text(label='Tokens consumed'), gr.Text(label='Cost incurred')]).launch()
