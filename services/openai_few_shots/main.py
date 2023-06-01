from pydantic import BaseModel
from fastapi import FastAPI
from functions import getGeneratedSummary, generate_answers, generate_summary
import gradio as gr

def process_text(context, question, compress: bool = False):
    tfy_summary = context
    if compress == True:
        tfy_summary = getGeneratedSummary(context)
    questions = [question]
    answer, tokens_consumed = generate_answers(tfy_summary, questions=questions)
    summary, summary_tokens_consumed = generate_summary(tfy_summary)

    tfy_summary_actual = ''
    if compress == True:
        tfy_summary_actual = tfy_summary

    return answer, tokens_consumed, '$'+ str(tokens_consumed * 	0.000002), tfy_summary_actual, summary, summary_tokens_consumed, '$'+ str(summary_tokens_consumed * 	0.000002)


gr.Interface(fn=process_text, inputs=[gr.TextArea(label='Context'), gr.Textbox(label='Questions'), gr.Checkbox(label='Apply context compression')], outputs=[gr.TextArea(label='Answer'), gr.Text(label='Tokens consumed'), gr.Text(label='Cost incurred'), gr.TextArea(label='TrueFoundry summary'), gr.TextArea(label='Chatgpt Summary'), gr.Text(label='Tokens consumed for summary'), gr.Text(label='Cost incurred for summary')]).launch(server_name="0.0.0.0", server_port=8080)
