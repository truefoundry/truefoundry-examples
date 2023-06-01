import json
from constants import FEW_SHOT_SUMMARY_TEMPLATE, GENERATE_ANSWERS_TEMPLATE
import os
import pandas as pd
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.llms import OpenAI
import openai 

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


GENERATE_ANSWERS_PROMPT = PromptTemplate(
    input_variables=["context", "questions"],
    template=GENERATE_ANSWERS_TEMPLATE,
)

def getGeneratedSummary(context):
    new_df = pd.read_csv('summary_excel.csv', encoding='ISO-8859-1')
    examples = new_df.to_dict('records')
    example_template="text: {text}\n summary: {summary} \n explanation: {explanation}"
    example_prompt = PromptTemplate(input_variables=["text", "summary", "explanation"], template=example_template)
    TEMPLATE = FEW_SHOT_SUMMARY_TEMPLATE

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples[:6], 
        example_prompt=example_prompt, 
        suffix=TEMPLATE, 
        input_variables=["context"]
    )


    def generate_summary_from_few_shots(context, model_name='gpt-3.5-turbo'):
        response = ''
        try:
            llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.9, 
                        max_tokens=-1, model_name=model_name)
            chain = LLMChain(llm=llm, prompt=few_shot_prompt)
            response = chain.run(context=context)
        except Exception as e:
            print("Error happened in generate_summary_text: ", response)
            print("Error message in generate_summary_text: ", e)
        return response


    generated_summary = generate_summary_from_few_shots(context, model_name='gpt-3.5-turbo')
    return generated_summary

def get_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt}
        ]

    )
    return response

def generate_answers(context, questions):
    response = get_response(GENERATE_ANSWERS_PROMPT.format(context=context, questions=questions))
    reply = json.loads(response['choices'][0]['message']['content'])
    return reply[0]['answer'], response['usage']['total_tokens']

