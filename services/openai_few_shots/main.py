from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.llms import OpenAI
os.environ["OPENAI_API_KEY"] = "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"

app = FastAPI()

class TextContext(BaseModel):
    text: str


def getGeneratedSummary(context):
    new_df = pd.read_csv('summary_excel.csv', encoding='ISO-8859-1')
    examples = new_df.to_dict('records')
    example_template="text: {text}\n summary: {summary} \n explanation: {explanation}"
    example_prompt = PromptTemplate(input_variables=["text", "summary", "explanation"], template=example_template)
    FEW_SHOT_SUMMARY_TEMPLATE = """

    Based on the examples provided, you need to do a summarization task for a context text with a few guidelines- 

    1. Ensure that almost no information is lost. If it means you are not able to summarize too much, that's fine too. 
    2. The summary does not have to be grammatically correct. 
    3. The summary does not have to be human readable. 

    Generate a summary which preserves the full meaning but minimizes the number of tokens.  If somebody asks almost any question from the text, the summary should be able to answer as well with the same correctness. 
    Don't give any explanations. As an output, just give the summarized text and don't use keywords like Summary as a prefix. You should just output the final summarized text. Remove new lines. 

    Here's the context- {context}

    """

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples[:6], 
        example_prompt=example_prompt, 
        suffix=FEW_SHOT_SUMMARY_TEMPLATE, 
        input_variables=["context"]
    )


    def generate_summary_from_few_shots(context, model_name='gpt-3.5-turbo'):
        response = ''
        try:
            llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0.9, 
                        max_tokens=-1, model_name=model_name)
            chain = LLMChain(llm=llm, prompt=few_shot_prompt)
            response = chain.run(context=context)
        except Exception as e:
            print("Error happened in generate_summary_text: ", response)
            print("Error message in generate_summary_text: ", e)
        return response


    generated_summary = generate_summary_from_few_shots(context, model_name='gpt-3.5-turbo')
    return generated_summary

@app.post("/process_text")
def process_text(text_context: TextContext):
    response = getGeneratedSummary(text_context.text)
    return {"result": response}
