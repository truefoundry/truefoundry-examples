FEW_SHOT_SUMMARY_TEMPLATE = """

    Based on the examples provided, you need to do a summarization task for a context text with a few guidelines- 

    1. Ensure that almost no information is lost. If it means you are not able to summarize too much, that's fine too. 
    2. The summary does not have to be grammatically correct. 
    3. The summary does not have to be human readable. 
    4. The final size of context should not be less than 50 percent of the original size of the context.

    Generate a summary which preserves the full meaning but minimizes the number of tokens.  If somebody asks almost any question from the text, the summary should be able to answer as well with the same correctness. 
    Don't give any explanations. As an output, just give the summarized text and don't use keywords like Summary as a prefix. You should just output the final summarized text. Remove new lines. 

    Here's the context- {context}

    """

GENERATE_ANSWERS_TEMPLATE = """
Given a text as context, and a set of questions, generate a response for each of the questions. Make sure the answers 
are from the context itself and not from the general knowledge. Don't give any answers that are not part of the context.
Return your response in the following format. 

It should be a list of well formatted jsons with exactly two keys and no new line or space characters.
First key being a question which should have the given question and second key should be an 
answer with the generated answer to that question. Example format of response: 
[{{"question":"question1","answer":"answer1"}},
{{"question":"question2","answer":"answer2"}}]. 

When the response is loaded as a json it should be
a list of python dictionaries with two keys each called prompt and completion. Context text: {context}
In the output, escape any characters in the string that will prevent you from creating a perfect json.
Please note- this is critical and without it your response will have no meaning to me. 
Do not include any explanations, only provide a  RFC8259 compliant JSON response 
following this format without deviation.

Context text- {context}
Here's the set of questions provided as a list of strings- {questions}
"""

GENERATE_SUMMARY_TEMPLATE = """
Summarize the following context and only return the summary and nothing else:
{context}
"""
