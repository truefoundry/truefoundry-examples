from pydantic import BaseModel
from fastapi import FastAPI
from functions import getGeneratedSummary, generate_answers

app = FastAPI()

class RequestBody(BaseModel):
    context: str
    question: str

class Response(BaseModel):
    answer: str
    tokens_consumed: int

@app.post("/get-ans")
def process_text(request_body: RequestBody, compress: bool = False):
    summary = request_body.context
    if compress == True:
        summary = getGeneratedSummary(request_body.context)
    questions = []
    tokens_consumed = len(summary.split())
    questions.append(request_body.question)
    answer=generate_answers(summary, questions=questions)
    return Response(answer=str(answer[0]['answer']), tokens_consumed=tokens_consumed)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
