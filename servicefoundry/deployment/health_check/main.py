from fastapi import FastAPI

app = FastAPI()


@app.get("/livez")
def liveness():
    return True


@app.get("/readyz")
def readyness():
    return False


@app.get("/")
async def root():
    return {"message": "Hello World"}
