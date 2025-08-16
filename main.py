from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, Azure Free Tier!"}

@app.get("/ping")
def ping():
    return {"status": "alive"}
