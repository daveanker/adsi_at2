from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load #only if model was saved using joblib
import pandas as pd #for data transformation before feeding data into model?

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer prediction is all ready to go!'