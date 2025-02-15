from fastapi import FastAPI, Query
from typing import List
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
import numpy as np
from numpy import argmax
from category_encoders.binary import BinaryEncoder
from sklearn.preprocessing import StandardScaler
import torch
from torch import Tensor
from src.models.pytorch import PytorchMultiClass
from src.utils.misc import capture

app = FastAPI()
model = torch.load('./models/beer_pred_cpu.pt')
model.eval()

@app.get("/")
def read_root():
    project_description = {
        'Objectives': 'The objective of the project is to accurately predict a beer type based on brewery name and rating criteria (aroma, appearance, palate, taste).',
        'Endpoints': '/ (GET): Overview, /health (GET): Health check, /beer/type (POST): Predict single beer, /beers/type (POST): Predict multiple beers, /model/architecture (GET): Display model architecture).',
        'Expected input parameters': '/beer/type requires the following parameters: brewery_name (string), review_aroma (int), review_appearance (int), review_palate (int), review_taste (int). /beers/type requires the same parameters, each as a list.',
        'Output format': 'Each endpoint prints a list or dictionary of results, with the exception of /model/architecture which prints stored stdout from the print(model) command.',
        'Github repo': 'github.com/daveanker/adsi_at2' 
    }
    return JSONResponse(project_description)

@app.get("/health", status_code=200)
def health_check():
    return 'Beer prediction app is ready to go.'

@app.post("/beer/type/")
def predict_single \
    (brewery_name: str=Query(..., description='Brewery name'),
    review_aroma: float=Query(..., description='Beer aroma rating (scale 1-5, in 0.5 increments)'),
    review_appearance: float=Query(..., description='Beer appearance rating (scale 1-5, in 0.5 increments)'),
    review_palate: float=Query(..., description='Beer palate rating (scale 1-5, in 0.5 increments)'),
    review_taste: float=Query(..., description='Beer taste rating (scale 1-5, in 0.5 increments)')):

    input_df = pd.DataFrame({'brewery_name': [brewery_name],
                       'review_aroma': [review_aroma],
                       'review_appearance': [review_appearance],
                       'review_palate': [review_palate],
                       'review_taste': [review_taste]})
    
    be_sc = load('./models/be_sc.joblib')
    input_df = be_sc.transform(input_df)
    input_tensor = torch.Tensor(np.array(input_df))
    pred_num = model(input_tensor).argmax(1)
    
    le = load('./models/le.joblib')
    pred_name = le.inverse_transform(pred_num.tolist())[0]

    return JSONResponse(pred_name)

@app.post("/beers/type/")
def predict_multiple \
    (brewery_name: List[str]=Query(..., description='List of beer brewery names'),
    review_aroma: List[float]=Query(..., description='List of beer aroma ratings (scale 1-5, in 0.5 increments)'),
    review_appearance: List[float]=Query(..., description='List of beer appearance ratings (scale 1-5, in 0.5 increments)'),
    review_palate: List[float]=Query(..., description='List of beer palate ratings (scale 1-5, in 0.5 increments)'),
    review_taste: List[float]=Query(..., description='List of beer taste ratings (scale 1-5, in 0.5 increments)')):

    input_df = pd.DataFrame({'brewery_name': brewery_name,
                       'review_aroma': review_aroma,
                       'review_appearance': review_appearance,
                       'review_palate': review_palate,
                       'review_taste': review_taste})

    be_sc = load('./models/be_sc.joblib')
    input_df = be_sc.transform(input_df)
    input_tensor = torch.Tensor(np.array(input_df))
    pred_nums = model(input_tensor).argmax(1)
    
    le = load('./models/le.joblib')
    pred_names = list(le.inverse_transform(pred_nums.tolist()))

    return JSONResponse(pred_names)

@app.get("/model/architecture")
def display_architecture():
    with capture() as output:
        print(model)
    return output
