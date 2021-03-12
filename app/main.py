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
from torchsummary import summary
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from src.models.pytorch import PytorchMultiClass
#from src.models.pytorch import get_device

app = FastAPI()
model = torch.load('./models/beer_pred.pt')

@app.get("/")
def read_root():
    return 'Objectives: X'\
    'Endpoints: '\
    'Expected input parameters: '\
    'Output format: '\
    'Github repo: '

@app.get("/health", status_code=200)
def healthcheck():
    return 'Beer prediction app is ready to go.'

@app.get("/model/architecture")
def architecture():
    return summary(model, (1000, 18))

@app.get("/beer/type/")
def predict \
    (brewery_name: str=Query(..., description='Brewery name'),
    review_aroma: float=Query(..., description='Aroma score'),
    review_appearance: float=Query(..., description='Appearance score'),
    review_palate: float=Query(..., description='Palate score'),
    review_taste: float=Query(..., description='Taste score')):

    input_df = pd.DataFrame({'brewery_name': [brewery_name],
                       'review_aroma': [review_aroma],
                       'review_appearance': [review_appearance],
                       'review_palate': [review_palate],
                       'review_taste': [review_taste]})
    
    pipe = load('./models/be_sc.joblib')
    input_df = pipe.transform(input_df)
    df_tensor = torch.Tensor(np.array(input_df))
    pred_num = model(df_tensor).argmax(1)
    
    le = load('./models/le.joblib')
    pred_name = le.inverse_transform(pred_num.tolist())[0]
    pred_name
    return JSONResponse(pred_name)

#@app.get("/beers/type/")
#def predict \
#    (brewery_name: List[int]=Query(..., description='Brewery name list'),
#    review_aroma: List[float]=Query(..., description='Aroma score list'),
#    review_appearance: List[float]=Query(..., description='Appearance score list'),
#    review_palate: List[float]=Query(..., description='Palate score list'),
#    review_taste: List[float]=Query(..., description='Taste score list')):
