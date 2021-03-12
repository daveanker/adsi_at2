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
from io import StringIO
import sys
from src.models.pytorch import PytorchMultiClass

app = FastAPI()
model = torch.load('./models/beer_pred.pt')
model.eval()

@app.get("/")
def read_root():
    project_description = {
        'Objectives': 'Objective is to...',
        'Endpoints': 'X',
        'Expected input parameters': 'X',
        'Output format': 'X',
        'Github repo': 'github.com/daveanker/adsi_at2' 
    }
    return JSONResponse(project_description)

@app.get("/health", status_code=200)
def health_check():
    return 'Beer prediction app is ready to go.'

@app.post("/beer/type/")
def predict_single \
    (brewery_name: str=Query(..., description='Brewery name'),
    review_aroma: float=Query(..., description='Aroma rating (1-5)'),
    review_appearance: float=Query(..., description='Appearance rating (1-5)'),
    review_palate: float=Query(..., description='Palate rating (1-5)'),
    review_taste: float=Query(..., description='Taste rating (1-5)')):

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

    return JSONResponse(pred_name)

@app.post("/beers/type/")
def predict_multiple \
    (brewery_name: List[str]=Query(..., description='Brewery name list'),
    review_aroma: List[float]=Query(..., description='Aroma rating list (1-5)'),
    review_appearance: List[float]=Query(..., description='Appearance rating list (1-5)'),
    review_palate: List[float]=Query(..., description='Palate rating list (1-5)'),
    review_taste: List[float]=Query(..., description='Taste rating list (1-5)')):

    input_df = pd.DataFrame({'brewery_name': brewery_name,
                       'review_aroma': review_aroma,
                       'review_appearance': review_appearance,
                       'review_palate': review_palate,
                       'review_taste': review_taste})

    pipe = load('./models/be_sc.joblib')
    input_df = pipe.transform(input_df)
    df_tensor = torch.Tensor(np.array(input_df))
    pred_nums = model(df_tensor).argmax(1)
    
    le = load('./models/le.joblib')
    pred_names = list(le.inverse_transform(pred_nums.tolist()))

    return JSONResponse(pred_names)

# Function to capture print() output for model architecture
# Source: https://stackoverflow.com/questions/16571150/how-to-capture-stdout-output-from-a-python-function-call
class capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

@app.get("/model/architecture")
def display_architecture():
    with capturing() as output:
        print(summary(model, (1000,18)))
    return output
