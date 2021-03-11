from fastapi import FastAPI
from fastapi import Query
from starlette.responses import JSONResponse
#from joblib import load
import pandas as pd
import torch
import torchvision
from torchvision import models
from torchsummary import summary
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from numpy import argmax

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 32)
        self.layer_2 = nn.Linear(32,16)
        self.layer_3 = nn.Linear(16,8)
        self.layer_out = nn.Linear(8, 104)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = F.dropout(F.relu(self.layer_2(x)), training=self.training)
        x = F.dropout(F.relu(self.layer_3(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)

app = FastAPI()

@app.get("/")
def read_root():
    return 'Objectives: X'\
    'Endpoints: '\
    'Expected input parameters: '\
    'Output format: '\
    'Github repo: '

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer prediction app is ready to go.'

@app.get("/model/architecture")
def architecture():
    #model = models.densenet121(pretrained=True)
    model = torch.load("./models/beer_pred.pt")
    return summary(model, (1000,5))

def format_features(brewery_name: str, review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
  return {
        'Brewery': [brewery_name],
        'Aroma (1-5)': [review_aroma],
        'Appearance (1-5)': [review_appearance],
        'Palate (1-5)': [review_palate],
        'Taste (1-5)': [review_taste]
    }

@app.get("/beer/type/")
def predict \
    (brewery_name: int=Query(..., description='Brewery name'),
    review_aroma: float=Query(..., description='Aroma score'),
    review_appearance: float=Query(..., description='Appearance score'),
    review_palate: float=Query(..., description='Palate score'),
    review_taste: float=Query(..., description='Taste score')):
    features = [brewery_name, review_aroma, review_appearance, review_palate, review_taste]
    # convert row to data
    features = Tensor([features])
    # make prediction
    model = torch.load("./models/beer_pred.pt")
    #model = models.densenet121(pretrained=True)
    yhat = model(features)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return argmax(yhat)
    #return JSONResponse(pred.tolist())

#@app.get("/beers/type/")
#def predict \
#    (brewery_name: List[int]=Query(..., description='Brewery name list'),
#    review_aroma: List[float]=Query(..., description='Aroma score list'),
#    review_appearance: List[float]=Query(..., description='Appearance score list'),
#    review_palate: List[float]=Query(..., description='Palate score list'),
#    review_taste: List[float]=Query(..., description='Taste score list')):
