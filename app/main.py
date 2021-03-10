from fastapi import FastAPI
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
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer prediction is all ready to go!'

@app.get("/model/architecture")
def architecture():
    #model = models.densenet121(pretrained=True)
    model = torch.load("../models/beer_pred.pt", encoding = 'ascii')
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
def predict(brewery_name: int=None, review_aroma: int=None, review_appearance: int=None, review_palate: int=None, review_taste: int=None):
    features = [brewery_name, review_aroma, review_appearance, review_palate, review_taste]
    # convert row to data
    features = Tensor([features])
    # make prediction
    model = torch.load("../models/beer_pred.pt", encoding = 'ascii')
    #model = models.densenet121(pretrained=True)
    yhat = model(features)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return argmax(yhat)
    #return JSONResponse(pred.tolist())

#@app.get("/beers/type/")


