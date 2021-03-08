from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load #only if model was saved using joblib
import pandas as pd #for data transformation before feeding data into model?
import torch
from torch import nn
from torch.nn import functional as F

app = FastAPI()

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 32)
        self.layer_out = nn.Linear(32, 104) #Number of classes in target variable?
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer prediction is all ready to go!'

@app.get("/model/architecture")
def architecture():

    import torch

    #model = PytorchMultiClass()
    model = torch.load("./models/beer_pred.pt")
    #model.eval()
    print(model)