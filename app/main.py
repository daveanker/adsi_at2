from fastapi import FastAPI
from starlette.responses import JSONResponse
#from joblib import load #only if model was saved using joblib
import pandas as pd
import torch
import torchvision
from torchvision import models
from torch import nn
from torch.nn import functional as F

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

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer prediction is all ready to go!'

@app.get("/model/architecture")
def architecture():
    model = models.densenet121(pretrained=True)
    #model = torch.load("../models/beer_pred.pt")
    return model

def format_features(brewery_name: Optional[str]=None, review_aroma: Optional[int]=None, review_appearance: Optional[int]=None, review_palate: Optional[int]=None, review_taste: Optional[int]=None):
  return {
        'Brewery': [brewery_name],
        'Aroma (1-5)': [review_aroma],
        'Appearance (1-5)': [review_appearance],
        'Palate (1-5)': [review_palate],
        'Taste (1-5)': [review_taste]
    }

@app.get("/beer/type/")
def predict(brewery_name: str, review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    model = torch.load("../models/beer_pred.pt")
    pred = model.predict(obs)
    return JSONResponse(pred.tolist())

#@app.get("/beers/type/")


