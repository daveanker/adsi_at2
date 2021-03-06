from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load #only if model was saved using joblib
import pandas as pd #for data transformation before feeding data into model?

app = FastAPI()

nn_pipe = load('../models/nn_pipeline.joblib')
nn_pipe.eval()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Beer prediction is all ready to go!'

def format_features(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
  return {
        'Brewery': [brewery_name],
        'Aroma': [review_aroma],
        'Appearance': [review_appearance],
        'Palate': [review_palate],
        'Taste': [review_taste],
    }

@app.post("/beer/type")
def predict(brewery_name: str,	review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
    features = format_features(brewery_name, review_aroma, review_appearance, review_palate, review_taste)
    obs = pd.DataFrame(features)
    pred = nn_pipe.predict(obs)
    return JSONResponse(pred.tolist())

@app.get("/model/architecture")
def architecture():
    print(model) #need to call this accurately


# 'beer/type/' (POST): Returning prediction for single input only
# 'beers/type/' (POST): Returning predictions for multiple inputs
# 'model/architecture/' (GET): Displaying the architecture of your neural network

#model=torch.load(PATH)
#model.eval()
#data = torch.randn(1, 3, 24, 24) # Load data here, this is just dummy data
#output = model(data)
#prediction = torch.argmax(output) #may need to add ', dim=1' after 'output'