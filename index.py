from fastapi import FastAPI
import numpy as np
import joblib 
from pydantic import BaseModel
from sklearn.datasets import load_iris

target_names = load_iris().target_names

app = FastAPI()

class input(BaseModel):
    features: list

md = joblib.load('./mlmodel.joblib')

@app.get('/')
def home():
    return {"message": "API is running"}

@app.get('/about')
def about():
    return {"message": "It is a ML model fastapi. use POST /predict to get the prediction output."}


@app.post('/predict')
def predict(data: input):
    x_input = np.array(data.features).reshape(1, -1)
    prediction = md.predict(x_input)
    return {"prediction": target_names[prediction][0]}