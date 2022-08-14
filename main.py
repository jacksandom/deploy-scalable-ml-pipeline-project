# Put the code for your API here.
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data

# Read model files
model = pd.read_pickle(r"starter/model/model.pkl")
Encoder = pd.read_pickle(r"starter/model/encoder.pkl")
LB = pd.read_pickle(r"starter/model/lb.pkl")

# Initialise FastAPI
app = FastAPI()

# Pydantic model
class DataIn(BaseModel):
    age: int = 39
    workclass: str = "State-gov"
    fnlgt: int = 77516
    education: str = "Bachelors"
    education_num: int = 13
    marital_status: str = "Never-married"
    occupation: str = "Adm-clerical"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = 2174
    capital_loss: int = 0
    hours_per_week: int = 40
    native_country: str = "United-States"

class DataOut(BaseModel):
    prediction: str = "Income <= 50k"

# Welcome on root domain
@app.get("/")
async def root():
    return {"Hello": "Welcome to the Model :)"}

# Model inference
@app.post("/predict", response_model=DataOut, status_code=200)
def get_prediction(payload: DataIn):
    # Reading data in
    age = payload.age
    workclass = payload.workclass
    fnlgt = payload.fnlgt
    education = payload.education
    education_num = payload.education_num
    marital_status = payload.marital_status
    occupation = payload.occupation
    relationship = payload.relationship
    race = payload.race
    sex = payload.sex
    capital_gain = payload.capital_gain
    capital_loss = payload.capital_loss
    hours_per_week = payload.hours_per_week
    native_country = payload.native_country

    # Convert to dataframe
    df = pd.DataFrame([{"age": age,
                        "workclass": workclass,
                        "fnlgt": fnlgt,
                        "education": education,
                        "education-num": education_num,
                        "marital-status": marital_status,
                        "occupation": occupation,
                        "relationship": relationship,
                        "race": race,
                        "sex": sex,
                        "capital-gain": capital_gain,
                        "capital-loss": capital_loss,
                        "hours-per-week": hours_per_week,
                        "native-country": native_country}])

    # Run inference
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_inference, y_inference, encoder, lb = process_data(
        df, categorical_features=cat_features, label=None, training=False, encoder=Encoder, lb=LB
    )

    pred = inference(model, X_inference)

    # Generate API response
    if pred == 0:
        pred = "Income <= 50k"
    elif pred == 1:
        pred = "Income > 50k"

    response = {"prediction": pred}
    return response
