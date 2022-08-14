import requests
import json

data = {
    "age": 30,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Masters",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "Asian-Pac-Islander",
    "sex": "Male",
    "capital_gain": 10000,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}
response = requests.post('https://salary-predictor-udacity-jss.herokuapp.com/predict/', data=json.dumps(data))

print(response.status_code)
print(response.json())
