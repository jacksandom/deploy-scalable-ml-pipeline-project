import json
from starlette.testclient import TestClient
from .main import app

client = TestClient(app)

# Test welcome Get response
def test_get():
    response = client.get("/")
    assert response.status_code == 200

# Test post response for predicted salary <= 50K
def test_post_1():
    input_dict = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["prediction"] == "Income <= 50k"

# Test post response for predicted salary > 50K
def test_post_2():
    input_dict = {
        "age": 60,
        "workclass": "Private",
        "fnlgt": 200000,
        "education": "Masters",
        "education_num": 14,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 100000,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=input_dict)
    assert response.status_code == 200
    assert json.loads(response.text)["prediction"] == "Income > 50k"
