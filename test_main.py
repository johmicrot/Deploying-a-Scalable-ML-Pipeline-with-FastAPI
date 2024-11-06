# test_main.py

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_root():
    """
    Test the GET method on the root endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API!"}

def test_post_predict_below_50k():
    """
    Test the POST method on /predict endpoint for a prediction of <=50K.
    """
    data = {
        "age": 28,
        "workclass": "Private",
        "fnlgt": 338409,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Single",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "Asian-Pac-Islander",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "India"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"

def test_post_predict_above_50k():
    """
    Test the POST method on /predict endpoint for a prediction of >50K.
    """
    data = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 123011,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"
