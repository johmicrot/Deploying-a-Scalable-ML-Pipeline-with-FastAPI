import requests
import json

# Define the API endpoint
url = "https://render-deployment-example-in7h.onrender.com/predict/"

# Define the payload (the data to be sent in the POST request)
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# Make the POST request with JSON payload
response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(data))

# Print the response result and status code
print("Response JSON:", response.json())
print("Status Code:", response.status_code)
