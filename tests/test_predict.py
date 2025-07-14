import requests

def test_prediction_api():
    url = "http://127.0.0.1:8000/predict"
    data = {
        "gender": "Male",
        "marital_status": "Single",
        "employment_status": "Employed",
        "education_level": "Bachelor",
        "subscription_type": "Premium",
        "age_group": "35-44",
        "number_of_children": 3,
        "children_per_age": 0.5,
        "log_annual_income": 10.5,
        "country": "France"
    }

    response = requests.post(url, json=data)
    
    assert response.status_code == 200, f"Status code was {response.status_code}"
    json_response = response.json()
    assert "prediction" in json_response, "'prediction' key missing in response"
    prediction = json_response["prediction"]
    assert isinstance(prediction, (int, float, str)), "Unexpected type for prediction"

def test_root_endpoint():
    url = "http://127.0.0.1:8000/"
    response = requests.get(url)
    assert response.status_code == 200
    json_response = response.json()
    assert "message" in json_response

