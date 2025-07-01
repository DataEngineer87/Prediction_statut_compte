from fastapi.testclient import TestClient
from Scripts.fast_api import app  # importe ton app FastAPI

client = TestClient(app)

def test_prediction_api():
    data = {
        "gender": "Male",
        "marital_status": "Single",
        "employment_status": "Employed",
        "education_level": "Bachelor",
        "number_of_children": 2,
        "country": "France",
        "subscription_type": "Premium",
        "age_group": "35-44",
        "children_per_age": 0.5,
        "log_annual_income": 10.5
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "prediction" in response.json()

