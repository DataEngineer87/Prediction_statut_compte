import sys
import os

# Ajouter la racine du projet au PYTHONPATH pour que Scripts soit trouvable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Scripts.fast_api import app  # importer l'app FastAPI

import requests

def test_prediction_api():
    url = "http://127.0.0.1:8000/predict"
    data = {
        "gender": "Male",
        "marital_status": "Single",
        "employment_status": "Employed",
        "education_level": "Bachelor",
        "number_of_children": 3,
        "country": "France",
        "subscription_type": "Premium",
        "age_group": "35-44",
        "children_per_age": 0.5,
        "log_annual_income": 10.5
    }

    response = requests.post(url, json=data)
    
    assert response.status_code == 200, f"Status code was {response.status_code}"
    json_response = response.json()
    assert "prediction" in json_response, "'prediction' key missing in response"
    prediction = json_response["prediction"]
    assert isinstance(prediction, (int, float, str)), "Unexpected type for prediction"

