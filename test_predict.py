import requests

def test_prediction_api():
    url = "http://127.0.0.1:8000/predict"
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

    response = requests.post(url, json=data)
    
    # Vérifier que la requête a réussi
    assert response.status_code == 200, f"Status code was {response.status_code}"

    json_response = response.json()
    
    # Vérifier que la clé "prediction" existe dans la réponse
    assert "prediction" in json_response, "'prediction' key missing in response"

    # Tu peux aussi vérifier que la prédiction a un format attendu
    prediction = json_response["prediction"]
    assert isinstance(prediction, (int, float, str)), "Unexpected type for prediction"

    # Par exemple si tu sais que la prédiction est un booléen ou une valeur spécifique, tu peux ajuster ici


