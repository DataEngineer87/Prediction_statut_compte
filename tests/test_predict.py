import requests

# URL locale de l'API
url = "http://127.0.0.1:8000/predict"

# Exemple de données utilisateur
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

print("Status code:", response.status_code)
print("Résultat:", response.json())

# Pour lancer le test FastApi dans le terminal: python test_predict.py

