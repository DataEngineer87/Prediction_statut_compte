#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests

# URL locale de l'API
url = "http://127.0.0.1:8000/predict"

# Exemple de données utilisateur (adapter aux valeurs attendues)
data = {
    "gender": "Male",
    "marital_status": "Single",
    "employment_status": "Employed",
    "education_level": "Bachelor",
    "country": "France",
    "subscription_type": "Premium",
    "age_group": "26-35",
    "number_of_children": 2,
    "children_per_age": 0.5,
    "log_annual_income": 10.5
}

response = requests.post(url, json=data)

# Afficher la réponse
print("Status code:", response.status_code)
print("Résultat:", response.json())

